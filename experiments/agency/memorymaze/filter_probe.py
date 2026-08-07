"""Phase 2: freeze the filter and measure what its state retains.

Nothing here is trained except the probes, and the probes are closed-form, so
this stage has no optimizer, no seed and no stopping rule of its own. A
difference between two conditions is a difference between the filters.

Four measurements, in increasing order of how much they separate a Tensor Brain
from a recurrent control:

1. **Linear probes** from the state to ground truth the filter never saw:
   the positions of all three targets, the agent's own position, and the vector
   to the current target. This is the benchmark's own protocol, it applies
   unchanged to every architecture, and it is the comparison a control can win.

2. **The memory-horizon curve**, which is the same probe conditioned on how long
   ago the target was last in view. An averaged score cannot tell a filter that
   remembers from one that is reading the current frame; this can.

3. **Native readout** -- the mutual information between the index the model
   spontaneously emits and which target is actually nearest. The index bank is
   never supervised, so this asks whether the model's own vocabulary carved a
   real distinction. A recurrent control emits no symbol, so the quantity does
   not exist for it: not a worse number, no measurement.

4. **Diagnostics** over the same pass, so an ordering can be attributed to
   saturation or to softmax sharpness rather than left as a bare score.

Probes are fit on one split and scored on another, both held out from Phase 1.
Adjacent steps within an episode are strongly correlated, so a random split
across pooled steps would leak and report a number that means nothing.

Masking is applied at the same rate the filter was trained at. Probing a filter
trained at ``rho = 0.9`` under full observation would measure a different system
from the one that was fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from experiments.agency.memorymaze.corpus import OfflineCorpus
from experiments.agency.memorymaze.env import COLOR_NAMES
from experiments.agency.memorymaze.filter import observation_mask
from experiments.agency.memorymaze.filter_diagnostics import (
    DiagnosticAccumulator,
    step_diagnostics,
)
from experiments.agency.memorymaze.horizon import horizon_curve, steps_since_visible, visibility
from experiments.agency.memorymaze.linear_probe import (
    classification_probe,
    mutual_information,
    regression_probe,
    ridge_apply,
    ridge_fit,
)

# Ground truth worth probing, and why each is in the list.
REGRESSION_TARGETS: dict[str, str] = {
    # The memory claim: where all three targets are. Only one is ever the
    # current goal and none is visible from most of the maze.
    "targets_pos": "positions of all three targets",
    # Self-localisation. Recoverable from recent observation alone, so it is the
    # easy quantity and acts as a floor on the probe's sensitivity.
    "agent_pos": "the agent's own position",
    # The quantity most directly useful for acting, and the one a filter with no
    # retention is least able to hold.
    "target_vec": "displacement to the current target",
}
# The three colours plus "no target in view".
NOTHING_VISIBLE = len(COLOR_NAMES)
NUM_COLOR_CLASSES = len(COLOR_NAMES) + 1


@dataclass(frozen=True)
class FilterRecording:
    """Filter states paired with the ground truth they ought to encode."""

    state: Float[Tensor, "samples width"]
    ground_truth: dict[str, Float[Tensor, "samples dims"]]
    nearest_color: Int[Tensor, " samples"]
    gaps: Int[Tensor, "samples targets"]
    # ``None`` for a filter with no index layer, and for the ``soft`` and
    # ``none`` variants, which never draw a symbol.
    index_outcome: Int[Tensor, " samples"] | None
    diagnostics: dict[str, float]

    def __len__(self) -> int:
        return int(self.state.shape[0])


def nearest_visible_color(
    targets_vec: Float[Tensor, "steps targets 2"], visible: Bool[Tensor, "steps targets"]
) -> Int[Tensor, " steps"]:
    """Which coloured target is nearest among those actually in view.

    Conditioning on visibility rather than on distance alone is what makes this
    a perceptual label: a target behind a wall is not something the frame could
    have reported.
    """

    distance = targets_vec.norm(dim=-1)
    masked = distance.masked_fill(~visible, float("inf"))
    nearest = masked.argmin(dim=-1)
    return torch.where(visible.any(dim=-1), nearest, torch.full_like(nearest, NOTHING_VISIBLE))


@torch.no_grad()
def record_filter(
    model,
    corpus: OfflineCorpus,
    *,
    episodes: int,
    start: int = 0,
    mask_probability: float,
    seed: int,
    warmup: int = 32,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> FilterRecording:
    """Replay whole episodes through a frozen filter and keep the states.

    ``warmup`` steps are discarded so the belief is never measured from the
    zeroed initial state, which encodes nothing and would flatter every
    architecture equally.

    Episodes are replayed in batches. One at a time would be a hundred thousand
    sequential single-example forward passes, which is the slowest way to use a
    GPU that exists; the filter is stateful along time only, so batching across
    episodes changes nothing about the result.
    """

    device = device or torch.device("cpu")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    diagnostics = DiagnosticAccumulator()

    states: list[Tensor] = []
    truths: dict[str, list[Tensor]] = {key: [] for key in REGRESSION_TARGETS}
    colors: list[Tensor] = []
    gaps: list[Tensor] = []
    outcomes: list[Tensor] = []

    available = min(episodes, len(corpus) - start)
    for first in range(start, start + available, batch_size):
        chunk = [
            corpus.episode(index)
            for index in range(first, min(first + batch_size, start + available))
        ]
        size = len(chunk)
        images = torch.stack([item["image"] for item in chunk], dim=1).to(device)
        actions = torch.stack([item["action"] for item in chunk], dim=1).to(device)
        steps = actions.shape[0]

        visible = [
            visibility(
                item["agent_pos"], item["targets_pos"], item["targets_vec"], item["maze_layout"]
            )
            for item in chunk
        ]
        chunk_gaps = torch.stack([steps_since_visible(item) for item in visible], dim=1)
        chunk_colors = torch.stack(
            [
                nearest_visible_color(item["targets_vec"], seen)
                for item, seen in zip(chunk, visible, strict=True)
            ],
            dim=1,
        )
        chunk_truth = {
            "targets_pos": torch.stack(
                [item["targets_pos"].reshape(steps, -1) for item in chunk], dim=1
            ),
            "agent_pos": torch.stack([item["agent_pos"] for item in chunk], dim=1),
            "target_vec": torch.stack([item["target_vec"] for item in chunk], dim=1),
        }

        state, context = model.filter.initial_state(size, device)
        for step in range(steps):
            observed = observation_mask(size, mask_probability, generator, device)
            trace = model.filter.step(state, context, images[step], actions[step], observed)
            state, context = trace.q, trace.context
            if step < warmup:
                continue
            states.append(trace.readout.cpu())
            diagnostics.update(step_diagnostics(trace, model.brain))
            if trace.index_outcome is not None:
                outcomes.append(trace.index_outcome.cpu())
            for key in REGRESSION_TARGETS:
                truths[key].append(chunk_truth[key][step])
            colors.append(chunk_colors[step])
            gaps.append(chunk_gaps[step])

    return FilterRecording(
        state=torch.cat(states),
        ground_truth={key: torch.cat(values) for key, values in truths.items()},
        nearest_color=torch.cat(colors),
        gaps=torch.cat(gaps),
        index_outcome=torch.cat(outcomes) if outcomes else None,
        diagnostics=diagnostics.means(),
    )


def probe_all(
    train: FilterRecording,
    test: FilterRecording,
    *,
    num_latent_indices: int,
    penalty: float = 1.0,
) -> dict:
    """Every Phase-2 measurement, over one train/test pair of recordings."""

    results: dict = {"regression": {}, "samples": {"train": len(train), "test": len(test)}}
    for key in REGRESSION_TARGETS:
        results["regression"][key] = regression_probe(
            train.state, train.ground_truth[key], test.state, test.ground_truth[key],
            penalty=penalty,
        )

    results["nearest_color"] = classification_probe(
        train.state, train.nearest_color, test.state, test.nearest_color,
        classes=NUM_COLOR_CLASSES, penalty=penalty,
    )

    # The native readout. Absent rather than zero for a control: a model with no
    # index layer emits no symbol, so there is nothing to score.
    results["native_readout"] = (
        mutual_information(
            test.index_outcome - int(test.index_outcome.min()),
            test.nearest_color,
            num_symbols=num_latent_indices,
            num_labels=NUM_COLOR_CLASSES,
        )
        if test.index_outcome is not None
        else None
    )

    results["horizon"] = memory_horizon(train, test, penalty=penalty)
    results["diagnostics"] = test.diagnostics
    return results


def memory_horizon(
    train: FilterRecording, test: FilterRecording, *, penalty: float = 1.0
) -> list[dict[str, float | str]]:
    """Per-target position error, bucketed by how stale that target is.

    The probe is the same one scored in ``probe_all``; only the conditioning is
    new. Fitting a separate probe per bucket would confound "the state has
    forgotten" with "this bucket had fewer samples to fit on".
    """

    weights = ridge_fit(train.state, train.ground_truth["targets_pos"], penalty)
    prediction = ridge_apply(weights, test.state)
    target = test.ground_truth["targets_pos"].double()
    # (samples, targets, 2) -> squared error summed over the two coordinates.
    squared = ((prediction - target).reshape(len(test), -1, 2) ** 2).sum(dim=-1)
    return horizon_curve(squared, test.gaps)
