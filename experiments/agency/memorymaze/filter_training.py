"""Phase 1: fit the filter. Pixels and actions only.

The objective is to reconstruct the frame the trajectory is currently at, from
the filter's own state, with observations withheld at probability ``rho``. On an
observed step that is an autoencoder; on a masked step there is no input to copy
from, so the only route to a reconstruction is the belief -- which is where the
retention pressure comes from.

What the loss must *not* contain is any ground truth. Agent positions, target
positions and the nearest-target colour are all available in the corpus, and
putting any of them in the objective would mean Phase 2 probes recover something
the loss put there. The separation is the study's main methodological guarantee,
so the objective is restricted to pixels and the recorded action.

Two details that would quietly invalidate the numbers if left out:

**Burn-in.** A window sampled from the middle of an episode is entered with a
zeroed state, which is simply the wrong belief. The first ``burn_in`` steps
therefore run without contributing to the loss, so the model is scored on steps
whose state was reached the way it would be at run time.

**A separate mask stream.** The masking generator is seeded independently of the
batch sampler, so two conditions trained on the same batches also see the same
mask pattern. Otherwise a difference between variants could be a difference in
which steps happened to be hidden.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn

from experiments.agency.memorymaze.corpus import OfflineCorpus
from experiments.agency.memorymaze.filter import (
    FilterConfig,
    FrameDecoder,
    TensorBrainFilter,
    build_filter,
    observation_mask,
)
from experiments.agency.memorymaze.filter_diagnostics import (
    DiagnosticAccumulator,
    step_diagnostics,
)


@dataclass(frozen=True)
class TrainConfig:
    """Every controlled variable of Phase 1."""

    steps: int = 6000
    batch_size: int = 24
    segment_steps: int = 64
    # Of the 64 steps in a window, the first 16 build a state and are not scored.
    burn_in: int = 16
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    log_every: int = 250
    seed: int = 0


@dataclass
class TrainingLog:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    diagnostics: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FilterModel(nn.Module):
    """A filter and its decoder, trained together as one module."""

    def __init__(self, config: FilterConfig) -> None:
        super().__init__()
        self.config = config
        self.filter = build_filter(config)
        self.decoder = FrameDecoder(config.state_dim, channels=config.channels)

    @property
    def encoder(self) -> nn.Module:
        return self.filter.encoder

    @property
    def brain(self) -> TensorBrainFilter | None:
        """The Tensor Brain filter, or ``None`` for a recurrent control."""

        return self.filter if isinstance(self.filter, TensorBrainFilter) else None

    def rollout(
        self,
        batch: dict[str, Tensor],
        mask_generator: torch.Generator,
        *,
        burn_in: int,
        diagnostics: DiagnosticAccumulator | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Run the filter over a time-major batch and return the loss and states."""

        images, actions = batch["image"], batch["action"]
        steps, size = actions.shape
        device = images.device
        state, context = self.filter.initial_state(size, device)
        losses: list[Tensor] = []
        readouts: list[Tensor] = []

        for step in range(steps):
            observed = observation_mask(
                size, self.config.mask_probability, mask_generator, device
            )
            trace = self.filter.step(
                state, context, images[step], actions[step], observed
            )
            state, context = trace.q, trace.context
            if step >= burn_in:
                reconstruction = self.decoder(trace.readout)
                losses.append(((reconstruction - images[step]) ** 2).mean())
                readouts.append(trace.readout)
                if diagnostics is not None:
                    diagnostics.update(step_diagnostics(trace, self.brain))
        return torch.stack(losses).mean(), readouts


def train_filter(
    model: FilterModel,
    corpus: OfflineCorpus,
    config: TrainConfig,
    *,
    device: torch.device | None = None,
    progress: bool = True,
) -> TrainingLog:
    """Fit one filter on the offline corpus."""

    device = device or torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Three independent streams: batch selection, masking, and model init. Two
    # conditions at the same seed then see identical batches and identical
    # masks, so their difference is architectural.
    batch_generator = torch.Generator().manual_seed(config.seed)
    mask_generator = torch.Generator().manual_seed(config.seed + 90_000)
    log = TrainingLog()

    for step in range(config.steps):
        batch = corpus.sample(config.batch_size, config.segment_steps, batch_generator)
        batch = {name: value.to(device) for name, value in batch.items()}
        diagnostics = DiagnosticAccumulator()
        loss, _ = model.rollout(
            batch, mask_generator, burn_in=config.burn_in, diagnostics=diagnostics
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % config.log_every == 0 or step == config.steps - 1:
            log.step.append(step)
            log.loss.append(float(loss.detach()))
            log.diagnostics.append(diagnostics.means())
            if progress:
                summary = " ".join(
                    f"{name}={value:.3f}" for name, value in diagnostics.means().items()
                )
                print(f"step {step}/{config.steps} loss {float(loss):.5f} {summary}", flush=True)
    return log
