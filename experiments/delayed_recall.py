"""A delayed cued-recall gate for the Tensor Brain memory mechanisms.

The XOR diagnostic in :mod:`experiments.evolution_overfit` establishes that an
evolution operator can create a nonlinear boundary inside one concept window.
This diagnostic asks the separate question the PVSG streaming experiments
depend on: is there any regime in which index feedback is worth more than the
analog trace the same observation already left in ``q``?

One trial presents a flagged cue, then ``delay`` distractor items drawn from the
same bank, then a recall step. Every presented item is decoded against the
shared bank *before* its own feedback, so no readout confirms itself, and the
recall readout must name the cue rather than the item seen most recently.
Distractors come from the cue's own bank, so maintenance survives interference
rather than only decay.

Two design choices decide whether the comparison can say anything at all.

**Views, not fixed features.** Each item owns a prototype, and every
presentation is a noisy view of it. With one fixed feature per item the index
embedding ``a_k`` would be a deterministic re-encoding of the input that a
learned ``g`` can reproduce, so feedback could not carry information and a null
would be a property of the task rather than of the mechanism. Under noisy views
``a_k`` is instead a prototype accumulated across occurrences, which is the
claim original §6 makes for entity indices on VRD-EX.

**The recall step.** ``silent`` presents nothing -- original Algorithm 1 with
``u = 0``, which QTB Equation (47) writes as ``q <- Wh + g(nu) + sum_k a_k``
with ``g(nu) = 0``. ``probe`` instead presents a view that is deliberately
ambiguous between the cue and a lure item, so bottom-up evidence narrows the
answer to two candidates and cannot choose between them. Only what was retained
from the cue's earlier presentation separates them, which is the toy form of
re-identifying an entity after occlusion against a visually similar distractor.

Three conditions share the optimizer, learning rate, gradient clip and step
budget: ``feedback`` is the full mechanism, ``no-feedback`` is the identical
schedule with the injection removed, and ``gru`` is a capacity-comparable
recurrent baseline reading the same inputs under the same supervision.

This is a mechanism gate, not a benchmark, and it is not evidence about PVSG.
It says whether these operations pay off anywhere; whether PVSG contains the
regime in which they pay is a separate question about the dataset.

One caution governs every reading of it. A run that has not trained long enough
sits at exactly chance instead of degrading gracefully, so an insufficient step
budget is indistinguishable from a delay the mechanism cannot reach. Establish
that a delay is unreachable by raising ``training_steps`` until the solve rate
stops moving, never by observing chance once.
"""

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from tb import OriginalTBDynamicContext, QTBEvolution, ReLUEvolution, TensorBrain
from tb.model import ScoreMode

RecallCondition = Literal["feedback", "no-feedback", "gru"]
EvolutionVariant = Literal["original", "qtb-sigmoid", "qtb-relu"]
RecallMode = Literal["silent", "probe"]

# The item bank and the evaluation trials are drawn from dedicated generators so
# that every condition and every delay sees the same sensory vocabulary and the
# same held-out trials. Only the model initialization follows the run seed.
ITEM_BANK_SEED = 20_260_807
EVALUATION_SEED = 20_260_808


@dataclass(frozen=True)
class RecallTrials:
    """A batch of trials: what perception receives, and what recall must name.

    ``views`` is what the model sees; ``presented`` is the identity supervision
    for those same steps. ``presented[:, 0]`` is the cue and the remaining
    columns are distractors, so ``presented.shape[1] - 1`` is the delay.
    ``cue_flag`` marks the first column and is the only thing distinguishing the
    cue from a distractor at input time.

    ``probe_view`` and ``lure`` are present only in ``probe`` recall mode. The
    probe is ambiguous between the cue and the lure by construction, so a model
    reading it alone cannot exceed one half.
    """

    views: Float[Tensor, "batch steps feature"]
    presented: Int[Tensor, "batch steps"]
    cue_flag: Float[Tensor, "batch steps"]
    cue: Int[Tensor, " batch"]
    lure: Int[Tensor, " batch"] | None = None
    probe_view: Float[Tensor, "batch feature"] | None = None


@dataclass(frozen=True)
class RecallResult:
    condition: RecallCondition
    delay: int
    seed: int
    recall_accuracy: float
    identity_accuracy: float
    lure_rate: float
    chance: float
    parameter_count: int
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class GateResult:
    """One condition at one delay, aggregated over seeds.

    Outcomes are close to bimodal: a run either maintains the cue and approaches
    one, or collapses to chance. A mean over seeds therefore reports how often
    the mechanism is reachable rather than how well it works, so ``solve_rate``
    is the headline and ``mean_recall_accuracy`` is reported beside it.
    """

    condition: RecallCondition
    delay: int
    solve_rate: float
    mean_recall_accuracy: float
    best_recall_accuracy: float
    mean_identity_accuracy: float
    mean_lure_rate: float
    chance: float
    parameter_count: int
    seeds: tuple[int, ...]


def _rms_normalize(features: Float[Tensor, "*rows feature"]) -> Float[Tensor, "*rows feature"]:
    """Put features at unit per-component RMS, as the PVSG loader does."""

    return math.sqrt(features.shape[-1]) * F.normalize(features, p=2, dim=-1)


def make_item_bank(
    num_items: int, feature_dim: int, *, seed: int = ITEM_BANK_SEED
) -> Float[Tensor, "items feature"]:
    """Return fixed item prototypes standing in for a frozen visual backbone."""

    if num_items <= 1 or feature_dim <= 0:
        raise ValueError("num_items must exceed one and feature_dim must be positive")
    generator = torch.Generator().manual_seed(seed)
    return _rms_normalize(torch.randn(num_items, feature_dim, generator=generator))


def sample_trials(
    item_bank: Float[Tensor, "items feature"],
    delay: int,
    batch_size: int,
    *,
    view_noise: float = 0.0,
    recall_mode: RecallMode = "silent",
    probe_mix: float = 0.5,
    generator: torch.Generator | None = None,
) -> RecallTrials:
    """Draw trials whose distractors are never the cue itself.

    Adding an offset in ``1, ..., num_items - 1`` modulo the bank size draws
    uniformly from the non-cue items without rejection sampling. Distractors may
    repeat across the delay, which keeps the task defined when ``delay`` exceeds
    the number of remaining items. The lure is drawn the same way, so it is
    never the cue but may coincide with a distractor.
    """

    if delay < 0 or batch_size <= 0:
        raise ValueError("delay must be non-negative and batch_size positive")
    if view_noise < 0:
        raise ValueError("view_noise must be non-negative")
    if not 0.0 <= probe_mix <= 1.0:
        raise ValueError("probe_mix must lie between zero and one")
    num_items = item_bank.shape[0]

    cue = torch.randint(num_items, (batch_size,), generator=generator)
    offsets = torch.randint(1, num_items, (batch_size, delay), generator=generator)
    presented = torch.cat([cue.unsqueeze(-1), (cue.unsqueeze(-1) + offsets) % num_items], dim=-1)
    cue_flag = torch.zeros_like(presented, dtype=torch.float32)
    cue_flag[:, 0] = 1.0

    views = item_bank[presented]
    if view_noise > 0:
        views = _rms_normalize(
            views + view_noise * torch.randn(views.shape, generator=generator)
        )

    if recall_mode == "silent":
        return RecallTrials(views=views, presented=presented, cue_flag=cue_flag, cue=cue)
    if recall_mode != "probe":
        raise ValueError(f"unknown recall mode: {recall_mode}")

    lure_offset = torch.randint(1, num_items, (batch_size,), generator=generator)
    lure = (cue + lure_offset) % num_items
    # An equal mixture of the two prototypes is equidistant from both, so the
    # probe alone identifies the pair but never which member of it.
    blended = probe_mix * item_bank[cue] + (1.0 - probe_mix) * item_bank[lure]
    if view_noise > 0:
        blended = blended + view_noise * torch.randn(blended.shape, generator=generator)
    return RecallTrials(
        views=views,
        presented=presented,
        cue_flag=cue_flag,
        cue=cue,
        lure=lure,
        probe_view=_rms_normalize(blended),
    )


def make_evolution(
    variant: EvolutionVariant, state_dim: int, hidden_dim: int
) -> OriginalTBDynamicContext | QTBEvolution | ReLUEvolution:
    """Construct one named evolution backend, as the XOR diagnostic does."""

    if variant == "original":
        return OriginalTBDynamicContext(state_dim, hidden_dim)
    if variant == "qtb-sigmoid":
        return QTBEvolution(state_dim, hidden_dim)
    if variant == "qtb-relu":
        return ReLUEvolution(state_dim, hidden_dim)
    raise ValueError(f"unknown evolution variant: {variant}")


class TensorBrainRecall(nn.Module):
    """The Tensor Brain schedule for one delayed-recall trial.

    Per presented item the schedule is input, identity readout, index feedback,
    evolution. The recall step then evolves once more -- integrating the probe
    first when there is one -- and reads the bank again, so the state that names
    the cue is one transition past the state that named the last distractor.
    ``feedback_gate`` of zero is the ablation: the code path is identical and
    only the injected embedding disappears.
    """

    def __init__(
        self,
        num_items: int,
        feature_dim: int,
        *,
        state_dim: int,
        hidden_dim: int,
        evolution: EvolutionVariant,
        score_mode: ScoreMode,
        feedback_gate: float,
    ) -> None:
        super().__init__()
        # The cue flag is one extra input component, so g maps feature_dim + 1
        # into pre-CBS coordinates. Everything before the Tensor Brain core is
        # the experiment's own perception, exactly as on PVSG.
        self.g = nn.Linear(feature_dim + 1, state_dim)
        self.brain = TensorBrain(
            state_dim,
            num_items,
            make_evolution(evolution, state_dim, hidden_dim),
            score_mode=score_mode,
        )
        self.feedback_gate = feedback_gate

    def forward(
        self, trials: RecallTrials
    ) -> tuple[Float[Tensor, "batch items"], Float[Tensor, "batch steps items"]]:
        """Return the recall logits and the per-step identity logits."""

        drive = self.g(torch.cat([trials.views, trials.cue_flag.unsqueeze(-1)], dim=-1))
        q = drive.new_zeros(drive.shape[0], self.brain.state_dim)
        context = None
        identity_logits = []
        for step in range(drive.shape[1]):
            q = self.brain.integrate_input(q, drive[:, step])
            identity_logits.append(self.brain.index_scores(q))
            q, _probabilities = self.brain.attend(q, feedback_gate=self.feedback_gate)
            q, context = self.brain.evolve(q, context)
        if trials.probe_view is not None:
            probe_flag = trials.probe_view.new_zeros(trials.probe_view.shape[0], 1)
            q = self.brain.integrate_input(
                q, self.g(torch.cat([trials.probe_view, probe_flag], dim=-1))
            )
        q, context = self.brain.evolve(q, context)
        return self.brain.index_scores(q), torch.stack(identity_logits, dim=1)


class GRURecall(nn.Module):
    """A conventional recurrent baseline over the same inputs and supervision.

    It reads the identical ``g`` output, carries a distributed hidden state
    instead of a pre-CBS with index feedback, and decodes through an ordinary
    readout head rather than through the shared index bank. The final cell
    application mirrors the Tensor Brain's recall transition, so both models
    place exactly one transition between the last item and the recall readout,
    and both see the probe when there is one.
    """

    def __init__(
        self,
        num_items: int,
        feature_dim: int,
        *,
        state_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.g = nn.Linear(feature_dim + 1, state_dim)
        self.cell = nn.GRUCell(state_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, num_items)
        self.hidden_dim = hidden_dim

    def forward(
        self, trials: RecallTrials
    ) -> tuple[Float[Tensor, "batch items"], Float[Tensor, "batch steps items"]]:
        drive = self.g(torch.cat([trials.views, trials.cue_flag.unsqueeze(-1)], dim=-1))
        hidden = drive.new_zeros(drive.shape[0], self.hidden_dim)
        identity_logits = []
        for step in range(drive.shape[1]):
            hidden = self.cell(drive[:, step], hidden)
            identity_logits.append(self.readout(hidden))
        if trials.probe_view is None:
            recall_drive = torch.zeros_like(drive[:, 0])
        else:
            probe_flag = trials.probe_view.new_zeros(trials.probe_view.shape[0], 1)
            recall_drive = self.g(torch.cat([trials.probe_view, probe_flag], dim=-1))
        hidden = self.cell(recall_drive, hidden)
        return self.readout(hidden), torch.stack(identity_logits, dim=1)


def _recall_losses(
    recall_logits: Float[Tensor, "batch items"],
    identity_logits: Float[Tensor, "batch steps items"],
    trials: RecallTrials,
    *,
    identity_loss_weight: float,
) -> Tensor:
    """Combine the recall objective with the per-step identity supervision.

    The identity term is what grounds the index bank. Without it the columns of
    ``A`` are unconstrained, feedback injects an arbitrary vector, and a null
    would say nothing about the mechanism. Every presented item is supervised,
    which is also what the PVSG object and pair schedules do.
    """

    recall_loss = F.cross_entropy(recall_logits, trials.cue)
    identity_loss = F.cross_entropy(
        identity_logits.flatten(0, 1), trials.presented.flatten(0, 1)
    )
    return recall_loss + identity_loss_weight * identity_loss


def build_model(
    condition: RecallCondition,
    num_items: int,
    feature_dim: int,
    *,
    state_dim: int,
    hidden_dim: int,
    gru_hidden_dim: int,
    evolution: EvolutionVariant,
    score_mode: ScoreMode,
    feedback_gate: float,
) -> TensorBrainRecall | GRURecall:
    """Construct one condition of the controlled comparison."""

    if condition in ("feedback", "no-feedback"):
        return TensorBrainRecall(
            num_items,
            feature_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            evolution=evolution,
            score_mode=score_mode,
            feedback_gate=feedback_gate if condition == "feedback" else 0.0,
        )
    if condition == "gru":
        return GRURecall(
            num_items, feature_dim, state_dim=state_dim, hidden_dim=gru_hidden_dim
        )
    raise ValueError(f"unknown recall condition: {condition}")


def train_condition(
    condition: RecallCondition,
    *,
    delay: int,
    recall_mode: RecallMode = "silent",
    view_noise: float = 0.0,
    probe_mix: float = 0.5,
    num_items: int = 16,
    feature_dim: int = 32,
    state_dim: int = 64,
    hidden_dim: int = 64,
    # Chosen so the baseline's parameter count lands within a few percent of the
    # Tensor Brain's at the default dimensions; both counts are reported so the
    # comparability is checkable rather than asserted.
    gru_hidden_dim: int = 32,
    evolution: EvolutionVariant = "qtb-relu",
    score_mode: ScoreMode = "direct",
    feedback_gate: float = 1.0,
    identity_loss_weight: float = 1.0,
    batch_size: int = 128,
    # An under-trained run fails at chance rather than degrading, so too small a
    # budget manufactures a delay wall that looks like a capacity limit. At
    # delay four the silent task solves 3/3 seeds at 12,000 steps and 0/5 at
    # 3,000. Raise the budget, not the learning rate, before reading a longer
    # delay as unreachable.
    training_steps: int = 12_000,
    learning_rate: float = 1e-3,
    gradient_clip: float = 1.0,
    evaluation_trials: int = 1_024,
    seed: int = 0,
) -> RecallResult:
    """Train one condition at one delay and evaluate it on held-out trials.

    Every condition receives the same optimizer, learning rate, gradient clip
    and step budget. Gradient clipping is ordinary practice for training through
    a recurrence and is applied identically everywhere; without it the runs are
    sharply bimodal and a single seed reports initialization rather than
    mechanism.
    """

    torch.manual_seed(seed)
    item_bank = make_item_bank(num_items, feature_dim)
    trial_options = {
        "view_noise": view_noise,
        "recall_mode": recall_mode,
        "probe_mix": probe_mix,
    }
    model = build_model(
        condition,
        num_items,
        feature_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        gru_hidden_dim=gru_hidden_dim,
        evolution=evolution,
        score_mode=score_mode,
        feedback_gate=feedback_gate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    model.train()
    for _step in range(training_steps):
        trials = sample_trials(item_bank, delay, batch_size, **trial_options)
        optimizer.zero_grad()
        recall_logits, identity_logits = model(trials)
        loss = _recall_losses(
            recall_logits,
            identity_logits,
            trials,
            identity_loss_weight=identity_loss_weight,
        )
        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach()))

    evaluation = sample_trials(
        item_bank,
        delay,
        evaluation_trials,
        generator=torch.Generator().manual_seed(EVALUATION_SEED),
        **trial_options,
    )
    model.eval()
    with torch.no_grad():
        recall_logits, identity_logits = model(evaluation)
        decoded = recall_logits.argmax(dim=-1)
        recall_accuracy = float((decoded == evaluation.cue).float().mean())
        identity_accuracy = float(
            (identity_logits.argmax(dim=-1) == evaluation.presented).float().mean()
        )
        lure_rate = (
            float((decoded == evaluation.lure).float().mean())
            if evaluation.lure is not None
            else 0.0
        )

    return RecallResult(
        condition=condition,
        delay=delay,
        seed=seed,
        recall_accuracy=recall_accuracy,
        identity_accuracy=identity_accuracy,
        lure_rate=lure_rate,
        # A probe narrows the answer to the cue and its lure, so a model reading
        # only the probe scores one half. That, not 1/num_items, is the floor a
        # probe-mode result has to clear.
        chance=0.5 if recall_mode == "probe" else 1.0 / num_items,
        parameter_count=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        loss_history=tuple(losses),
    )


def run_gate(
    condition: RecallCondition,
    *,
    delay: int,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    solved_margin: float = 0.25,
    **options: Any,
) -> GateResult:
    """Aggregate one condition at one delay over seeds.

    ``solved_margin`` separates the two modes rather than setting a quality bar:
    a run counts as solved when it clears chance by that margin. Silent recall
    lands near one against a chance of 1/16, and probe recall has to clear one
    half, so the same margin serves both without a mode-specific threshold.
    """

    if not seeds:
        raise ValueError("at least one seed is required")
    results = [train_condition(condition, delay=delay, seed=seed, **options) for seed in seeds]
    accuracies = [result.recall_accuracy for result in results]
    chance = results[0].chance
    return GateResult(
        condition=condition,
        delay=delay,
        solve_rate=sum(value >= chance + solved_margin for value in accuracies) / len(accuracies),
        mean_recall_accuracy=sum(accuracies) / len(accuracies),
        best_recall_accuracy=max(accuracies),
        mean_identity_accuracy=sum(result.identity_accuracy for result in results)
        / len(results),
        mean_lure_rate=sum(result.lure_rate for result in results) / len(results),
        chance=chance,
        parameter_count=results[0].parameter_count,
        seeds=tuple(seeds),
    )


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=("feedback", "no-feedback", "gru"))
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--recall-mode", default="silent", choices=("silent", "probe"))
    parser.add_argument("--view-noise", type=float, default=0.0)
    parser.add_argument("--probe-mix", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--evolution", default="qtb-relu", choices=("original", "qtb-sigmoid", "qtb-relu")
    )
    parser.add_argument("--num-items", type=int, default=16)
    parser.add_argument("--training-steps", type=int, default=12_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--feedback-gate", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="append the gate result as one JSON line")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_arguments(argv)
    gate = run_gate(
        arguments.condition,
        delay=arguments.delay,
        seeds=tuple(arguments.seeds),
        recall_mode=arguments.recall_mode,
        view_noise=arguments.view_noise,
        probe_mix=arguments.probe_mix,
        evolution=arguments.evolution,
        num_items=arguments.num_items,
        training_steps=arguments.training_steps,
        learning_rate=arguments.learning_rate,
        feedback_gate=arguments.feedback_gate,
    )
    record = {
        **vars(gate),
        "recall_mode": arguments.recall_mode,
        "view_noise": arguments.view_noise,
        "evolution": arguments.evolution,
        "training_steps": arguments.training_steps,
        "learning_rate": arguments.learning_rate,
    }
    record["seeds"] = list(gate.seeds)
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


if __name__ == "__main__":
    main()
