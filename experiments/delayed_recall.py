"""A delayed cued-recall gate for the Tensor Brain memory mechanisms.

The XOR diagnostic in :mod:`experiments.evolution_overfit` establishes that an
evolution operator can create a nonlinear boundary inside one concept window.
This diagnostic asks the separate question the PVSG streaming experiments
depend on: can a symbolic index survive a delay filled with competing input?

One trial presents a flagged cue item, then ``delay`` unflagged distractor items
drawn from the same bank, then a recall step with no input at all. That last
step is original Algorithm 1 with ``u = 0``, which QTB Equation (47) writes as
``q <- Wh + g(nu) + sum_k a_k`` with ``g(nu) = 0``. Every presented item is
decoded against the same index bank *before* its own feedback, so no readout
confirms itself, and the recall readout must name the cue rather than the item
seen most recently.

Three conditions answer the gate:

``feedback``
    the full mechanism, with index feedback at ``feedback_gate``;
``no-feedback``
    the identical schedule with the injection removed, so the cue survives only
    as the analog trace its input left in ``q``;
``gru``
    a capacity-comparable recurrent baseline reading the same inputs.

Distractors are drawn from the same bank as the cue, so maintenance has to
survive interference rather than merely decay. Recall accuracy as a function of
``delay`` is the measurement. The gate passes when ``feedback`` clears chance at
a delay where ``no-feedback`` does not; it fails, informatively, when the two
coincide at every delay.

This is a mechanism gate, not a benchmark. It says whether these operations can
hold an index across time at all, which is a precondition for reading anything
into a streaming PVSG result. Trials are resampled every training step, so a
reported accuracy is generalization to unseen trials rather than overfitting.

One caution governs every reading of it. A run that has not trained long enough
sits at exactly chance instead of degrading gracefully, so an insufficient step
budget is indistinguishable from a delay the mechanism cannot reach. Establish
that a delay is unreachable by raising ``training_steps`` until the solve rate
stops moving, never by observing chance once.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from tb import OriginalTBDynamicContext, QTBEvolution, ReLUEvolution, TensorBrain
from tb.model import ScoreMode

RecallCondition = Literal["feedback", "no-feedback", "gru"]
EvolutionVariant = Literal["original", "qtb-sigmoid", "qtb-relu"]

# The item bank and the evaluation trials are drawn from dedicated generators so
# that every condition and every delay sees the same sensory vocabulary and the
# same held-out trials. Only the model initialization follows the run seed.
ITEM_BANK_SEED = 20_260_807
EVALUATION_SEED = 20_260_808


@dataclass(frozen=True)
class RecallTrials:
    """A batch of trials: the presented item sequence and the cue to recall.

    ``presented[:, 0]`` is the cue and the remaining columns are distractors, so
    ``presented.shape[1] - 1`` is the delay. ``cue_flag`` marks the first column
    and is the only thing distinguishing the cue from a distractor at input
    time; without it the task would be unsolvable rather than merely hard.
    """

    presented: Int[Tensor, "batch steps"]
    cue_flag: Float[Tensor, "batch steps"]
    cue: Int[Tensor, " batch"]


@dataclass(frozen=True)
class RecallResult:
    condition: RecallCondition
    delay: int
    seed: int
    recall_accuracy: float
    identity_accuracy: float
    chance: float
    parameter_count: int
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class GateResult:
    """One condition at one delay, aggregated over seeds.

    Outcomes here are close to bimodal: a run either maintains the cue and
    approaches one, or collapses to exactly chance. A mean over seeds therefore
    reports how often the mechanism is reachable rather than how well it works,
    which is what a gate should measure, so ``solve_rate`` is the headline and
    ``mean_recall_accuracy`` is reported beside it rather than instead of it.
    """

    condition: RecallCondition
    delay: int
    solve_rate: float
    mean_recall_accuracy: float
    best_recall_accuracy: float
    mean_identity_accuracy: float
    chance: float
    parameter_count: int
    seeds: tuple[int, ...]


def make_item_bank(
    num_items: int, feature_dim: int, *, seed: int = ITEM_BANK_SEED
) -> Float[Tensor, "items feature"]:
    """Return fixed sensory features standing in for frozen DINO evidence.

    Each item is scaled to unit per-component RMS by the same convention the
    PVSG loader applies to real features, so the input reaching ``g`` has the
    pre-CBS scale the rest of the project assumes.
    """

    if num_items <= 1 or feature_dim <= 0:
        raise ValueError("num_items must exceed one and feature_dim must be positive")
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(num_items, feature_dim, generator=generator)
    return math.sqrt(feature_dim) * F.normalize(raw, p=2, dim=-1)


def sample_trials(
    num_items: int,
    delay: int,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> RecallTrials:
    """Draw trials whose distractors are never the cue itself.

    Adding an offset in ``1, ..., num_items - 1`` modulo the bank size draws
    uniformly from the non-cue items without rejection sampling. Distractors may
    repeat across the delay, which keeps the task defined when ``delay`` exceeds
    the number of remaining items.
    """

    if delay < 0 or batch_size <= 0:
        raise ValueError("delay must be non-negative and batch_size positive")
    cue = torch.randint(num_items, (batch_size,), generator=generator)
    offsets = torch.randint(1, num_items, (batch_size, delay), generator=generator)
    distractors = (cue.unsqueeze(-1) + offsets) % num_items
    presented = torch.cat([cue.unsqueeze(-1), distractors], dim=-1)
    cue_flag = torch.zeros_like(presented, dtype=torch.float32)
    cue_flag[:, 0] = 1.0
    return RecallTrials(presented=presented, cue_flag=cue_flag, cue=cue)


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
    evolution. The recall step then evolves once more with no input and reads the
    bank again, so the state that names the cue is one transition past the state
    that named the last distractor. ``feedback_gate`` of zero is the ablation:
    the code path is identical and only the injected embedding disappears.
    """

    def __init__(
        self,
        item_bank: Float[Tensor, "items feature"],
        *,
        state_dim: int,
        hidden_dim: int,
        evolution: EvolutionVariant,
        score_mode: ScoreMode,
        feedback_gate: float,
    ) -> None:
        super().__init__()
        num_items, feature_dim = item_bank.shape
        self.register_buffer("items", item_bank)
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

        features = self.items[trials.presented]
        drive = self.g(torch.cat([features, trials.cue_flag.unsqueeze(-1)], dim=-1))
        q = drive.new_zeros(drive.shape[0], self.brain.state_dim)
        context = None
        identity_logits = []
        for step in range(drive.shape[1]):
            q = self.brain.integrate_input(q, drive[:, step])
            identity_logits.append(self.brain.index_scores(q))
            q, _probabilities = self.brain.attend(q, feedback_gate=self.feedback_gate)
            q, context = self.brain.evolve(q, context)
        # Recall: no input integration, one transition, then read the same bank.
        q, context = self.brain.evolve(q, context)
        return self.brain.index_scores(q), torch.stack(identity_logits, dim=1)


class GRURecall(nn.Module):
    """A conventional recurrent baseline over the same inputs and supervision.

    It reads the identical ``g`` output, carries a distributed hidden state
    instead of a pre-CBS with index feedback, and decodes through an ordinary
    readout head rather than through the shared index bank. The final zero-input
    cell application mirrors the Tensor Brain's recall transition, so both models
    place exactly one transition between the last item and the recall readout.
    """

    def __init__(
        self,
        item_bank: Float[Tensor, "items feature"],
        *,
        state_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        num_items, feature_dim = item_bank.shape
        self.register_buffer("items", item_bank)
        self.g = nn.Linear(feature_dim + 1, state_dim)
        self.cell = nn.GRUCell(state_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, num_items)
        self.hidden_dim = hidden_dim

    def forward(
        self, trials: RecallTrials
    ) -> tuple[Float[Tensor, "batch items"], Float[Tensor, "batch steps items"]]:
        features = self.items[trials.presented]
        drive = self.g(torch.cat([features, trials.cue_flag.unsqueeze(-1)], dim=-1))
        hidden = drive.new_zeros(drive.shape[0], self.hidden_dim)
        identity_logits = []
        for step in range(drive.shape[1]):
            hidden = self.cell(drive[:, step], hidden)
            identity_logits.append(self.readout(hidden))
        hidden = self.cell(torch.zeros_like(drive[:, 0]), hidden)
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
    item_bank: Float[Tensor, "items feature"],
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
            item_bank,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            evolution=evolution,
            score_mode=score_mode,
            feedback_gate=feedback_gate if condition == "feedback" else 0.0,
        )
    if condition == "gru":
        return GRURecall(item_bank, state_dim=state_dim, hidden_dim=gru_hidden_dim)
    raise ValueError(f"unknown recall condition: {condition}")


def train_condition(
    condition: RecallCondition,
    *,
    delay: int,
    num_items: int = 16,
    feature_dim: int = 32,
    state_dim: int = 64,
    hidden_dim: int = 64,
    # Chosen so the baseline's parameter count lands within a few percent of the
    # Tensor Brain's at the default dimensions; both counts are reported so the
    # comparability is checkable rather than asserted.
    gru_hidden_dim: int = 32,
    evolution: EvolutionVariant = "qtb-sigmoid",
    score_mode: ScoreMode = "direct",
    feedback_gate: float = 1.0,
    identity_loss_weight: float = 1.0,
    batch_size: int = 128,
    # An under-trained run fails at chance rather than degrading, so too small a
    # budget manufactures a delay wall that looks like a capacity limit. At
    # delay four this configuration solves 3/3 seeds at 12,000 steps and 0/5 at
    # 3,000. Raise the budget, not the learning rate, before reading a longer
    # delay as unreachable.
    training_steps: int = 12_000,
    learning_rate: float = 1e-3,
    gradient_clip: float = 1.0,
    evaluation_trials: int = 1_024,
    seed: int = 0,
) -> RecallResult:
    """Train one condition at one delay and evaluate it on held-out trials.

    Every condition receives the same optimizer, learning rate, gradient clip,
    and step budget. Gradient clipping is ordinary practice for training through
    a recurrence and is applied identically everywhere; without it the runs are
    sharply bimodal and a single seed reports initialization rather than
    mechanism.
    """

    torch.manual_seed(seed)
    item_bank = make_item_bank(num_items, feature_dim)
    model = build_model(
        condition,
        item_bank,
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
        trials = sample_trials(num_items, delay, batch_size)
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
        num_items,
        delay,
        evaluation_trials,
        generator=torch.Generator().manual_seed(EVALUATION_SEED),
    )
    model.eval()
    with torch.no_grad():
        recall_logits, identity_logits = model(evaluation)
        recall_accuracy = float(
            (recall_logits.argmax(dim=-1) == evaluation.cue).float().mean()
        )
        identity_accuracy = float(
            (identity_logits.argmax(dim=-1) == evaluation.presented).float().mean()
        )

    return RecallResult(
        condition=condition,
        delay=delay,
        seed=seed,
        recall_accuracy=recall_accuracy,
        identity_accuracy=identity_accuracy,
        chance=1.0 / num_items,
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
    solved_threshold: float = 0.5,
    **options: Any,
) -> GateResult:
    """Aggregate one condition at one delay over seeds.

    ``solved_threshold`` separates the two modes rather than setting a quality
    bar. At sixteen items, chance is 0.0625 and a maintaining run lands near one,
    so any cut well inside that gap gives the same solve rate.
    """

    if not seeds:
        raise ValueError("at least one seed is required")
    results = [train_condition(condition, delay=delay, seed=seed, **options) for seed in seeds]
    accuracies = [result.recall_accuracy for result in results]
    return GateResult(
        condition=condition,
        delay=delay,
        solve_rate=sum(value >= solved_threshold for value in accuracies) / len(accuracies),
        mean_recall_accuracy=sum(accuracies) / len(accuracies),
        best_recall_accuracy=max(accuracies),
        mean_identity_accuracy=sum(result.identity_accuracy for result in results)
        / len(results),
        chance=results[0].chance,
        parameter_count=results[0].parameter_count,
        seeds=tuple(seeds),
    )


if __name__ == "__main__":
    _conditions: tuple[RecallCondition, ...] = ("feedback", "no-feedback", "gru")
    for _evolution in ("qtb-sigmoid", "qtb-relu", "original"):
        print(f"\nevolution={_evolution}")
        print(
            f"{'delay':>6} {'condition':>12} {'solved':>7} "
            f"{'mean':>7} {'best':>7} {'identity':>9} {'params':>8}"
        )
        for _delay in (1, 2, 4, 8, 16):
            for _condition in _conditions:
                _gate = run_gate(_condition, delay=_delay, evolution=_evolution)
                print(
                    f"{_gate.delay:>6} {_gate.condition:>12} "
                    f"{_gate.solve_rate:>7.2f} "
                    f"{_gate.mean_recall_accuracy:>7.3f} "
                    f"{_gate.best_recall_accuracy:>7.3f} "
                    f"{_gate.mean_identity_accuracy:>9.3f} "
                    f"{_gate.parameter_count:>8}"
                )
