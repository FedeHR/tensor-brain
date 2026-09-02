"""Training and evaluation for one measurement schedule.

Every condition is trained identically apart from its collapse schedule. Two
choices follow the chain-of-thought literature rather than convenience.

Curriculum
    Stage ``k`` runs the condition's own collapse at the first ``k`` intermediate
    hops and teacher-forces the remainder onto the gold path, with the step loss
    applied only at the teacher-forced hops. This is COCONUT's staged curriculum
    transcribed to the Tensor Brain; the paper reports that latent thoughts
    trained without it do no better than no chain at all. The last stage is the
    condition itself, and evaluation only ever happens there: teacher forcing
    writes a gold node into the workspace, whose child *is* the correct terminal,
    so a teacher-forced chain is an oracle and never an eval condition.

Step supervision target
    Two choices are available and they are not equivalent. ``path`` puts all mass
    on one gold node, which is what a written-out chain-of-thought trace supplies.
    ``frontier`` spreads mass over every node the search could legitimately be at,
    which is the well-posed target when several children are equally valid. On
    this task ``path`` is misspecified by construction: the gold path tie-breaks
    arbitrarily among ``b`` equally valid children.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.measurement_cot.collapse import CollapseSpec
from experiments.measurement_cot.data import QuerySet
from experiments.measurement_cot.graph import LayeredDAG
from experiments.measurement_cot.model import MeasurementChain
from tb.vocabulary import get_candidate_positions

SupervisionTarget = Literal["none", "path", "frontier"]


@dataclass
class TrainConfig:
    """Optimisation settings shared by every condition."""

    steps: int = 4000
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    supervision: SupervisionTarget = "frontier"
    supervision_weight: float = 1.0
    curriculum_stages: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    eval_every: int = 500
    device: str = "cpu"


@dataclass
class TrainResult:
    """Outcome of one training run."""

    train_accuracy: float
    test_accuracy: float
    test_accuracy_std: float
    history: list[dict[str, float]] = field(default_factory=list)


def frontier_targets(
    graph: LayeredDAG, queries: QuerySet, hop: int, device: torch.device
) -> Tensor:
    """Uniform distribution over the layer-``hop`` nodes reachable from each start."""

    mask = graph.frontier_masks[hop].to(device)[queries.start_position]
    counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    return mask.float() / counts


def step_supervision_loss(
    model: MeasurementChain,
    queries: QuerySet,
    q_states: dict[int, Tensor],
    target: SupervisionTarget,
) -> Tensor:
    """Cross-entropy from the candidate scores towards the chosen step target."""

    if target == "none" or not q_states:
        return next(iter(q_states.values())).new_zeros(()) if q_states else torch.zeros(())
    total = None
    for hop, q in q_states.items():
        candidates = model.candidates_at(hop)
        scores = model.tb.index_scores(q, candidates)
        if target == "path":
            positions = get_candidate_positions(
                candidates, model.graph.to_global(hop, queries.gold_path[:, hop])
            )
            loss = F.cross_entropy(scores, positions)
        else:
            distribution = frontier_targets(model.graph, queries, hop, scores.device)
            loss = -(distribution * F.log_softmax(scores, dim=-1)).sum(-1).mean()
        total = loss if total is None else total + loss
    return total / len(q_states)


def stage_schedule(
    schedule: list[CollapseSpec], stage: int, total_stages: int
) -> tuple[list[CollapseSpec], list[int]]:
    """Return the stage's schedule and which hops keep step supervision."""

    if total_stages <= 0:
        return list(schedule), []
    replaced = min(stage, len(schedule))
    teacher = CollapseSpec(mode="teacher")
    staged = list(schedule[:replaced]) + [teacher] * (len(schedule) - replaced)
    supervised_hops = list(range(replaced + 1, len(schedule) + 1))
    return staged, supervised_hops


def evaluate(
    model: MeasurementChain,
    queries: QuerySet,
    schedule: list[CollapseSpec],
    *,
    repeats: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """Mean and standard deviation of accuracy over stochastic repeats."""

    was_training = model.training
    model.eval()
    accuracies = []
    with torch.no_grad():
        for _ in range(repeats):
            trace = model(queries, schedule, generator=generator)
            accuracies.append(float((trace.logits.argmax(-1) == queries.answer).float().mean()))
    model.train(was_training)
    values = torch.tensor(accuracies)
    return float(values.mean()), float(values.std(unbiased=False))


def train_chain(
    graph: LayeredDAG,
    train_queries: QuerySet,
    test_queries: QuerySet,
    schedule: list[CollapseSpec],
    config: TrainConfig,
    *,
    model_kwargs: dict | None = None,
    eval_repeats: int = 5,
) -> tuple[MeasurementChain, TrainResult]:
    """Train one chain under a fixed measurement schedule."""

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = MeasurementChain(graph, **(model_kwargs or {})).to(device)
    train_queries = train_queries.to(device)
    test_queries = test_queries.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    history: list[dict[str, float]] = []

    num_stages = config.curriculum_stages
    stage_length = max(1, config.steps // (num_stages + 1)) if num_stages else config.steps

    for step in range(config.steps):
        stage = min(step // stage_length, num_stages) if num_stages else 0
        active_schedule, supervised_hops = stage_schedule(schedule, stage, num_stages)

        batch_index = torch.randint(
            len(train_queries),
            (min(config.batch_size, len(train_queries)),),
            generator=generator,
            device=device,
        )
        batch = train_queries.index(batch_index)

        needs_states = config.supervision != "none" and bool(supervised_hops)
        trace = model(batch, active_schedule, record=needs_states, generator=generator)
        loss = F.cross_entropy(trace.logits, batch.answer)
        if needs_states:
            states = {
                hop: trace.pre_feedback_state[hop - 1]
                for hop in supervised_hops
                if hop - 1 < len(trace.pre_feedback_state)
            }
            if states:
                loss = loss + config.supervision_weight * step_supervision_loss(
                    model, batch, states, config.supervision
                )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if (step + 1) % config.eval_every == 0 or step == config.steps - 1:
            test_mean, _ = evaluate(model, test_queries, schedule, repeats=1, generator=generator)
            history.append(
                {
                    "step": float(step + 1),
                    "stage": float(stage),
                    "loss": float(loss),
                    "test_accuracy": test_mean,
                }
            )

    train_accuracy, _ = evaluate(model, train_queries, schedule, repeats=1, generator=generator)
    test_accuracy, test_std = evaluate(
        model, test_queries, schedule, repeats=eval_repeats, generator=generator
    )
    return model, TrainResult(
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        test_accuracy_std=test_std,
        history=history,
    )
