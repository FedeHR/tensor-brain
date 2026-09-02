r"""Measurements taken on a trained chain, not further training.

Three quantities carry the argument.

Frontier mass
    How much of the candidate distribution sits on nodes the search could
    legitimately occupy at that hop. The Tensor Brain index layer exists at every
    step by construction, so the superposition that latent reasoning is supposed
    to maintain can simply be read off rather than probed for.

Jensen gap
    :math:`\lVert \Phi(\bar q) - \mathbb{E}_p[\Phi(\alpha q + \beta a)]\rVert`, where
    :math:`\bar q = \alpha q + \beta\,\mathbb{E}_p[a]`;
    the difference between evolving the averaged state and averaging the evolved
    states. In unitary quantum computation measurements may always be deferred to
    the end without changing the result; here the evolution operator is nonlinear,
    so measurement and evolution do not commute and this gap is exactly the
    failure of that deferred-measurement principle. Candidate sets are small
    enough that the expectation is computed exactly, with no sampling.

Zeno trajectory
    What repeated measurement of the *same* concept window does, with no evolution
    in between. This is the regime the QTB draft points at when it describes
    repeated application of an operator without fresh perceptual input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from experiments.measurement_cot.collapse import CollapseSpec, collapse_weights, index_entropy
from experiments.measurement_cot.data import QuerySet
from experiments.measurement_cot.model import MeasurementChain


@dataclass
class StepReport:
    """Per-hop summary of what the chain believed and how much it hedged."""

    hop: int
    frontier_mass: float
    frontier_chance: float
    entropy: float
    max_entropy: float
    effective_alternatives: float
    frontier_size: float
    top1_in_frontier: float


@torch.no_grad()
def step_report(
    model: MeasurementChain,
    queries: QuerySet,
    schedule: list[CollapseSpec],
    *,
    generator: torch.Generator | None = None,
) -> list[StepReport]:
    """Read the index layer at every intermediate hop of one chain."""

    was_training = model.training
    model.eval()
    trace = model(queries, schedule, record=True, generator=generator)
    reports = []
    for hop, probabilities in enumerate(trace.step_probabilities, start=1):
        mask = model.graph.frontier_masks[hop].to(probabilities.device)[queries.start_position]
        mass = (probabilities * mask.float()).sum(-1)
        entropy = index_entropy(probabilities)
        top1 = probabilities.argmax(-1)
        reports.append(
            StepReport(
                hop=hop,
                frontier_mass=float(mass.mean()),
                frontier_chance=float(mask.float().mean(-1).mean()),
                entropy=float(entropy.mean()),
                max_entropy=float(torch.tensor(float(probabilities.shape[-1])).log()),
                effective_alternatives=float(entropy.exp().mean()),
                frontier_size=float(mask.sum(-1).float().mean()),
                top1_in_frontier=float(mask.gather(-1, top1.unsqueeze(-1)).float().mean()),
            )
        )
    model.train(was_training)
    return reports


@torch.no_grad()
def monte_carlo_convergence(
    model: MeasurementChain,
    queries: QuerySet,
    hop: int,
    sample_counts: tuple[int, ...],
    *,
    trials: int = 32,
    generator: torch.Generator | None = None,
) -> list[dict[str, float]]:
    r"""How fast an ``M``-draw collapse approaches the exact expected feedback.

    The feedback an ``M``-sample measurement applies is a Monte-Carlo estimate of
    the feedback ``expected`` applies exactly, so the distance between them should
    fall as :math:`M^{-1/2}`. This is the sense in which continuous thought is the
    mean-field limit of a chain of discrete measurements.
    """

    was_training = model.training
    model.eval()
    q, probabilities, candidates = _state_at_hop(model, queries, hop)
    embeddings = model.tb.A[:, candidates].T
    exact = probabilities @ embeddings

    rows = []
    for samples in sample_counts:
        spec = CollapseSpec(mode="sample", samples=samples)
        distances = []
        for _ in range(trials):
            weights, _ = collapse_weights(
                torch.log(probabilities.clamp_min(1e-30)), spec, generator=generator
            )
            distances.append((weights @ embeddings - exact).norm(dim=-1))
        stacked = torch.stack(distances)
        rows.append(
            {
                "samples": float(samples),
                "distance": float(stacked.mean()),
                "distance_std": float(stacked.std()),
                "relative": float(stacked.mean() / exact.norm(dim=-1).mean().clamp_min(1e-9)),
            }
        )
    model.train(was_training)
    return rows


@torch.no_grad()
def jensen_gap(
    model: MeasurementChain,
    queries: QuerySet,
    hop: int,
    feedback_gates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    *,
    retain_gate: float | None = None,
) -> list[dict[str, float]]:
    r"""Exact gap between evolving the mean state and averaging the evolved states.

    For each candidate :math:`k` the collapsed state is
    :math:`\alpha q + \beta a_k`; the degenerate alternative is
    :math:`\alpha q + \beta\sum_k p_k a_k`. Both are pushed through the evolution
    operator and compared. A second-order expansion predicts the gap grows like
    :math:`\beta^2\operatorname{tr}(\nabla^2\Phi\,\operatorname{Cov}_p[a])`, so the
    spread :math:`\operatorname{tr}\operatorname{Cov}_p[a]`, which equals
    :math:`\sum_k p_k\lVert a_k\rVert^2 - \lVert\bar a\rVert^2`, is reported
    alongside as the state-independent part of that prediction.
    """

    was_training = model.training
    model.eval()
    alpha = model.retain_gate if retain_gate is None else retain_gate
    q, probabilities, candidates = _state_at_hop(model, queries, hop)
    embeddings = model.tb.A[:, candidates].T
    mean_embedding = probabilities @ embeddings
    spread = (
        (probabilities * embeddings.pow(2).sum(-1)).sum(-1) - mean_embedding.pow(2).sum(-1)
    ).clamp_min(0.0)
    entropy = index_entropy(probabilities)

    rows = []
    for beta in feedback_gates:
        evolved_mean, _ = model.tb.evolve(alpha * q + beta * mean_embedding)
        # Exact expectation over the candidate set: every outcome is evolved and
        # then averaged under p, so no sampling error enters the gap.
        averaged = torch.zeros_like(evolved_mean)
        for position in range(embeddings.shape[0]):
            evolved, _ = model.tb.evolve(alpha * q + beta * embeddings[position])
            averaged = averaged + probabilities[:, position : position + 1] * evolved
        gap = (evolved_mean - averaged).norm(dim=-1)
        scale = evolved_mean.norm(dim=-1).clamp_min(1e-9)
        rows.append(
            {
                "feedback_gate": beta,
                "gap": float(gap.mean()),
                "gap_std": float(gap.std()),
                "relative_gap": float((gap / scale).mean()),
                "entropy": float(entropy.mean()),
                "spread": float(spread.mean()),
                "predicted_scale": float((beta**2) * spread.mean()),
            }
        )
    model.train(was_training)
    return rows


@torch.no_grad()
def jensen_gap_by_entropy(
    model: MeasurementChain,
    queries: QuerySet,
    hop: int,
    *,
    feedback_gate: float = 1.0,
    bins: int = 8,
) -> list[dict[str, float]]:
    """Per-query Jensen gap bucketed by how uncertain that query's step was."""

    was_training = model.training
    model.eval()
    alpha = model.retain_gate
    q, probabilities, candidates = _state_at_hop(model, queries, hop)
    embeddings = model.tb.A[:, candidates].T
    mean_embedding = probabilities @ embeddings
    evolved_mean, _ = model.tb.evolve(alpha * q + feedback_gate * mean_embedding)
    averaged = torch.zeros_like(evolved_mean)
    for position in range(embeddings.shape[0]):
        evolved, _ = model.tb.evolve(alpha * q + feedback_gate * embeddings[position])
        averaged = averaged + probabilities[:, position : position + 1] * evolved
    gap = (evolved_mean - averaged).norm(dim=-1)
    spread = (
        (probabilities * embeddings.pow(2).sum(-1)).sum(-1) - mean_embedding.pow(2).sum(-1)
    ).clamp_min(0.0)
    entropy = index_entropy(probabilities)

    order = entropy.argsort()
    chunks = torch.chunk(order, bins)
    rows = []
    for chunk in chunks:
        if chunk.numel() == 0:
            continue
        rows.append(
            {
                "entropy": float(entropy[chunk].mean()),
                "gap": float(gap[chunk].mean()),
                "gap_std": float(gap[chunk].std()),
                "spread": float(spread[chunk].mean()),
                "count": float(chunk.numel()),
            }
        )
    model.train(was_training)
    return rows


@torch.no_grad()
def jensen_gap_by_temperature(
    model: MeasurementChain,
    queries: QuerySet,
    hop: int,
    temperatures: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0),
    *,
    feedback_gate: float = 1.0,
) -> list[dict[str, float]]:
    r"""The Jensen gap as the candidate distribution is driven towards a point mass.

    Bucketing a trained chain's own steps by entropy does not test
    Corollary~1: on a search task the chain is never confident, so
    :math:`\operatorname{tr}\operatorname{Cov}_p[a]` sits near its maximum at every
    step and the prediction has no range to be tested over. Tempering the chain's own
    scores sweeps the spread across its whole range using the states the model
    actually visits, which is what makes the vanishing-at-a-point-mass claim
    falsifiable here.
    """

    was_training = model.training
    model.eval()
    alpha = model.retain_gate
    q, _, candidates = _state_at_hop(model, queries, hop)
    scores = model.tb.index_scores(q, candidates)
    embeddings = model.tb.A[:, candidates].T

    rows = []
    for temperature in temperatures:
        probabilities = torch.softmax(scores / temperature, dim=-1)
        mean_embedding = probabilities @ embeddings
        evolved_mean, _ = model.tb.evolve(alpha * q + feedback_gate * mean_embedding)
        averaged = torch.zeros_like(evolved_mean)
        for position in range(embeddings.shape[0]):
            evolved, _ = model.tb.evolve(alpha * q + feedback_gate * embeddings[position])
            averaged = averaged + probabilities[:, position : position + 1] * evolved
        gap = (evolved_mean - averaged).norm(dim=-1)
        spread = (
            (probabilities * embeddings.pow(2).sum(-1)).sum(-1) - mean_embedding.pow(2).sum(-1)
        ).clamp_min(0.0)
        rows.append(
            {
                "temperature": temperature,
                "gap": float(gap.mean()),
                "gap_std": float(gap.std()),
                "spread": float(spread.mean()),
                "entropy": float(index_entropy(probabilities).mean()),
                "max_probability": float(probabilities.max(dim=-1).values.mean()),
            }
        )
    model.train(was_training)
    return rows


@torch.no_grad()
def zeno_trajectory(
    model: MeasurementChain,
    queries: QuerySet,
    hop: int,
    *,
    repeats: int = 12,
    retain_gate: float = 1.0,
    feedback_gate: float = 1.0,
    spec: CollapseSpec | None = None,
) -> list[dict[str, float]]:
    r"""Repeated measurement of one concept window with no evolution in between.

    Each repeat applies :math:`q \leftarrow \alpha q + \beta\sum_k w_k a_k` to the
    same window. Because the feedback adds the measured embedding back into the
    state, the score of whatever was measured rises, and the distribution sharpens
    towards a fixed point. This is the analogue of holding a state still by
    measuring it repeatedly, and it costs evolution steps nothing.
    """

    was_training = model.training
    model.eval()
    spec = spec or CollapseSpec(mode="expected")
    q, probabilities, candidates = _state_at_hop(model, queries, hop)
    embeddings = model.tb.A[:, candidates].T
    mask = model.graph.frontier_masks[hop].to(q.device)[queries.start_position].float()
    initial_mode = probabilities.argmax(-1)

    rows = []
    for repeat in range(repeats + 1):
        scores = model.tb.index_scores(q, candidates)
        probabilities = torch.softmax(scores, dim=-1)
        entropy = index_entropy(probabilities)
        rows.append(
            {
                "repeat": float(repeat),
                "entropy": float(entropy.mean()),
                "effective_alternatives": float(entropy.exp().mean()),
                "frontier_mass": float((probabilities * mask).sum(-1).mean()),
                "mode_probability": float(
                    probabilities.gather(-1, initial_mode.unsqueeze(-1)).mean()
                ),
                "mode_agreement": float((probabilities.argmax(-1) == initial_mode).float().mean()),
            }
        )
        if repeat == repeats:
            break
        weights, _ = collapse_weights(scores, spec)
        feedback = weights @ embeddings
        if model.feedback_norm == "unit":
            feedback = feedback / feedback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q = retain_gate * q + feedback_gate * feedback * model.write_scale
    model.train(was_training)
    return rows


@torch.no_grad()
def _state_at_hop(
    model: MeasurementChain, queries: QuerySet, hop: int
) -> tuple[Float[Tensor, "queries state"], Float[Tensor, "queries candidates"], Tensor]:
    """Run the chain up to ``hop`` and return the pre-feedback state and beliefs."""

    q = model.write_query(queries)
    context = None
    for current in range(1, hop + 1):
        q, context = model.tb.evolve(q, context)
        if current == hop:
            break
        candidates = model.candidates_at(current)
        scores = model.tb.index_scores(q, candidates)
        weights, _ = collapse_weights(scores, CollapseSpec(mode="expected"))
        feedback = weights @ model.tb.A[:, candidates].T
        if model.feedback_norm == "unit":
            feedback = feedback / feedback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q = model.retain_gate * q + model.feedback_gate * feedback * model.write_scale
    candidates = model.candidates_at(hop)
    probabilities = torch.softmax(model.tb.index_scores(q, candidates), dim=-1)
    return q, probabilities, candidates
