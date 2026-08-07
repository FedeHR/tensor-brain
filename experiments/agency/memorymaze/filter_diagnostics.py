r"""Diagnostics that tie a filter's probe score to a mechanism.

A grid of probe scores says which variant won. These say *why*, and they are the
difference between a result and a leaderboard. Each one is cheap and is
accumulated over the same evaluation pass.

``index_entropy``
    Mean entropy of the candidate softmax. It sets the regime: the drift
    correction cancels the write entirely as the softmax sharpens, so a low
    entropy predicts ``corrected`` behaving like ``none``, and a high entropy
    predicts it behaving like ``raw`` with the bias removed. Without this number
    an ordering between those three variants cannot be interpreted.

``drift_norm``
    :math:`\lVert \sum_j p_j a_j \rVert`, the systematic component of the raw
    write. This is the quantity ``corrected`` subtracts, so it measures how much
    work the correction is doing.

``saturated_fraction``
    Fraction of representation units with :math:`\lvert q \rvert` past the point
    where :math:`\sigma` is flat. This is the mechanism three earlier studies
    measured behind the ``alpha`` result: a saturated unit passes no gradient and
    carries no information, so a state that saturates has stopped filtering
    whatever its score says.

``log_partition_variance``
    Monte-Carlo :math:`\mathrm{Var}[\log Z]` over samples
    :math:`i \sim \mathrm{Bernoulli}(\gamma)`, where
    :math:`\log Z = \operatorname{logsumexp}_k (a_{0,k} + i^\top a_k)`. The
    factorized-Bernoulli index distribution treats :math:`\log Z` as if it were
    constant in :math:`i`; this measures how badly that holds, which is the
    approximation error the score modes differ over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor

# `sigmoid` is flat past about 4 in absolute value: sigma(4) = 0.982, and its
# derivative there is under 0.018. A unit beyond that is effectively pinned.
SATURATION_THRESHOLD = 4.0


def index_entropy(probabilities: Float[Tensor, "batch indices"]) -> float:
    """Mean entropy of the candidate distribution, in nats."""

    safe = probabilities.clamp_min(1e-12)
    return float(-(probabilities * safe.log()).sum(dim=-1).mean())


def drift_norm(
    probabilities: Float[Tensor, "batch indices"], embeddings: Float[Tensor, "state indices"]
) -> float:
    """Mean norm of the expected write, ``|| A p ||``."""

    return float((probabilities @ embeddings.T).norm(dim=-1).mean())


def saturated_fraction(
    q: Float[Tensor, "batch state"], threshold: float = SATURATION_THRESHOLD
) -> float:
    """Fraction of units whose CBS activation is pinned against 0 or 1."""

    return float((q.abs() > threshold).double().mean())


@torch.no_grad()
def log_partition_variance(
    q: Float[Tensor, "batch state"],
    embeddings: Float[Tensor, "state indices"],
    bias: Float[Tensor, " indices"],
    *,
    samples: int = 32,
    generator: torch.Generator | None = None,
) -> float:
    r"""Monte-Carlo :math:`\mathrm{Var}[\log Z]` under :math:`i \sim \mathrm{Bern}(\gamma)`."""

    gamma = torch.sigmoid(q)
    draws = torch.bernoulli(
        gamma.unsqueeze(0).expand(samples, *gamma.shape).contiguous(), generator=generator
    )
    scores = draws @ embeddings + bias
    return float(torch.logsumexp(scores, dim=-1).var(dim=0).mean())


@dataclass
class DiagnosticAccumulator:
    """Running means over an evaluation pass."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + value
        self.counts[name] = self.counts.get(name, 0) + 1

    def update(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.add(name, value)

    def means(self) -> dict[str, float]:
        return {
            name: self.totals[name] / self.counts[name]
            for name in sorted(self.totals)
            if self.counts[name]
        }


def step_diagnostics(
    trace, brain, *, generator: torch.Generator | None = None
) -> dict[str, float]:
    """Everything measurable about one filter step.

    Index-layer quantities are omitted rather than zero-filled for a model with
    no index layer, so an absent measurement never averages in as a value.
    """

    values = {
        "state_norm": float(trace.q.norm(dim=-1).mean()),
        "saturated_fraction": saturated_fraction(trace.q),
    }
    if trace.index_probabilities is None or brain is None:
        return values
    candidates = brain.latent_bank
    embeddings = brain.brain.A[:, candidates]
    values["index_entropy"] = index_entropy(trace.index_probabilities)
    values["drift_norm"] = drift_norm(trace.index_probabilities, embeddings)
    values["log_partition_variance"] = log_partition_variance(
        trace.q, embeddings, brain.brain.index_bias(candidates), generator=generator
    )
    return values
