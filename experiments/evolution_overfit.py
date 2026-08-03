"""A tiny deterministic overfitting diagnostic for evolution operators.

The four inputs form an XOR problem. Direct index scoring is linear in
``sigmoid(q)``, so it cannot solve the task. The hidden evolution operator must
create the nonlinear decision boundary before the index layer decodes it.

This is intentionally an overfitting diagnostic, not a performance benchmark.
It answers whether an evolution variant can learn a known finite problem under
the same optimizer, parameter dimensions, and training schedule.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

from tb import OriginalTBDynamicContext, QTBEvolution, ReLUEvolution, TensorBrain

EvolutionVariant = Literal["original", "qtb-sigmoid", "qtb-relu"]


@dataclass(frozen=True)
class XORProblem:
    input_q: Float[Tensor, "samples state"]
    target: Int[Tensor, " samples"]


@dataclass(frozen=True)
class OverfitResult:
    variant: EvolutionVariant
    final_loss: float
    accuracy: float
    loss_history: tuple[float, ...]


def make_xor_problem() -> XORProblem:
    """Return the smallest non-linearly separable evolution problem."""

    return XORProblem(
        input_q=torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        ),
        target=torch.tensor([0, 1, 1, 0]),
    )


def make_evolution(
    variant: EvolutionVariant, state_dim: int, hidden_dim: int
) -> OriginalTBDynamicContext | QTBEvolution | ReLUEvolution:
    """Construct one named backend for the controlled comparison."""

    if variant == "original":
        return OriginalTBDynamicContext(state_dim, hidden_dim)
    if variant == "qtb-sigmoid":
        return QTBEvolution(state_dim, hidden_dim)
    if variant == "qtb-relu":
        return ReLUEvolution(state_dim, hidden_dim)
    raise ValueError(f"unknown evolution variant: {variant}")


def train_variant(
    variant: EvolutionVariant,
    *,
    epochs: int = 1_000,
    learning_rate: float = 0.05,
    hidden_dim: int = 8,
    seed: int = 0,
) -> OverfitResult:
    """Overfit one evolution variant to the complete four-example problem."""

    torch.manual_seed(seed)
    problem = make_xor_problem()
    brain = TensorBrain(
        2,
        2,
        make_evolution(variant, 2, hidden_dim),
        score_mode="learned-bias",
    )
    optimizer = torch.optim.Adam(brain.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _epoch in range(epochs):
        optimizer.zero_grad()
        evolved_q, _context = brain.evolve(problem.input_q)
        scores = brain.index_scores(evolved_q)
        loss = F.cross_entropy(scores, problem.target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        evolved_q, _context = brain.evolve(problem.input_q)
        scores = brain.index_scores(evolved_q)
        accuracy = float((scores.argmax(dim=-1) == problem.target).float().mean())

    return OverfitResult(variant, losses[-1], accuracy, tuple(losses))


if __name__ == "__main__":
    for _variant in ("original", "qtb-sigmoid", "qtb-relu"):
        _result = train_variant(_variant)
        print(
            f"{_result.variant}: loss={_result.final_loss:.6f}, "
            f"accuracy={_result.accuracy:.3f}"
        )
