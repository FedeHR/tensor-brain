r"""The collapse dial: one feedback operator that contains CoT and latent CoT.

Every intermediate reasoning step in this study writes back into the pre-CBS with

.. math::
    q \leftarrow \alpha q + \beta \sum_k w_k a_k ,

and the conditions differ *only* in how the feedback weights :math:`w` are formed
from the candidate index scores. Two of these are already in the Tensor Brain
core: ``expected`` at temperature one is :meth:`tb.TensorBrain.attend`, and
``sample`` with a single draw is :meth:`tb.TensorBrain.measure`. QTB calls the
former a degenerate measurement whose outcome is not revealed, which is the sense
in which latent chain-of-thought and token chain-of-thought are the same operator
at two settings of one knob.

The two knobs are deliberately separate:

``temperature``
    sharpens or flattens the candidate distribution itself (bias);
``samples``
    how many Monte-Carlo draws estimate the expectation (variance). ``samples=1``
    is a single collapse, ``samples=inf`` is the exact expectation.

Discrete modes use a straight-through estimator so that a hard forward pass keeps
a usable gradient; the surrogate is always the tempered softmax, which is the
gradient the corresponding soft mode would have produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

CollapseMode = Literal["none", "pause", "expected", "sample", "argmax", "teacher"]


@dataclass(frozen=True)
class CollapseSpec:
    """How one reasoning step turns candidate scores into feedback weights."""

    mode: CollapseMode = "expected"
    temperature: float = 1.0
    samples: int = 1
    straight_through: bool = True

    def label(self) -> str:
        if self.mode in ("none", "pause", "teacher", "argmax"):
            return self.mode
        if self.mode == "expected":
            return f"expected(t={self.temperature:g})"
        return f"sample(t={self.temperature:g},M={self.samples})"


def candidate_probabilities(
    scores: Float[Tensor, "batch indices"],
    temperature: float = 1.0,
) -> Float[Tensor, "batch indices"]:
    """Tempered candidate distribution ``p = softmax(scores / temperature)``."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return torch.softmax(scores / temperature, dim=-1)


def collapse_weights(
    scores: Float[Tensor, "batch indices"],
    spec: CollapseSpec,
    *,
    teacher_distribution: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "batch indices"], Float[Tensor, "batch indices"]]:
    """Return ``(weights, probabilities)`` for one reasoning step.

    ``probabilities`` is always the untempered candidate distribution, so that the
    reported entropies and frontier masses describe the model's own belief rather
    than the sharpened distribution the feedback happened to use.
    """

    reference = torch.softmax(scores, dim=-1)

    if spec.mode in ("none", "pause"):
        # Neither reads the candidate distribution; the caller supplies the pause
        # direction itself, so the weights are unused.
        return torch.zeros_like(scores), reference

    tempered = candidate_probabilities(scores, spec.temperature)

    if spec.mode == "expected":
        return tempered, reference

    if spec.mode == "teacher":
        # The teacher is a *distribution*, not a single node. Writing back one gold
        # node would put a single path in the workspace while the step target is
        # the whole frontier reachable from the start, which is a target the state
        # cannot support. Feeding back the true frontier keeps the bootstrap
        # consistent with what the chain is asked to represent, and is identical
        # across every condition.
        if teacher_distribution is None:
            raise ValueError("teacher collapse requires teacher_distribution")
        return teacher_distribution, reference

    if spec.mode == "argmax":
        hard = torch.zeros_like(scores)
        hard.scatter_(-1, tempered.argmax(dim=-1, keepdim=True), 1.0)
        return _maybe_straight_through(hard, tempered, spec), reference

    if spec.mode == "sample":
        if spec.samples < 1:
            raise ValueError("samples must be at least one")
        # An M-draw average of one-hot vectors. M = 1 is a single measurement;
        # increasing M walks the estimator towards `expected` at the same
        # temperature, which is the self-consistency limit taken inside a step
        # rather than across whole trajectories.
        draws = torch.multinomial(
            tempered.reshape(-1, tempered.shape[-1]),
            num_samples=spec.samples,
            replacement=True,
            generator=generator,
        ).reshape(*tempered.shape[:-1], spec.samples)
        hard = torch.zeros_like(tempered)
        share = torch.full_like(draws, 1.0 / spec.samples, dtype=tempered.dtype)
        hard.scatter_add_(-1, draws, share)
        return _maybe_straight_through(hard, tempered, spec), reference

    raise ValueError(f"unknown collapse mode: {spec.mode}")


def _maybe_straight_through(
    hard: Float[Tensor, "batch indices"],
    soft: Float[Tensor, "batch indices"],
    spec: CollapseSpec,
) -> Float[Tensor, "batch indices"]:
    if not spec.straight_through:
        return hard
    return soft + (hard - soft).detach()


def index_entropy(probabilities: Float[Tensor, "batch indices"]) -> Float[Tensor, " batch"]:
    """Shannon entropy in nats of each row of a candidate distribution."""

    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return -(probabilities * safe.log()).sum(dim=-1)
