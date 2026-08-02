"""The shared representation and index layers of the Tensor Brain."""

from collections.abc import Sequence
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from tb.evolution import Evolution


class TensorBrain(nn.Module):
    r"""Shared Tensor Brain index and representation operations.

    ``A[:, k]`` is the embedding :math:`a_k` of global symbolic index ``k``.
    The cognitive brain state is always derived as ``gamma = sigmoid(q)``.
    """

    def __init__(
        self,
        state_dim: int,
        num_indices: int,
        evolution: Evolution | None,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or num_indices <= 0:
            raise ValueError("state_dim and num_indices must be positive")
        self.state_dim = state_dim
        self.num_indices = num_indices
        self.evolution = evolution
        self.A = nn.Parameter(torch.empty(state_dim, num_indices))
        self.a0 = nn.Parameter(torch.zeros(num_indices))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # A is a bank of bidirectional index embeddings, not a conventional
        # [fan_out, fan_in] linear weight. This scale gives every column an
        # expected squared norm of one, independently of num_indices.
        nn.init.normal_(self.A, mean=0.0, std=self.state_dim**-0.5)
        nn.init.zeros_(self.a0)

    def _resolve_candidate_indices(
        self,
        candidates: Int[Tensor, " indices"] | Sequence[int] | None,
        device: torch.device,
    ) -> Int[Tensor, " indices"]:
        """Resolve candidate IDs on ``q``'s device; ``None`` means all global indices."""

        if candidates is None:
            return torch.arange(self.num_indices, device=device)
        return torch.as_tensor(candidates, dtype=torch.long, device=device)

    def integrate_input(
        self,
        q: Float[Tensor, "*batch state"],
        input_drive: Float[Tensor, "*batch state"],
        *,
        input_gate: Float[Tensor, "..."] | float = 1.0,
    ) -> Float[Tensor, "*batch state"]:
        r"""Add an already mapped input contribution to the pre-CBS.

        ``input_drive`` represents :math:`g(\nu)` and must already be in
        representation-layer coordinates. Feature extraction, normalization,
        and any projection into ``state_dim`` remain outside the Tensor Brain
        core. The update is :math:`q' = q + \mu g(\nu)`, where ``input_gate``
        is :math:`\mu` and may be any value broadcastable to ``q``.
        """

        return q + input_gate * input_drive

    def index_scores(
        self,
        q: Float[Tensor, "*batch state"],
        candidates: Int[Tensor, " indices"] | Sequence[int] | None = None,
    ) -> Float[Tensor, "*batch indices"]:
        r"""Return :math:`a_{0,k} + a_k^\top\sigma(q)` for candidate indices.

        ``candidates=None`` scores the complete global index layer.
        """

        indices = self._resolve_candidate_indices(candidates, q.device)
        gamma = torch.sigmoid(q)
        return gamma @ self.A[:, indices] + self.a0[indices]

    def attend(
        self,
        q: Float[Tensor, "*batch state"],
        candidates: Int[Tensor, " indices"] | Sequence[int] | None = None,
    ) -> tuple[
        Float[Tensor, "*batch state"],
        Float[Tensor, "*batch indices"],
    ]:
        r"""Apply deterministic expected index feedback.

        Returns ``(updated_q, probabilities)`` where
        :math:`q' = q + \sum_k p_k a_k`.
        ``candidates=None`` attends over the complete global index layer.
        """

        indices = self._resolve_candidate_indices(candidates, q.device)
        probabilities = torch.softmax(self.index_scores(q, indices), dim=-1)
        feedback = probabilities @ self.A[:, indices].T
        return q + feedback, probabilities

    def measure(
        self,
        q: Float[Tensor, "*batch state"],
        candidates: Int[Tensor, " indices"] | Sequence[int] | None = None,
        *,
        outcome: Int[Tensor, "*batch"] | Int[Tensor, ""] | int | None = None,
        selection: Literal["teacher", "sample", "argmax"] = "sample",
        retain_gate: Float[Tensor, "..."] | float = 1.0,
        feedback_gate: Float[Tensor, "..."] | float = 1.0,
    ) -> tuple[
        Float[Tensor, "*batch state"],
        Int[Tensor, "*batch"],
        Float[Tensor, "*batch indices"],
    ]:
        r"""Generate or apply an index outcome and update the pre-CBS.

        Candidate probabilities use local positions ``0, ..., K-1``. A selected
        position is mapped back to its global index ``k`` before retrieving
        ``A[:, k]``. ``selection="teacher"`` uses the supplied global ``outcome``;
        ``selection="sample"`` draws from the candidate distribution; and
        ``selection="argmax"`` implements the paper's winner-take-all
        approximation. ``candidates=None`` measures over the complete global
        index layer. The update is
        :math:`q' = \alpha q + \beta a_k`. Gates may be tensors, including
        learned parameters; their parameterization is owned by the experiment.
        """

        indices = self._resolve_candidate_indices(candidates, q.device)
        probabilities = torch.softmax(self.index_scores(q, indices), dim=-1)
        if selection == "teacher":
            outcome_index = torch.as_tensor(outcome, dtype=torch.long, device=q.device)
            if outcome_index.ndim == 0:
                outcome_index = outcome_index.expand(q.shape[:-1])
            if bool((~torch.isin(outcome_index, indices)).any()):
                raise ValueError("outcome must be one of the candidate global indices")
        elif selection == "sample":
            position = torch.distributions.Categorical(probabilities).sample()
            outcome_index = indices[position]
        elif selection == "argmax":
            position = probabilities.argmax(dim=-1)
            outcome_index = indices[position]
        else:
            raise ValueError(
                "measurement selection mode not supported, must be one of sample, argmax"
            )
        outcome_embedding = self.A.T[outcome_index]
        q_next = retain_gate * q + feedback_gate * outcome_embedding
        return q_next, outcome_index, probabilities

    def evolve(
        self,
        q: Float[Tensor, "*batch state"],
        context: Float[Tensor, "*batch context"] | None = None,
    ) -> tuple[
        Float[Tensor, "*batch state"],
        Float[Tensor, "*batch context"] | None,
    ]:
        """Move between concept windows using the configured evolution backend."""

        if self.evolution is None:
            raise RuntimeError("this Tensor Brain has no evolution operator")
        return self.evolution(q, context)
