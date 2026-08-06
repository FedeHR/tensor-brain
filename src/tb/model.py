"""The shared representation and index layers of the Tensor Brain."""

from collections.abc import Sequence
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from tb.evolution import Evolution

ScoreMode = Literal["direct", "centered", "learned-bias", "softplus-bias"]


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
        *,
        score_mode: ScoreMode = "direct",
    ) -> None:
        super().__init__()
        if state_dim <= 0 or num_indices <= 0:
            raise ValueError("state_dim and num_indices must be positive")
        self.state_dim = state_dim
        self.num_indices = num_indices
        self.evolution = evolution
        self.score_mode = score_mode
        self.A = nn.Parameter(torch.empty(state_dim, num_indices))
        if score_mode == "learned-bias":
            self.a0 = nn.Parameter(torch.zeros(num_indices))
        elif score_mode in ("direct", "centered", "softplus-bias"):
            self.register_parameter("a0", None)
        else:
            raise ValueError(f"unknown score mode: {score_mode}")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # A is a bank of bidirectional index embeddings, not a conventional
        # [fan_out, fan_in] linear weight. This scale gives every column an
        # expected squared norm of one, independently of num_indices.
        nn.init.normal_(self.A, mean=0.0, std=self.state_dim**-0.5)
        if self.a0 is not None:
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
        r"""Score candidate indices under the configured direct-score condition.

        ``candidates=None`` scores the complete global index layer.
        """

        indices = self._resolve_candidate_indices(candidates, q.device)
        gamma = torch.sigmoid(q)
        if self.score_mode == "centered":
            gamma = gamma - 0.5
        return gamma @ self.A[:, indices] + self.index_bias(indices)

    def index_bias(
        self,
        candidates: Int[Tensor, " indices"] | Sequence[int] | None = None,
    ) -> Float[Tensor, " indices"]:
        r"""Return the effective index offset for the configured score mode.

        ``softplus-bias`` is QTB's factorized-Bernoulli log-normalizer
        :math:`a_{0,k}=-\sum_l\operatorname{softplus}(a_{l,k})`. It is explicit
        in current-arXiv Equation (31) and follows from latest-draft Equations
        (25), (32)-(34).
        """

        indices = self._resolve_candidate_indices(candidates, self.A.device)
        if self.a0 is not None:
            return self.a0[indices]
        if self.score_mode == "softplus-bias":
            return -F.softplus(self.A[:, indices]).sum(dim=0)
        return self.A.new_zeros(indices.shape)

    def attend(
        self,
        q: Float[Tensor, "*batch state"],
        candidates: Int[Tensor, " indices"] | Sequence[int] | None = None,
        *,
        feedback_gate: Float[Tensor, "..."] | float = 1.0,
    ) -> tuple[
        Float[Tensor, "*batch state"],
        Float[Tensor, "*batch indices"],
    ]:
        r"""Apply deterministic expected index feedback.

        Returns ``(updated_q, probabilities)`` where
        :math:`q' = q + \beta \sum_k p_k a_k`.
        ``candidates=None`` attends over the complete global index layer.

        Current-arXiv Algorithm 2 writes this attention step without a gate. Accepting
        Algorithm 3's :math:`\beta` here makes injected magnitude one variable across
        both top-down operations, which is what lets a gate sweep reach the
        differentiable attention path and therefore act during training. The default
        reproduces the ungated equation exactly. Attention and measurement magnitudes
        are deliberately not separated; a distinct attention gate would be the finer
        factorization if the two ever need to move independently.
        """

        indices = self._resolve_candidate_indices(candidates, q.device)
        probabilities = torch.softmax(self.index_scores(q, indices), dim=-1)
        feedback = probabilities @ self.A[:, indices].T
        return q + feedback_gate * feedback, probabilities

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
