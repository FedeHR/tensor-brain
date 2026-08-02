"""Perception models for the initial PVSG Section 6 comparison."""

from typing import Literal

from jaxtyping import Float, Int
from torch import Tensor, nn

from tb import TensorBrain
from tb.evolution import Evolution

FeedbackMode = Literal["p-sa", "p-samp"]


class PDirect(nn.Module):
    """Independently score each label from its own bottom-up feature."""

    def __init__(self, state_dim: int, num_indices: int) -> None:
        super().__init__()
        self.brain = TensorBrain(state_dim, num_indices, evolution=None)

    def forward(
        self,
        subject_features: Float[Tensor, "*batch state"],
        object_features: Float[Tensor, "*batch state"],
        union_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        predicate_candidates: Int[Tensor, " predicates"],
    ) -> dict[str, Tensor]:
        """Return independent identity and predicate logits.

        Scene evidence is absent because P-Direct has no scene target and does not
        transport scene information into later readouts.
        """

        subject_q = self.brain.integrate_input(
            subject_features.new_zeros(subject_features.shape), subject_features
        )
        object_q = self.brain.integrate_input(
            object_features.new_zeros(object_features.shape), object_features
        )
        predicate_q = self.brain.integrate_input(
            union_features.new_zeros(union_features.shape), union_features
        )
        return {
            "subject_identity_logits": self.brain.index_scores(subject_q, identity_candidates),
            "object_identity_logits": self.brain.index_scores(object_q, identity_candidates),
            "predicate_logits": self.brain.index_scores(predicate_q, predicate_candidates),
        }


class IntegralTB(nn.Module):
    """Run the paper's four evidence windows with identity feedback."""

    def __init__(
        self,
        state_dim: int,
        num_indices: int,
        evolution: Evolution,
    ) -> None:
        super().__init__()
        self.brain = TensorBrain(state_dim, num_indices, evolution)

    def forward(
        self,
        scene_features: Float[Tensor, "*batch state"],
        subject_features: Float[Tensor, "*batch state"],
        object_features: Float[Tensor, "*batch state"],
        union_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        predicate_candidates: Int[Tensor, " predicates"],
        *,
        feedback_mode: FeedbackMode = "p-sa",
    ) -> dict[str, Tensor]:
        """Return logits from the explicit Section 6 perception schedule.

        Training uses differentiable P-SA feedback. P-Samp is the evaluation-only
        winner-take-all condition of the same checkpoint.
        """

        if feedback_mode == "p-samp" and self.training:
            raise ValueError("P-Samp is evaluation-only; train the checkpoint with P-SA")
        if feedback_mode not in ("p-sa", "p-samp"):
            raise ValueError("feedback_mode must be 'p-sa' or 'p-samp'")

        # Complete-scene window.
        q = self.brain.integrate_input(
            scene_features.new_zeros(scene_features.shape), scene_features
        )
        q, context = self.brain.evolve(q)

        # Subject window: identify, feed the identity back, then evolve.
        q = self.brain.integrate_input(q, subject_features)
        subject_identity_logits = self.brain.index_scores(q, identity_candidates)
        if feedback_mode == "p-sa":
            q, _ = self.brain.attend(q, identity_candidates)
        else:
            q, _, _ = self.brain.measure(q, identity_candidates, selection="argmax")
        q, context = self.brain.evolve(q, context)

        # Object window: identify, feed the identity back, then evolve.
        q = self.brain.integrate_input(q, object_features)
        object_identity_logits = self.brain.index_scores(q, identity_candidates)
        if feedback_mode == "p-sa":
            q, _ = self.brain.attend(q, identity_candidates)
        else:
            q, _, _ = self.brain.measure(q, identity_candidates, selection="argmax")
        q, context = self.brain.evolve(q, context)

        # Predicate window.
        q = self.brain.integrate_input(q, union_features)
        predicate_logits = self.brain.index_scores(q, predicate_candidates)

        return {
            "subject_identity_logits": subject_identity_logits,
            "object_identity_logits": object_identity_logits,
            "predicate_logits": predicate_logits,
        }
