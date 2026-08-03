"""Perception models for the initial PVSG Section 6 comparison."""

from collections.abc import Mapping
from typing import Literal, NotRequired, TypedDict

from jaxtyping import Float, Int
from torch import Tensor, nn

from tb import TensorBrain
from tb.evolution import Evolution

FeedbackMode = Literal["p-sa", "p-samp"]
CategoryCandidates = Mapping[str, Int[Tensor, " categories"]]


class PerceptionOutputs(TypedDict):
    """Logits at the explicit identity, category, and predicate readout points."""

    subject_identity_logits: Tensor
    object_identity_logits: Tensor
    subject_category_logits: dict[str, Tensor]
    object_category_logits: dict[str, Tensor]
    predicate_logits: Tensor
    trace: NotRequired[dict[str, dict[str, Tensor]]]


def _category_logits(
    brain: TensorBrain,
    q: Float[Tensor, "*batch state"],
    candidates_by_level: CategoryCandidates | None,
) -> dict[str, Tensor]:
    if candidates_by_level is None:
        return {}
    return {
        level: brain.index_scores(q, candidates)
        for level, candidates in candidates_by_level.items()
    }


def _identity_feedback(
    brain: TensorBrain,
    q: Float[Tensor, "*batch state"],
    identity_candidates: Int[Tensor, " identities"],
    feedback_mode: FeedbackMode,
) -> tuple[Float[Tensor, "*batch state"], Float[Tensor, "*batch identities"]]:
    if feedback_mode == "p-sa":
        return brain.attend(q, identity_candidates)
    q_next, _outcome, probabilities = brain.measure(
        q, identity_candidates, selection="argmax"
    )
    return q_next, probabilities


def _feedback_vectors(
    brain: TensorBrain,
    probabilities: Float[Tensor, "*batch identities"],
    identity_candidates: Int[Tensor, " identities"],
) -> tuple[Float[Tensor, "*batch state"], Float[Tensor, "*batch state"]]:
    candidate_embeddings = brain.A[:, identity_candidates].T
    expected = probabilities @ candidate_embeddings
    winner_indices = identity_candidates[probabilities.argmax(dim=-1)]
    winner = brain.A.T[winner_indices]
    return expected, winner


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
        *,
        category_candidates: CategoryCandidates | None = None,
        return_trace: bool = False,
    ) -> PerceptionOutputs:
        """Return independent identity, category, and predicate logits.

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
        outputs: PerceptionOutputs = {
            "subject_identity_logits": self.brain.index_scores(subject_q, identity_candidates),
            "object_identity_logits": self.brain.index_scores(object_q, identity_candidates),
            "subject_category_logits": _category_logits(
                self.brain, subject_q, category_candidates
            ),
            "object_category_logits": _category_logits(
                self.brain, object_q, category_candidates
            ),
            "predicate_logits": self.brain.index_scores(predicate_q, predicate_candidates),
        }
        if return_trace:
            outputs["trace"] = {
                "subject": {
                    "input_drive": subject_features,
                    "q_after_input": subject_q,
                },
                "object": {
                    "input_drive": object_features,
                    "q_after_input": object_q,
                },
                "predicate": {
                    "input_drive": union_features,
                    "q_after_input": predicate_q,
                },
            }
        return outputs


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
        category_candidates: CategoryCandidates | None = None,
        feedback_mode: FeedbackMode = "p-sa",
        return_trace: bool = False,
    ) -> PerceptionOutputs:
        """Return logits from the explicit Section 6 perception schedule.

        Training uses differentiable P-SA feedback. P-Samp is the evaluation-only
        winner-take-all condition of the same checkpoint.
        """

        if feedback_mode == "p-samp" and self.training:
            raise ValueError("P-Samp is evaluation-only; train the checkpoint with P-SA")
        if feedback_mode not in ("p-sa", "p-samp"):
            raise ValueError("feedback_mode must be 'p-sa' or 'p-samp'")

        # Complete-scene window.
        trace: dict[str, dict[str, Tensor]] | None = {} if return_trace else None
        q_before_input = scene_features.new_zeros(scene_features.shape)
        q = self.brain.integrate_input(q_before_input, scene_features)
        q_after_input = q
        q, context = self.brain.evolve(q)
        if trace is not None:
            trace["scene"] = {
                "input_drive": scene_features,
                "q_before_input": q_before_input,
                "q_after_input": q_after_input,
                "q_after_evolution": q,
            }

        # Subject window: identify, feed the identity back, decode unary categories, then evolve.
        q_before_input = q
        q = self.brain.integrate_input(q, subject_features)
        q_after_input = q
        subject_identity_logits = self.brain.index_scores(q, identity_candidates)
        q, subject_probabilities = _identity_feedback(
            self.brain, q, identity_candidates, feedback_mode
        )
        q_after_feedback = q
        subject_category_logits = _category_logits(self.brain, q, category_candidates)
        q, context = self.brain.evolve(q, context)
        if trace is not None:
            subject_expected, subject_winner = _feedback_vectors(
                self.brain, subject_probabilities, identity_candidates
            )
            trace["subject"] = {
                "input_drive": subject_features,
                "q_before_input": q_before_input,
                "q_after_input": q_after_input,
                "identity_probabilities": subject_probabilities,
                "expected_feedback": subject_expected,
                "winner_feedback": subject_winner,
                "applied_feedback": q_after_feedback - q_after_input,
                "q_after_feedback": q_after_feedback,
                "q_after_evolution": q,
            }

        # Object window: identify, feed the identity back, decode unary categories, then evolve.
        q_before_input = q
        q = self.brain.integrate_input(q, object_features)
        q_after_input = q
        object_identity_logits = self.brain.index_scores(q, identity_candidates)
        q, object_probabilities = _identity_feedback(
            self.brain, q, identity_candidates, feedback_mode
        )
        q_after_feedback = q
        object_category_logits = _category_logits(self.brain, q, category_candidates)
        q, context = self.brain.evolve(q, context)
        if trace is not None:
            object_expected, object_winner = _feedback_vectors(
                self.brain, object_probabilities, identity_candidates
            )
            trace["object"] = {
                "input_drive": object_features,
                "q_before_input": q_before_input,
                "q_after_input": q_after_input,
                "identity_probabilities": object_probabilities,
                "expected_feedback": object_expected,
                "winner_feedback": object_winner,
                "applied_feedback": q_after_feedback - q_after_input,
                "q_after_feedback": q_after_feedback,
                "q_after_evolution": q,
            }

        # Predicate window.
        q_before_input = q
        q = self.brain.integrate_input(q, union_features)
        predicate_logits = self.brain.index_scores(q, predicate_candidates)
        if trace is not None:
            trace["predicate"] = {
                "input_drive": union_features,
                "q_before_input": q_before_input,
                "q_after_input": q,
            }

        outputs: PerceptionOutputs = {
            "subject_identity_logits": subject_identity_logits,
            "object_identity_logits": object_identity_logits,
            "subject_category_logits": subject_category_logits,
            "object_category_logits": object_category_logits,
            "predicate_logits": predicate_logits,
        }
        if trace is not None:
            outputs["trace"] = trace
        return outputs
