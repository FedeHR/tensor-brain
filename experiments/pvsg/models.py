"""Perception models for the initial PVSG Section 6 comparison."""

from collections.abc import Mapping
from typing import Literal, NotRequired, TypedDict

from jaxtyping import Float, Int
from torch import Tensor, nn

from tb import ScoreMode, TensorBrain
from tb.evolution import Evolution

FeedbackMode = Literal["p-sa", "p-samp", "none"]
CategoryCandidates = Mapping[str, Int[Tensor, " categories"]]
Trace = dict[str, dict[str, Tensor]]


class PerceptionOutputs(TypedDict):
    """Logits at the explicit identity, category, and predicate readout points."""

    subject_identity_logits: Tensor
    object_identity_logits: Tensor
    subject_category_logits: dict[str, Tensor]
    object_category_logits: dict[str, Tensor]
    predicate_logits: Tensor
    trace: NotRequired[Trace]


class ObjectOutputs(TypedDict):
    """Logits from one scene-and-entity observation."""

    identity_logits: Tensor
    category_logits: dict[str, Tensor]
    trace: NotRequired[Trace]


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


def _index_feedback(
    brain: TensorBrain,
    q: Float[Tensor, "*batch state"],
    candidates: Int[Tensor, " indices"],
    feedback_mode: FeedbackMode,
) -> tuple[Float[Tensor, "*batch state"], Float[Tensor, "*batch indices"]]:
    if feedback_mode == "p-sa":
        return brain.attend(q, candidates)
    if feedback_mode == "none":
        return q, brain.index_scores(q, candidates).softmax(dim=-1)
    q_next, _outcome, probabilities = brain.measure(
        q, candidates, selection="argmax"
    )
    return q_next, probabilities


def _sequential_category_logits(
    brain: TensorBrain,
    q: Float[Tensor, "*batch state"],
    candidates_by_level: CategoryCandidates,
    feedback_mode: Literal["p-sa", "p-samp"],
) -> dict[str, Tensor]:
    """Decode ordered hierarchy levels, feeding each prediction into the next."""

    logits = {}
    levels = tuple(candidates_by_level.items())
    for position, (level, candidates) in enumerate(levels):
        logits[level] = brain.index_scores(q, candidates)
        if position + 1 < len(levels):
            q, _probabilities = _index_feedback(
                brain, q, candidates, feedback_mode
            )
    return logits


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


def _record_window(
    trace: Trace | None,
    window: str,
    input_drive: Tensor,
    q_before_input: Tensor,
    q_after_input: Tensor,
    **states: Tensor,
) -> None:
    """Record diagnostic tensors without performing any model operation."""

    if trace is not None:
        trace[window] = {
            "input_drive": input_drive,
            "q_before_input": q_before_input,
            "q_after_input": q_after_input,
            **states,
        }


def _record_identity_window(
    trace: Trace | None,
    window: str,
    brain: TensorBrain,
    identity_candidates: Int[Tensor, " identities"],
    input_drive: Tensor,
    q_before_input: Tensor,
    q_after_input: Tensor,
    q_after_feedback: Tensor,
    probabilities: Tensor,
    *,
    q_after_evolution: Tensor | None = None,
) -> None:
    """Add feedback diagnostics to one already-computed identity window."""

    if trace is None:
        return
    expected, winner = _feedback_vectors(brain, probabilities, identity_candidates)
    states = {
        "identity_probabilities": probabilities,
        "expected_feedback": expected,
        "winner_feedback": winner,
        "applied_feedback": q_after_feedback - q_after_input,
        "q_after_feedback": q_after_feedback,
    }
    if q_after_evolution is not None:
        states["q_after_evolution"] = q_after_evolution
    _record_window(
        trace, window, input_drive, q_before_input, q_after_input, **states
    )


class PDirect(nn.Module):
    """Independently score each label from its own bottom-up feature."""

    def __init__(
        self,
        state_dim: int,
        num_indices: int,
        *,
        score_mode: ScoreMode = "direct",
    ) -> None:
        super().__init__()
        self.brain = TensorBrain(
            state_dim, num_indices, evolution=None, score_mode=score_mode
        )

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
            trace: Trace = {}
            for window, features, q in (
                ("subject", subject_features, subject_q),
                ("object", object_features, object_q),
                ("predicate", union_features, predicate_q),
            ):
                _record_window(
                    trace, window, features, features.new_zeros(features.shape), q
                )
            outputs["trace"] = trace
        return outputs

    def forward_object(
        self,
        object_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        *,
        category_candidates: CategoryCandidates | None = None,
        return_trace: bool = False,
    ) -> ObjectOutputs:
        """Independently decode one entity, matching P-Direct's local evidence."""

        q_before_input = object_features.new_zeros(object_features.shape)
        q = self.brain.integrate_input(q_before_input, object_features)
        outputs: ObjectOutputs = {
            "identity_logits": self.brain.index_scores(q, identity_candidates),
            "category_logits": _category_logits(self.brain, q, category_candidates),
        }
        if return_trace:
            trace: Trace = {}
            _record_window(trace, "object", object_features, q_before_input, q)
            outputs["trace"] = trace
        return outputs


class IntegralTB(nn.Module):
    """Run the paper's four evidence windows with identity feedback."""

    def __init__(
        self,
        state_dim: int,
        num_indices: int,
        evolution: Evolution,
        *,
        score_mode: ScoreMode = "direct",
    ) -> None:
        super().__init__()
        self.brain = TensorBrain(
            state_dim, num_indices, evolution, score_mode=score_mode
        )

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
        if feedback_mode not in ("p-sa", "p-samp", "none"):
            raise ValueError("feedback_mode must be 'p-sa', 'p-samp', or 'none'")

        # Complete-scene window.
        trace: Trace | None = {} if return_trace else None
        q_before_input = scene_features.new_zeros(scene_features.shape)
        q = self.brain.integrate_input(q_before_input, scene_features)
        q_after_input = q
        q, context = self.brain.evolve(q)
        _record_window(
            trace, "scene", scene_features, q_before_input, q_after_input,
            q_after_evolution=q,
        )

        # Subject window: identify, feed the identity back, decode unary categories, then evolve.
        q_before_input = q
        q = self.brain.integrate_input(q, subject_features)
        q_after_input = q
        subject_identity_logits = self.brain.index_scores(q, identity_candidates)
        q, subject_probabilities = _index_feedback(
            self.brain, q, identity_candidates, feedback_mode
        )
        q_after_feedback = q
        subject_category_logits = _category_logits(self.brain, q, category_candidates)
        q, context = self.brain.evolve(q, context)
        _record_identity_window(
            trace, "subject", self.brain, identity_candidates, subject_features,
            q_before_input, q_after_input, q_after_feedback, subject_probabilities,
            q_after_evolution=q,
        )

        # Object window: identify, feed the identity back, decode unary categories, then evolve.
        q_before_input = q
        q = self.brain.integrate_input(q, object_features)
        q_after_input = q
        object_identity_logits = self.brain.index_scores(q, identity_candidates)
        q, object_probabilities = _index_feedback(
            self.brain, q, identity_candidates, feedback_mode
        )
        q_after_feedback = q
        object_category_logits = _category_logits(self.brain, q, category_candidates)
        q, context = self.brain.evolve(q, context)
        _record_identity_window(
            trace, "object", self.brain, identity_candidates, object_features,
            q_before_input, q_after_input, q_after_feedback, object_probabilities,
            q_after_evolution=q,
        )

        # Predicate window.
        q_before_input = q
        q = self.brain.integrate_input(q, union_features)
        predicate_logits = self.brain.index_scores(q, predicate_candidates)
        _record_window(trace, "predicate", union_features, q_before_input, q)

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

    def forward_object(
        self,
        scene_features: Float[Tensor, "*batch state"],
        object_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        *,
        category_candidates: CategoryCandidates | None = None,
        feedback_mode: FeedbackMode = "p-sa",
        sequential_categories: bool = False,
        return_trace: bool = False,
    ) -> ObjectOutputs:
        """Run the paper's scene-to-entity identity and unary-readout schedule."""

        if feedback_mode == "p-samp" and self.training:
            raise ValueError("P-Samp is evaluation-only; train the checkpoint with P-SA")
        if feedback_mode not in ("p-sa", "p-samp", "none"):
            raise ValueError("feedback_mode must be 'p-sa', 'p-samp', or 'none'")
        if sequential_categories and self.training:
            raise ValueError("sequential category feedback is evaluation-only")
        if sequential_categories and feedback_mode == "none":
            raise ValueError("sequential category feedback requires P-SA or P-Samp")
        if sequential_categories and return_trace:
            raise ValueError("sequential category traces are not implemented")

        trace: Trace | None = {} if return_trace else None
        q_before_input = scene_features.new_zeros(scene_features.shape)
        q = self.brain.integrate_input(q_before_input, scene_features)
        q_after_input = q
        q, _context = self.brain.evolve(q)
        _record_window(
            trace, "scene", scene_features, q_before_input, q_after_input,
            q_after_evolution=q,
        )

        q_before_input = q
        q = self.brain.integrate_input(q, object_features)
        q_after_input = q
        identity_logits = self.brain.index_scores(q, identity_candidates)
        q, probabilities = _index_feedback(
            self.brain, q, identity_candidates, feedback_mode
        )
        category_logits = (
            _sequential_category_logits(
                self.brain, q, category_candidates or {}, feedback_mode
            )
            if sequential_categories
            else _category_logits(self.brain, q, category_candidates)
        )
        _record_identity_window(
            trace, "object", self.brain, identity_candidates, object_features,
            q_before_input, q_after_input, q, probabilities,
        )
        outputs: ObjectOutputs = {
            "identity_logits": identity_logits,
            "category_logits": category_logits,
        }
        if trace is not None:
            outputs["trace"] = trace
        return outputs
