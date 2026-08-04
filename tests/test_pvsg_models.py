import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from experiments.pvsg.baselines import FlatFusion, FusedLinear, LinearProbe
from experiments.pvsg.diagnostics import object_scale_trace_rows
from experiments.pvsg.models import IntegralTB, PDirect
from tb.evolution import Evolution


class CountingEvolution(Evolution):
    def forward(
        self,
        q: Float[Tensor, "*batch state"],
        context: Float[Tensor, "*batch context"] | None = None,
    ) -> tuple[
        Float[Tensor, "*batch state"],
        Float[Tensor, "*batch context"],
    ]:
        if context is None:
            context = torch.zeros_like(q)
        context_next = context + 1.0
        return 2.0 * q + context_next, context_next


def set_index_parameters(model: PDirect | IntegralTB) -> None:
    with torch.no_grad():
        model.brain.A.copy_(
            torch.tensor(
                [
                    [1.0, -0.7, 0.3, -0.2, 0.8],
                    [-0.4, 0.9, 0.6, -1.0, 0.1],
                ]
            )
        )
        if model.brain.a0 is not None:
            model.brain.a0.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3, -0.1]))


def test_p_direct_scores_each_feature_independently() -> None:
    model = PDirect(state_dim=2, num_indices=5)
    set_index_parameters(model)
    subject = torch.tensor([[0.2, -0.4], [0.7, 0.1]])
    object_ = torch.tensor([[-0.3, 0.8], [0.5, -0.6]])
    union = torch.tensor([[0.1, 0.9], [-0.8, 0.4]])
    identities = torch.tensor([0, 1])
    predicates = torch.tensor([2, 3, 4])
    categories = {
        "object_category/source": torch.tensor([2, 4]),
        "object_category/coarse": torch.tensor([3, 4]),
    }

    scores = model(
        subject,
        object_,
        union,
        identities,
        predicates,
        category_candidates=categories,
        return_trace=True,
    )

    assert model.brain.evolution is None
    torch.testing.assert_close(
        scores["subject_identity_logits"], model.brain.index_scores(subject, identities)
    )
    torch.testing.assert_close(
        scores["object_identity_logits"], model.brain.index_scores(object_, identities)
    )
    torch.testing.assert_close(
        scores["predicate_logits"], model.brain.index_scores(union, predicates)
    )
    for level, candidates in categories.items():
        torch.testing.assert_close(
            scores["subject_category_logits"][level],
            model.brain.index_scores(subject, candidates),
        )
        torch.testing.assert_close(
            scores["object_category_logits"][level],
            model.brain.index_scores(object_, candidates),
        )
    torch.testing.assert_close(scores["trace"]["subject"]["q_after_input"], subject)


def test_integral_p_sa_matches_the_explicit_four_window_schedule() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())
    set_index_parameters(model)
    scene = torch.tensor([[0.2, -0.1]])
    subject = torch.tensor([[0.4, 0.3]])
    object_ = torch.tensor([[-0.2, 0.5]])
    union = torch.tensor([[0.1, -0.4]])
    identities = torch.tensor([0, 1])
    predicates = torch.tensor([2, 3, 4])
    categories = {
        "object_category/source": torch.tensor([2, 4]),
        "object_category/coarse": torch.tensor([3, 4]),
    }

    scores = model(
        scene,
        subject,
        object_,
        union,
        identities,
        predicates,
        category_candidates=categories,
        return_trace=True,
    )

    subject_q = 2.0 * scene + 1.0 + subject
    expected_subject = model.brain.index_scores(subject_q, identities)
    traced_subject_input = subject_q
    subject_q, _ = model.brain.attend(subject_q, identities)
    expected_subject_categories = {
        level: model.brain.index_scores(subject_q, candidates)
        for level, candidates in categories.items()
    }
    object_q = 2.0 * subject_q + 2.0 + object_
    expected_object = model.brain.index_scores(object_q, identities)
    traced_object_input = object_q
    object_q, _ = model.brain.attend(object_q, identities)
    expected_object_categories = {
        level: model.brain.index_scores(object_q, candidates)
        for level, candidates in categories.items()
    }
    predicate_q = 2.0 * object_q + 3.0 + union
    expected_predicate = model.brain.index_scores(predicate_q, predicates)

    torch.testing.assert_close(scores["subject_identity_logits"], expected_subject)
    torch.testing.assert_close(scores["object_identity_logits"], expected_object)
    torch.testing.assert_close(scores["predicate_logits"], expected_predicate)
    torch.testing.assert_close(
        scores["trace"]["subject"]["q_after_input"], traced_subject_input
    )
    torch.testing.assert_close(
        scores["trace"]["object"]["q_after_input"], traced_object_input
    )
    torch.testing.assert_close(
        scores["trace"]["subject"]["applied_feedback"], subject_q - traced_subject_input
    )
    for level in categories:
        torch.testing.assert_close(
            scores["subject_category_logits"][level],
            expected_subject_categories[level],
        )
        torch.testing.assert_close(
            scores["object_category_logits"][level],
            expected_object_categories[level],
        )


def test_p_samp_is_an_evaluation_mode_of_the_integral_checkpoint() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())
    set_index_parameters(model)
    inputs = (
        torch.tensor([[0.2, -0.1]]),
        torch.tensor([[0.4, 0.3]]),
        torch.tensor([[-0.2, 0.5]]),
        torch.tensor([[0.1, -0.4]]),
        torch.tensor([0, 1]),
        torch.tensor([2, 3, 4]),
    )

    with pytest.raises(ValueError, match="evaluation-only"):
        model(*inputs, feedback_mode="p-samp")

    parameter_ids = {id(parameter) for parameter in model.parameters()}
    model.eval()
    p_sa_scores = model(*inputs, feedback_mode="p-sa")
    p_samp_scores = model(*inputs, feedback_mode="p-samp")

    assert {id(parameter) for parameter in model.parameters()} == parameter_ids
    assert p_sa_scores["subject_category_logits"] == {}
    assert p_samp_scores["subject_category_logits"] == {}
    assert not torch.equal(p_sa_scores["predicate_logits"], p_samp_scores["predicate_logits"])


def test_integral_predicate_loss_reaches_identity_embeddings_through_feedback() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())
    set_index_parameters(model)
    identities = torch.tensor([0, 1])
    predicates = torch.tensor([2, 3, 4])

    scores = model(
        torch.tensor([[0.2, -0.1]]),
        torch.tensor([[0.4, 0.3]]),
        torch.tensor([[-0.2, 0.5]]),
        torch.tensor([[0.1, -0.4]]),
        identities,
        predicates,
    )
    scores["predicate_logits"].sum().backward()

    assert model.brain.A.grad is not None
    assert model.brain.A.grad[:, identities].abs().sum() > 0


def test_integral_category_loss_reaches_identity_embeddings_through_feedback() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())
    set_index_parameters(model)
    identities = torch.tensor([0, 1])

    scores = model(
        torch.tensor([[0.2, -0.1]]),
        torch.tensor([[0.4, 0.3]]),
        torch.tensor([[-0.2, 0.5]]),
        torch.tensor([[0.1, -0.4]]),
        identities,
        torch.tensor([2, 3, 4]),
        category_candidates={"object_category/source": torch.tensor([2, 3, 4])},
    )
    scores["subject_category_logits"]["object_category/source"].sum().backward()

    assert model.brain.A.grad is not None
    assert model.brain.A.grad[:, identities].abs().sum() > 0


def test_object_schedule_is_the_scene_and_single_entity_prefix() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())
    set_index_parameters(model)
    scene = torch.tensor([[0.2, -0.1]])
    object_ = torch.tensor([[0.4, 0.3]])
    identities = torch.tensor([0, 1])
    categories = {"object_category/source": torch.tensor([2, 3, 4])}

    outputs = model.forward_object(
        scene,
        object_,
        identities,
        category_candidates=categories,
        return_trace=True,
    )
    q_before_feedback = 2.0 * scene + 1.0 + object_
    q_after_feedback, _ = model.brain.attend(q_before_feedback, identities)

    torch.testing.assert_close(
        outputs["identity_logits"],
        model.brain.index_scores(q_before_feedback, identities),
    )
    torch.testing.assert_close(
        outputs["category_logits"]["object_category/source"],
        model.brain.index_scores(q_after_feedback, categories["object_category/source"]),
    )
    assert set(outputs["trace"]) == {"scene", "object"}
    rows = object_scale_trace_rows(
        model,
        outputs,
        {"identity": identities, **categories},
        {"scene_raw_l2": torch.tensor([2.0]), "object_raw_l2": torch.tensor([1.0])},
    )
    assert {"raw_input_norm", "operation_delta", "readout"} <= {
        row["kind"] for row in rows
    }


def test_no_feedback_control_keeps_the_post_input_state() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())
    set_index_parameters(model)
    outputs = model.forward_object(
        torch.tensor([[0.2, -0.1]]),
        torch.tensor([[0.4, 0.3]]),
        torch.tensor([0, 1]),
        feedback_mode="none",
        return_trace=True,
    )

    torch.testing.assert_close(
        outputs["trace"]["object"]["q_after_feedback"],
        outputs["trace"]["object"]["q_after_input"],
    )


def test_sequential_hierarchy_feedback_conditions_later_levels() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())
    set_index_parameters(model)
    scene = torch.tensor([[0.2, -0.1]])
    object_ = torch.tensor([[0.4, 0.3]])
    identities = torch.tensor([0, 1])
    categories = {
        "object_category/fine": torch.tensor([2, 3]),
        "object_category/coarse": torch.tensor([3, 4]),
    }

    model.eval()
    outputs = model.forward_object(
        scene,
        object_,
        identities,
        category_candidates=categories,
        sequential_categories=True,
    )
    q = 2.0 * scene + 1.0 + object_
    q, _ = model.brain.attend(q, identities)
    expected_fine = model.brain.index_scores(q, categories["object_category/fine"])
    q, _ = model.brain.attend(q, categories["object_category/fine"])

    torch.testing.assert_close(
        outputs["category_logits"]["object_category/fine"], expected_fine
    )
    torch.testing.assert_close(
        outputs["category_logits"]["object_category/coarse"],
        model.brain.index_scores(q, categories["object_category/coarse"]),
    )


def test_sequential_hierarchy_feedback_is_evaluation_only() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())

    with pytest.raises(ValueError, match="evaluation-only"):
        model.forward_object(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.tensor([0, 1]),
            category_candidates={"object_category/source": torch.tensor([2, 3])},
            sequential_categories=True,
        )


@pytest.mark.parametrize("model_type", (LinearProbe, FlatFusion))
def test_non_tb_baselines_return_pair_readouts(model_type) -> None:
    model = (
        model_type(state_dim=2, num_indices=5, hidden_dim=3)
        if model_type is FlatFusion
        else model_type(state_dim=2, num_indices=5)
    )
    identities = torch.tensor([0, 1])
    predicates = torch.tensor([2, 3, 4])
    categories = {"object_category/source": torch.tensor([2, 4])}
    scene = torch.tensor([[0.2, -0.1]])
    subject = torch.tensor([[0.4, 0.3]])
    object_ = torch.tensor([[-0.2, 0.5]])
    union = torch.tensor([[0.1, -0.4]])

    if isinstance(model, FlatFusion):
        pair = model(
            scene,
            subject,
            object_,
            union,
            identities,
            predicates,
            category_candidates=categories,
        )
    else:
        pair = model(
            subject,
            object_,
            union,
            identities,
            predicates,
            category_candidates=categories,
        )

    assert pair["subject_identity_logits"].shape == (1, 2)
    assert pair["predicate_logits"].shape == (1, 3)


def test_linear_probe_returns_object_readouts() -> None:
    model = LinearProbe(state_dim=2, num_indices=5)
    single = model.forward_object(
        torch.tensor([[-0.2, 0.5]]),
        torch.tensor([0, 1]),
        category_candidates={"object_category/source": torch.tensor([2, 4])},
    )

    assert single["identity_logits"].shape == (1, 2)
    assert single["category_logits"]["object_category/source"].shape == (1, 2)


def test_fused_linear_object_readout_uses_scene_and_object_evidence() -> None:
    model = FusedLinear(state_dim=2, num_indices=5, num_sources=2)
    outputs = model.forward_object(
        torch.tensor([[0.2, -0.1]]),
        torch.tensor([[-0.2, 0.5]]),
        torch.tensor([0, 1]),
        category_candidates={"object_category/source": torch.tensor([2, 4])},
    )

    assert model.readout.in_features == 4
    assert outputs["identity_logits"].shape == (1, 2)
    assert outputs["category_logits"]["object_category/source"].shape == (1, 2)
