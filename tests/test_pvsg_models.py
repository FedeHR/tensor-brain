import pytest
import torch
from jaxtyping import Float
from torch import Tensor

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
        model.brain.a0.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3, -0.1]))


def test_p_direct_scores_each_feature_independently() -> None:
    model = PDirect(state_dim=2, num_indices=5)
    set_index_parameters(model)
    subject = torch.tensor([[0.2, -0.4], [0.7, 0.1]])
    object_ = torch.tensor([[-0.3, 0.8], [0.5, -0.6]])
    union = torch.tensor([[0.1, 0.9], [-0.8, 0.4]])
    identities = torch.tensor([0, 1])
    predicates = torch.tensor([2, 3, 4])

    scores = model(subject, object_, union, identities, predicates)

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


def test_integral_p_sa_matches_the_explicit_four_window_schedule() -> None:
    model = IntegralTB(state_dim=2, num_indices=5, evolution=CountingEvolution())
    set_index_parameters(model)
    scene = torch.tensor([[0.2, -0.1]])
    subject = torch.tensor([[0.4, 0.3]])
    object_ = torch.tensor([[-0.2, 0.5]])
    union = torch.tensor([[0.1, -0.4]])
    identities = torch.tensor([0, 1])
    predicates = torch.tensor([2, 3, 4])

    scores = model(scene, subject, object_, union, identities, predicates)

    subject_q = 2.0 * scene + 1.0 + subject
    expected_subject = model.brain.index_scores(subject_q, identities)
    subject_q, _ = model.brain.attend(subject_q, identities)
    object_q = 2.0 * subject_q + 2.0 + object_
    expected_object = model.brain.index_scores(object_q, identities)
    object_q, _ = model.brain.attend(object_q, identities)
    predicate_q = 2.0 * object_q + 3.0 + union
    expected_predicate = model.brain.index_scores(predicate_q, predicates)

    torch.testing.assert_close(scores["subject_identity_logits"], expected_subject)
    torch.testing.assert_close(scores["object_identity_logits"], expected_object)
    torch.testing.assert_close(scores["predicate_logits"], expected_predicate)


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
