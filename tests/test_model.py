import math

import pytest
import torch
from jaxtyping import TypeCheckError

from tb import QTBEvolution, TensorBrain


def make_brain() -> TensorBrain:
    brain = TensorBrain(2, 3, QTBEvolution(2, 2))
    with torch.no_grad():
        brain.A.copy_(torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]]))
        brain.a0.copy_(torch.tensor([0.1, -0.2, 0.3]))
    return brain


def test_input_integration_matches_the_paper_equation_for_a_batch() -> None:
    brain = make_brain()
    q = torch.tensor([[0.4, -0.7], [0.1, 0.3]])
    input_drive = torch.tensor([[1.2, -0.5], [-0.4, 0.8]])

    q_next = brain.integrate_input(q, input_drive, input_gate=0.25)

    torch.testing.assert_close(q_next, q + 0.25 * input_drive)


def test_input_integration_preserves_gradients_through_input_and_gate() -> None:
    brain = make_brain()
    q = torch.tensor([0.4, -0.7], requires_grad=True)
    input_drive = torch.tensor([1.2, -0.5], requires_grad=True)
    input_gate = torch.tensor(0.25, requires_grad=True)

    brain.integrate_input(q, input_drive, input_gate=input_gate).sum().backward()

    assert q.grad is not None
    assert input_drive.grad is not None
    assert input_gate.grad is not None


def test_input_integration_requires_input_in_state_coordinates() -> None:
    brain = make_brain()

    with pytest.raises(TypeCheckError):
        brain.integrate_input(torch.zeros(2), torch.zeros(3))


def test_index_scores_match_the_paper_equation() -> None:
    brain = make_brain()
    q = torch.tensor([0.4, -0.7])
    candidates = torch.tensor([0, 2])
    expected = brain.a0[candidates] + brain.A[:, candidates].T @ q.sigmoid()
    torch.testing.assert_close(brain.index_scores(q, candidates), expected)


def test_omitted_candidates_score_the_complete_global_index_set() -> None:
    brain = make_brain()
    q = torch.tensor([0.4, -0.7])
    expected = brain.a0 + brain.A.T @ q.sigmoid()

    torch.testing.assert_close(brain.index_scores(q), expected)


def test_attention_is_expected_index_feedback() -> None:
    brain = make_brain()
    q = torch.tensor([0.4, -0.7])
    candidates = torch.tensor([0, 2])
    q_next, probabilities = brain.attend(q, candidates)
    expected = q + brain.A[:, candidates] @ probabilities
    torch.testing.assert_close(q_next, expected)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))


def test_teacher_forced_measurement_uses_global_index_and_gates() -> None:
    brain = make_brain()
    q = torch.tensor([0.4, -0.7])
    q_next, outcome, probabilities = brain.measure(
        q,
        [0, 2],
        outcome=2,
        selection="teacher",
        retain_gate=0.25,
        feedback_gate=0.75,
    )
    torch.testing.assert_close(q_next, 0.25 * q + 0.75 * brain.A[:, 2])
    assert outcome.item() == 2
    assert probabilities.shape == (2,)


def test_sampled_measurement_returns_only_candidates_for_a_batch() -> None:
    torch.manual_seed(4)
    brain = make_brain()
    q = torch.zeros(16, 2)
    _, outcomes, probabilities = brain.measure(q, [0, 2])
    assert set(outcomes.tolist()) <= {0, 2}
    assert probabilities.shape == (16, 2)


def test_argmax_measurement_uses_global_winner_take_all_index() -> None:
    brain = make_brain()
    q = torch.tensor([[0.4, -0.7], [-1.0, 1.2]])
    candidates = torch.tensor([0, 2])
    expected = candidates[brain.index_scores(q, candidates).argmax(dim=-1)]
    _, outcomes, _ = brain.measure(q, candidates, selection="argmax")
    torch.testing.assert_close(outcomes, expected)


@pytest.mark.parametrize(
    ("retain_gate", "feedback_gate"),
    [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
)
def test_canonical_measurement_gate_regimes(retain_gate: float, feedback_gate: float) -> None:
    brain = make_brain()
    q = torch.tensor([0.4, -0.7])
    q_next, _, _ = brain.measure(
        q,
        [0, 2],
        outcome=2,
        selection="teacher",
        retain_gate=retain_gate,
        feedback_gate=feedback_gate,
    )
    expected = retain_gate * q + feedback_gate * brain.A[:, 2]
    torch.testing.assert_close(q_next, expected)


def test_measurement_gates_can_be_learned_tensors() -> None:
    brain = make_brain()
    q = torch.tensor([0.4, -0.7])
    retain_gate = torch.tensor(0.6, requires_grad=True)
    feedback_gate = torch.tensor(1.2, requires_grad=True)

    q_next, _, _ = brain.measure(
        q,
        [0, 2],
        outcome=2,
        selection="teacher",
        retain_gate=retain_gate,
        feedback_gate=feedback_gate,
    )
    q_next.sum().backward()

    assert retain_gate.grad is not None
    assert feedback_gate.grad is not None


def test_index_embeddings_start_with_unit_expected_column_norm() -> None:
    torch.manual_seed(8)
    state_dim = 256
    brain = TensorBrain(state_dim, 1024, QTBEvolution(state_dim, 32))
    expected_std = 1 / math.sqrt(state_dim)
    embeddings = brain.A.detach()
    assert abs(float(embeddings.std()) - expected_std) < 0.001
    assert abs(float(embeddings.square().sum(dim=0).mean()) - 1.0) < 0.02


def test_shared_embedding_receives_gradients_from_scoring_and_feedback() -> None:
    brain = make_brain()
    q = torch.tensor([0.4, -0.7], requires_grad=True)
    q_next, probabilities = brain.attend(q, [0, 2])
    (q_next.sum() + probabilities[0]).backward()
    assert brain.A.grad is not None
    assert brain.A.grad[:, [0, 2]].abs().sum() > 0


def test_invalid_measurement_outcome_is_rejected() -> None:
    brain = make_brain()
    with pytest.raises(ValueError):
        brain.measure(torch.zeros(2), [0, 2], outcome=1, selection="teacher")


def test_runtime_types_reject_float_index_tensors() -> None:
    brain = make_brain()
    with pytest.raises(TypeCheckError):
        brain.index_scores(torch.zeros(2), torch.tensor([0.0, 2.0]))


def test_runtime_types_match_outcome_batch_to_q_batch() -> None:
    brain = make_brain()
    with pytest.raises(TypeCheckError):
        brain.measure(
            torch.zeros(2, 2),
            [0, 2],
            outcome=torch.tensor([0, 2, 0]),
            selection="teacher",
        )


def test_scalar_teacher_outcome_broadcasts_across_batch() -> None:
    brain = make_brain()
    q = torch.zeros(3, 2)
    _, outcomes, _ = brain.measure(q, [0, 2], outcome=2, selection="teacher")
    torch.testing.assert_close(outcomes, torch.full((3,), 2))
