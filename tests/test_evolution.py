import torch
from torch.nn import functional as F

from tb import (
    OriginalTBDynamicContext,
    QTBEvolution,
    ReLUEvolution,
    VanillaRNNDynamicContext,
)


def test_qtb_evolution_matches_algorithm_one() -> None:
    evolution = QTBEvolution(2, 3)
    with torch.no_grad():
        evolution.V.copy_(torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.6]]))
        evolution.v0.copy_(torch.tensor([0.1, -0.2, 0.05]))
        evolution.W.copy_(torch.tensor([[0.7, -0.4, 0.2], [-0.1, 0.3, 0.5]]))
    q = torch.tensor([0.25, -0.5])
    h = torch.sigmoid(F.linear(q.sigmoid(), evolution.V, evolution.v0))
    expected_q = F.linear(h, evolution.W)
    actual_q, context = evolution(q)
    torch.testing.assert_close(actual_q, expected_q)
    assert context is None


def test_qtb_evolution_uses_sigmoid_hidden_and_xavier_initialization() -> None:
    evolution = QTBEvolution(4, 5)

    assert evolution.hidden_activation == "sigmoid"
    assert evolution.V.abs().max() <= (6 / (4 + 5)) ** 0.5


def test_relu_evolution_changes_only_hidden_activation_and_uses_kaiming() -> None:
    evolution = ReLUEvolution(2, 3)
    with torch.no_grad():
        evolution.V.copy_(torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.6]]))
        evolution.v0.copy_(torch.tensor([0.1, -0.2, 0.05]))
        evolution.W.copy_(torch.tensor([[0.7, -0.4, 0.2], [-0.1, 0.3, 0.5]]))
    q = torch.tensor([0.25, -0.5])
    expected_h = F.relu(F.linear(q.sigmoid(), evolution.V, evolution.v0))
    expected_q = F.linear(expected_h, evolution.W)

    actual_q, context = evolution(q)

    torch.testing.assert_close(actual_q, expected_q)
    assert context is None
    assert evolution.hidden_activation == "relu"


def test_original_recurrence_matches_algorithm_one() -> None:
    evolution = OriginalTBDynamicContext(2, 3)
    with torch.no_grad():
        evolution.V.copy_(torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.6]]))
        evolution.B.copy_(torch.tensor([[0.5, 0.1, -0.2], [0.0, 0.7, 0.3], [-0.4, 0.2, 0.6]]))
        evolution.W.copy_(torch.tensor([[0.7, -0.4, 0.2], [-0.1, 0.3, 0.5]]))
    q = torch.tensor([0.25, -0.5])
    h = torch.tensor([0.1, -0.2, 0.3])
    expected_h = F.linear(
        torch.sigmoid(torch.sigmoid(h) + F.linear(q.sigmoid(), evolution.V)),
        evolution.B,
    )
    expected_q = F.linear(expected_h.sigmoid(), evolution.W)
    actual_q, actual_h = evolution(q, h)
    torch.testing.assert_close(actual_h, expected_h)
    torch.testing.assert_close(actual_q, expected_q)


def test_original_recurrence_initializes_context_to_zero() -> None:
    evolution = OriginalTBDynamicContext(4, 5)
    q = torch.randn(3, 4)
    q_next, context = evolution(q)
    assert q_next.shape == (3, 4)
    assert context.shape == (3, 5)


def test_vanilla_rnn_control_preserves_batch_shape() -> None:
    evolution = VanillaRNNDynamicContext(4, 5)
    q_next, context = evolution(torch.randn(3, 4))
    assert q_next.shape == (3, 4)
    assert context.shape == (3, 5)
