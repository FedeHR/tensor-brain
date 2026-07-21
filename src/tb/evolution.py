"""Evolution operators between Tensor Brain concept windows."""

import math
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


def _reset_matrix(parameter: Float[Tensor, "rows columns"]) -> None:
    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))


class Evolution(nn.Module, ABC):
    """Interface for a transition between two concept windows.

    ``context`` is backend-specific. The QTB feed-forward transition has no
    persistent context, while recurrent implementations return a hidden state.
    """

    @abstractmethod
    def forward(
        self,
        q: Float[Tensor, "*batch state"],
        context: Float[Tensor, "*batch context"] | None = None,
    ) -> tuple[
        Float[Tensor, "*batch state"],
        Float[Tensor, "*batch context"] | None,
    ]:
        """Return the next pre-CBS and optional persistent context."""


class QTBEvolution(Evolution):
    r"""One-hidden-layer evolution from QTB Algorithm 1.

    .. math::
        h = \sigma(v_0 + V\sigma(q)), \qquad q' = Wh.

    The hidden activation ``h`` is not recurrent state in this formulation.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.V = nn.Parameter(torch.empty(hidden_dim, state_dim))
        self.v0 = nn.Parameter(torch.zeros(hidden_dim))
        self.W = nn.Parameter(torch.empty(state_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_matrix(self.V)
        _reset_matrix(self.W)
        nn.init.zeros_(self.v0)

    def forward(
        self,
        q: Float[Tensor, "*batch state"],
        context: Float[Tensor, "*batch context"] | None = None,
    ) -> tuple[Float[Tensor, "*batch state"], None]:
        if context is not None:
            raise ValueError("QTBEvolution has no persistent context")
        gamma = torch.sigmoid(q)
        h = torch.sigmoid(F.linear(gamma, self.V, self.v0))
        q_next = F.linear(h, self.W)
        return q_next, None


class OriginalTBDynamicContext(Evolution):
    r"""Recurrent dynamic-context equation from the original TB Algorithm 1.

    .. math::
        h' = B\sigma(\sigma(h) + V\sigma(q)), \qquad
        q' = W\sigma(h').

    Here ``h`` and ``h'`` are dynamic-context preactivations. A missing context
    is initialized to zero, matching the paper algorithm.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.V = nn.Parameter(torch.empty(hidden_dim, state_dim))
        self.B = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W = nn.Parameter(torch.empty(state_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_matrix(self.V)
        _reset_matrix(self.B)
        _reset_matrix(self.W)

    def forward(
        self,
        q: Float[Tensor, "*batch state"],
        context: Float[Tensor, "*batch context"] | None = None,
    ) -> tuple[
        Float[Tensor, "*batch state"],
        Float[Tensor, "*batch context"],
    ]:
        gamma = torch.sigmoid(q)
        if context is None:
            context = q.new_zeros(*q.shape[:-1], self.hidden_dim)
        if context.shape != (*q.shape[:-1], self.hidden_dim):
            raise ValueError(
                "context must have shape "
                f"{(*q.shape[:-1], self.hidden_dim)}, got {tuple(context.shape)}"
            )
        h_next = F.linear(
            torch.sigmoid(torch.sigmoid(context) + F.linear(gamma, self.V)),
            self.B,
        )
        q_next = F.linear(torch.sigmoid(h_next), self.W)
        return q_next, h_next


class VanillaRNNDynamicContext(Evolution):
    """Conventional tanh RNN control, not the original TB recurrence."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNNCell(state_dim, hidden_dim, nonlinearity="tanh")
        self.W = nn.Linear(hidden_dim, state_dim, bias=False)

    def forward(
        self,
        q: Float[Tensor, "*batch state"],
        context: Float[Tensor, "*batch context"] | None = None,
    ) -> tuple[
        Float[Tensor, "*batch state"],
        Float[Tensor, "*batch context"],
    ]:
        if context is None:
            context = q.new_zeros(*q.shape[:-1], self.hidden_dim)
        gamma = torch.sigmoid(q)
        h_next = self.rnn(gamma, context)
        return self.W(h_next), h_next
