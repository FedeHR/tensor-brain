"""Evolution operators between Tensor Brain concept windows."""

from abc import ABC, abstractmethod
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


def _reset_matrix(
    parameter: Float[Tensor, "rows columns"],
    *,
    activation: Literal["sigmoid", "relu", "linear"],
) -> None:
    """Initialize a matrix for the activation used immediately downstream."""

    if activation in ("sigmoid", "linear"):
        gain = 0.5 if activation == "sigmoid" else 1.0
        nn.init.xavier_uniform_(parameter, gain=gain)
    else:
        nn.init.kaiming_uniform_(parameter, nonlinearity="relu")


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


class _FeedForwardEvolution(Evolution):
    """Shared implementation for explicitly named feed-forward variants."""

    hidden_activation: Literal["sigmoid", "relu"]

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        *,
        hidden_activation: Literal["sigmoid", "relu"],
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.V = nn.Parameter(torch.empty(hidden_dim, state_dim))
        self.v0 = nn.Parameter(torch.zeros(hidden_dim))
        self.W = nn.Parameter(torch.empty(state_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_matrix(self.V, activation=self.hidden_activation)
        _reset_matrix(self.W, activation="linear")
        nn.init.zeros_(self.v0)

    def _activate_hidden(self, values: Tensor) -> Tensor:
        if self.hidden_activation == "sigmoid":
            return torch.sigmoid(values)
        return F.relu(values)

    def forward(
        self,
        q: Float[Tensor, "*batch state"],
        context: Float[Tensor, "*batch context"] | None = None,
    ) -> tuple[Float[Tensor, "*batch state"], None]:
        if context is not None:
            raise ValueError("feed-forward evolution has no persistent context")
        gamma = torch.sigmoid(q)
        h = self._activate_hidden(F.linear(gamma, self.V, self.v0))
        q_next = F.linear(h, self.W)
        return q_next, None


class QTBEvolution(_FeedForwardEvolution):
    r"""One-hidden-layer evolution from QTB Algorithm 1.

    .. math::
        h = \sigma(v_0 + V\sigma(q)), \qquad q' = Wh.

    The hidden activation ``h`` is not recurrent state in this formulation.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__(state_dim, hidden_dim, hidden_activation="sigmoid")


class ReLUEvolution(_FeedForwardEvolution):
    r"""ReLU hidden evolution with the same CBS and unrestricted ``q`` boundary.

    This is an experimental extension, not the paper equation.  The input remains
    ``gamma = sigmoid(q)`` and the output remains a linear pre-CBS ``q_next`` so
    that only the hidden evolution nonlinearity changes.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__(state_dim, hidden_dim, hidden_activation="relu")


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
        _reset_matrix(self.V, activation="sigmoid")
        _reset_matrix(self.B, activation="sigmoid")
        _reset_matrix(self.W, activation="linear")

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
