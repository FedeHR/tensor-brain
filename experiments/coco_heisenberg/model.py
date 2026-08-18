"""Fit the index layer on COCO, and hand it to the inference ladder.

The likelihood is the paper's own emission model,

    P(k | x) = softmax(a0 + A^T x)_k ,      x in {0,1}^n  presence bits
                                            k in 1..K     caption words

so fitting ``(A, a0)`` by maximum likelihood is multinomial logistic regression
of the named word on the presence vector. Nothing about the update rule enters
training: the index layer is fitted to the data, and every inference rule is then
scored against the *same* learned model. That keeps the comparison about the
update rule rather than about model quality.

The prior is fitted separately, in two forms, because the scoping work showed
that prior misspecification and measurement-process misspecification are
confounded and must be separated:

    factorized  independent Bernoulli marginals -- what the theory assumes
    empirical   the full joint over all 2^n states -- the ceiling
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import Corpus

DTYPE = torch.float64


@dataclass(frozen=True)
class IndexLayer:
    """A fitted index layer plus the priors it was fitted alongside."""

    A: torch.Tensor                    # [categories, symbols]
    a0: torch.Tensor                   # [symbols]
    q_prior: torch.Tensor              # [categories] factorized prior logits
    joint_prior: torch.Tensor          # [2^categories] empirical joint prior
    history: list[dict[str, float]]

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def num_symbols(self) -> int:
        return int(self.A.shape[1])


def _design(corpus: Corpus) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the corpus into (presence, symbol) training pairs."""

    lengths = np.asarray([len(s) for s in corpus.symbols])
    rows = np.repeat(np.arange(corpus.num_images), lengths)
    symbols = np.concatenate([np.asarray(s, dtype=np.int64) for s in corpus.symbols])
    presence = torch.as_tensor(corpus.presence[rows], dtype=DTYPE)
    return presence, torch.as_tensor(symbols, dtype=torch.long)


def sufficient_statistics(corpus: Corpus) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse the corpus to the counts the likelihood actually depends on.

    ``x`` takes only ``2^n`` distinct values, so the log likelihood

        sum_pairs [ a0_k + a_k^T x - logZ(x) ]

    is a function of the ``[2^n, K]`` table of how often symbol ``k`` was named
    for presence pattern ``x``. Fitting against that table is exact and roughly
    two orders of magnitude cheaper than iterating over every pair.
    """

    n = corpus.num_categories
    weights = (2 ** np.arange(n - 1, -1, -1)).astype(np.int64)
    codes = corpus.presence.astype(np.int64) @ weights

    lengths = np.asarray([len(s) for s in corpus.symbols])
    rows = np.repeat(codes, lengths)
    symbols = np.concatenate([np.asarray(s, dtype=np.int64) for s in corpus.symbols])

    flat = np.bincount(rows * corpus.num_symbols + symbols,
                       minlength=(2**n) * corpus.num_symbols)
    counts = torch.as_tensor(flat.reshape(2**n, corpus.num_symbols), dtype=DTYPE)

    codes_all = torch.arange(2**n).unsqueeze(1)
    bit = 2 ** torch.arange(n - 1, -1, -1, dtype=torch.long)
    states = (codes_all & bit).ne(0).to(DTYPE)
    return states, counts


def fit_factorized_prior(corpus: Corpus) -> torch.Tensor:
    rate = torch.as_tensor(corpus.presence.mean(axis=0), dtype=DTYPE).clamp(1e-6, 1 - 1e-6)
    return torch.log(rate) - torch.log1p(-rate)


def fit_joint_prior(corpus: Corpus, *, smoothing: float = 0.5) -> torch.Tensor:
    """Empirical distribution over all ``2^n`` presence patterns, Laplace-smoothed."""

    n = corpus.num_categories
    weights = (2 ** np.arange(n - 1, -1, -1)).astype(np.int64)
    codes = corpus.presence.astype(np.int64) @ weights
    counts = np.bincount(codes, minlength=2**n).astype(np.float64) + smoothing
    return torch.as_tensor(counts / counts.sum(), dtype=DTYPE)


def fit_index_layer(
    corpus: Corpus,
    *,
    steps: int = 400,
    learning_rate: float = 0.5,
    weight_decay: float = 1e-4,
    seed: int = 0,
    log_every: int = 50,
) -> IndexLayer:
    """Maximum-likelihood fit of ``(A, a0)``, i.e. softmax regression of k on x."""

    torch.manual_seed(seed)
    states, counts = sufficient_statistics(corpus)
    n, K = corpus.num_categories, corpus.num_symbols
    total = counts.sum()

    A = torch.zeros(n, K, dtype=DTYPE, requires_grad=True)
    a0 = torch.zeros(K, dtype=DTYPE, requires_grad=True)
    optimizer = torch.optim.Adam([A, a0], lr=learning_rate)

    history: list[dict[str, float]] = []
    for step in range(steps):
        scores = a0.unsqueeze(0) + states @ A
        log_partition = torch.logsumexp(scores, dim=-1)
        # exact mean cross-entropy over every (x, k) pair in the corpus
        loss = (counts.sum(-1) * log_partition - (counts * scores).sum(-1)).sum() / total
        penalty = weight_decay * A.pow(2).sum()

        optimizer.zero_grad()
        (loss + penalty).backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "nll": float(loss.detach())})

    A, a0 = A.detach(), a0.detach()
    return IndexLayer(
        A=A,
        a0=a0,
        q_prior=fit_factorized_prior(corpus),
        joint_prior=fit_joint_prior(corpus),
        history=history,
    )


def held_out_nll(layer: IndexLayer, corpus: Corpus) -> float:
    """Mean negative log likelihood of the named symbols under the fitted layer."""

    presence, symbols = _design(corpus)
    scores = layer.a0.unsqueeze(0) + presence @ layer.A
    return float(torch.nn.functional.cross_entropy(scores, symbols))


def uniform_symbol_nll(corpus: Corpus) -> float:
    """The do-nothing reference for :func:`held_out_nll`."""

    return float(np.log(corpus.num_symbols))
