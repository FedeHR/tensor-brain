"""Closed-form linear probes, shared by the policy and filter studies.

Ridge regression in closed form rather than a fitted head, because a probe
trained by gradient descent reports a property of the optimizer as much as of
the representation. The closed form has one hyperparameter, no seed and no
stopping rule, so a difference between two architectures is a difference between
the architectures.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor


def ridge_fit(
    inputs: Float[Tensor, "samples width"],
    outputs: Float[Tensor, "samples dims"],
    penalty: float = 1.0,
) -> Float[Tensor, "width1 dims"]:
    """Closed-form ridge regression with an unpenalised bias column."""

    design = torch.cat([inputs, torch.ones(inputs.shape[0], 1)], dim=1).double()
    gram = design.T @ design
    regularizer = penalty * torch.eye(gram.shape[0], dtype=gram.dtype)
    regularizer[-1, -1] = 0.0  # never penalise the bias
    return torch.linalg.solve(gram + regularizer, design.T @ outputs.double())


def ridge_apply(
    weights: Float[Tensor, "width1 dims"], inputs: Float[Tensor, "samples width"]
) -> Float[Tensor, "samples dims"]:
    design = torch.cat([inputs, torch.ones(inputs.shape[0], 1)], dim=1).double()
    return design @ weights


def r2_score(
    prediction: Float[Tensor, "samples dims"], target: Float[Tensor, "samples dims"]
) -> float:
    r""":math:`R^2` against the *test* mean.

    Zero is the score of predicting the mean; a negative score is worse than
    that. Computed against the test mean rather than the train mean so a shifted
    held-out distribution cannot flatter the probe.
    """

    target = target.double()
    residual = ((prediction - target) ** 2).sum()
    total = ((target - target.mean(dim=0)) ** 2).sum()
    return float(1.0 - residual / total.clamp_min(1e-12))


def regression_probe(
    train_state: Float[Tensor, "train width"],
    train_target: Float[Tensor, "train dims"],
    test_state: Float[Tensor, "test width"],
    test_target: Float[Tensor, "test dims"],
    *,
    penalty: float = 1.0,
) -> dict[str, float]:
    """Fit state -> target on one split, score on another."""

    weights = ridge_fit(train_state, train_target, penalty)
    prediction = ridge_apply(weights, test_state)
    return {
        "r2": r2_score(prediction, test_target),
        "rmse": float(((prediction - test_target.double()) ** 2).mean().sqrt()),
        "dims": int(test_target.shape[1]),
    }


def classification_probe(
    train_state: Float[Tensor, "train width"],
    train_label: Int[Tensor, " train"],
    test_state: Float[Tensor, "test width"],
    test_label: Int[Tensor, " test"],
    *,
    classes: int,
    penalty: float = 1.0,
) -> dict[str, float]:
    """Least-squares one-hot probe, scored by argmax accuracy.

    The simplest deterministic linear classifier, chosen for the same reason as
    the closed-form regression: it introduces no optimizer of its own.
    """

    onehot = torch.zeros(train_state.shape[0], classes)
    onehot.scatter_(1, train_label[:, None], 1.0)
    weights = ridge_fit(train_state, onehot, penalty)
    predicted = ridge_apply(weights, test_state).argmax(dim=1)
    majority = test_label.bincount(minlength=classes).max() / test_label.shape[0]
    return {
        "accuracy": float((predicted == test_label).double().mean()),
        "majority_baseline": float(majority),
        "chance": 1.0 / classes,
    }


def mutual_information(
    symbols: Int[Tensor, " samples"],
    labels: Int[Tensor, " samples"],
    *,
    num_symbols: int,
    num_labels: int,
) -> dict[str, float]:
    """Information a discrete symbol stream carries about a discrete label.

    This is the measurement that has no recurrent-control equivalent. The index
    bank is never supervised, so the symbols it emits are the model's own
    vocabulary; asking how much they tell us about which target is nearest asks
    whether that vocabulary spontaneously carved a real distinction. A GRU emits
    no symbols, so there is nothing to compute -- not a worse number, no
    quantity.

    Normalised by the label entropy, so 1.0 means the symbol determines the
    label and 0.0 means it says nothing.
    """

    joint = torch.zeros(num_symbols, num_labels, dtype=torch.double)
    joint.index_put_(
        (symbols.long(), labels.long()), torch.ones(symbols.shape[0], dtype=torch.double),
        accumulate=True,
    )
    joint /= joint.sum().clamp_min(1.0)
    symbol_marginal = joint.sum(dim=1, keepdim=True)
    label_marginal = joint.sum(dim=0, keepdim=True)

    support = joint > 0
    ratio = joint[support] / (symbol_marginal @ label_marginal)[support]
    information = float((joint[support] * ratio.log()).sum())

    label_support = label_marginal[label_marginal > 0]
    label_entropy = float(-(label_support * label_support.log()).sum())
    # Purity: how often the most common label under a symbol is the right one.
    purity = float(joint.max(dim=1).values.sum())
    return {
        "mutual_information": information,
        "normalized": information / label_entropy if label_entropy > 0 else 0.0,
        "label_entropy": label_entropy,
        "purity": purity,
    }
