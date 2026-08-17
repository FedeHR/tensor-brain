"""Score every update rule on held-out COCO images.

Two families of measurement, in the order the thesis should report them:

**Downstream.** The belief ``gamma`` is a prediction of which supercategories the
image really contains, so it is scored against the instance annotations: NLL,
accuracy, macro-F1, average precision, calibration, and exact-set accuracy. This
is what a consumer of the state would actually experience.

**Fidelity.** KL to the exact posterior *under the same learned model*, which
isolates the update rule from model quality. The scoping work showed the two
families can rank methods differently, so both are reported and neither is
allowed to stand in for the other.

Rules are evaluated against a shared precomputation (all ``2^n`` states, their
scores and log-partitions), which makes an M-sweep over thousands of images cheap.
:func:`cross_check` verifies the fast path against the reference implementation in
``experiments.bayes_approximation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import Corpus
from .model import IndexLayer

DTYPE = torch.float64


@dataclass(frozen=True)
class Precomputed:
    """Everything shared across images: states, scores, log-partition, priors."""

    states: torch.Tensor        # [configs, categories]
    log_cond: torch.Tensor      # [configs, symbols]  log P(k | x)
    log_partition: torch.Tensor  # [configs]
    log_prior_factorized: torch.Tensor   # [configs]
    log_prior_empirical: torch.Tensor    # [configs]
    q_prior: torch.Tensor       # [categories]
    A: torch.Tensor
    a0: torch.Tensor

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])


def precompute(layer: IndexLayer) -> Precomputed:
    n = layer.state_dim
    codes = torch.arange(2**n).unsqueeze(1)
    bit = 2 ** torch.arange(n - 1, -1, -1, dtype=torch.long)
    states = (codes & bit).ne(0).to(DTYPE)

    scores = layer.a0.unsqueeze(0) + states @ layer.A
    log_partition = torch.logsumexp(scores, dim=-1)
    log_cond = scores - log_partition.unsqueeze(1)

    q = layer.q_prior
    log_prior_factorized = (states * q).sum(-1) - torch.nn.functional.softplus(q).sum()
    log_prior_empirical = layer.joint_prior.clamp_min(1e-300).log()

    return Precomputed(
        states=states,
        log_cond=log_cond,
        log_partition=log_partition,
        log_prior_factorized=log_prior_factorized,
        log_prior_empirical=log_prior_empirical,
        q_prior=q,
        A=layer.A,
        a0=layer.a0,
    )


def affine_correction(pre: Precomputed) -> torch.Tensor:
    """Least-squares slope of ``log Z`` under the prior; the free, order-invariant fix."""

    weight = torch.softmax(pre.log_prior_factorized, dim=0)
    design = torch.cat([torch.ones(pre.states.shape[0], 1, dtype=DTYPE), pre.states], dim=1)
    root = weight.sqrt().unsqueeze(1)
    solution = torch.linalg.lstsq(
        design * root, (pre.log_partition * weight.sqrt()).unsqueeze(1)
    ).solution
    return solution[1:, 0]


def log_partition_stats(pre: Precomputed) -> dict[str, float]:
    weight = torch.softmax(pre.log_prior_factorized, dim=0)
    mean = (weight * pre.log_partition).sum()
    var = float((weight * (pre.log_partition - mean) ** 2).sum())

    c = affine_correction(pre)
    fitted = (pre.states * c).sum(-1)
    residual = pre.log_partition - fitted
    residual_mean = (weight * residual).sum()
    residual_var = float((weight * (residual - residual_mean) ** 2).sum())
    return {
        "var_log_partition": var,
        "var_residual": residual_var,
        "fraction_affine": 1.0 - residual_var / var if var > 0 else float("nan"),
        "mean_partition": float(mean),
    }


# ---------------------------------------------------------------------------
# Update rules. Each returns gamma, the factorized belief over presence bits.
# ---------------------------------------------------------------------------


def _marginals(log_joint: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    posterior = torch.softmax(log_joint, dim=0)
    return posterior @ states


def _joint(log_joint: torch.Tensor) -> torch.Tensor:
    return torch.softmax(log_joint, dim=0)


def _gamma_to_joint(gamma: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    g = gamma.clamp(1e-12, 1 - 1e-12)
    log_joint = (states * g.log() + (1 - states) * (1 - g).log()).sum(-1)
    return torch.softmax(log_joint, dim=0)


def run_rules(
    pre: Precomputed,
    symbols: list[int],
    *,
    correction: torch.Tensor,
    rules: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Return ``{rule: gamma}`` for one image's revealed symbols."""

    ks = torch.as_tensor(symbols, dtype=torch.long)
    m = len(symbols)
    evidence = pre.A[:, ks].sum(dim=1)
    out: dict[str, torch.Tensor] = {}

    if "prior" in rules:
        out["prior"] = torch.sigmoid(pre.q_prior)

    if "heisenberg" in rules:
        out["heisenberg"] = torch.sigmoid(pre.q_prior + evidence)

    if "heisenberg-gauge" in rules:
        out["heisenberg-gauge"] = torch.sigmoid(pre.q_prior + evidence - m * correction)

    if "heisenberg-pe" in rules:
        q = pre.q_prior.clone()
        for k in ks:
            p = torch.softmax(pre.a0 + pre.A.t() @ torch.sigmoid(q), dim=0)
            q = q + pre.A[:, k] - pre.A @ p
        out["heisenberg-pe"] = torch.sigmoid(q)

    if "exact" in rules or "bayes-marginals" in rules:
        log_joint = pre.log_prior_factorized + pre.log_cond[:, ks].sum(dim=1)
        gamma = _marginals(log_joint, pre.states)
        if "exact" in rules:
            out["exact"] = gamma
        if "bayes-marginals" in rules:
            out["bayes-marginals"] = gamma

    if "exact-empirical-prior" in rules:
        log_joint = pre.log_prior_empirical + pre.log_cond[:, ks].sum(dim=1)
        out["exact-empirical-prior"] = _marginals(log_joint, pre.states)

    if "adf" in rules:
        q = pre.q_prior.clone()
        for k in ks:
            log_prior = (pre.states * q).sum(-1) - torch.nn.functional.softplus(q).sum()
            gamma = _marginals(log_prior + pre.log_cond[:, k], pre.states)
            gamma = gamma.clamp(1e-12, 1 - 1e-12)
            q = torch.log(gamma) - torch.log1p(-gamma)
        out["adf"] = torch.sigmoid(q)

    return out


def exact_joint(pre: Precomputed, symbols: list[int], *, empirical: bool = False) -> torch.Tensor:
    ks = torch.as_tensor(symbols, dtype=torch.long)
    base = pre.log_prior_empirical if empirical else pre.log_prior_factorized
    return _joint(base + pre.log_cond[:, ks].sum(dim=1))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, computed as in sklearn's AP."""

    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    true_positive = np.cumsum(labels)
    precision = true_positive / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


def _expected_calibration_error(prob: np.ndarray, actual: np.ndarray, bins: int = 12) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(prob[mask].mean() - actual[mask].mean())
    return float(total)


@dataclass
class Accumulator:
    """Collects per-image results for one rule and reduces them to a metric row."""

    nll: list[float]
    accuracy: list[float]
    exact_set: list[float]
    seconds: list[float]
    joint_kl: list[float]
    marginal_kl: list[float]
    predicted: list[np.ndarray]
    actual: list[np.ndarray]

    @classmethod
    def empty(cls) -> Accumulator:
        return cls([], [], [], [], [], [], [], [])

    def add(self, gamma: torch.Tensor, truth: torch.Tensor, seconds: float) -> None:
        g = gamma.clamp(1e-12, 1 - 1e-12)
        self.nll.append(float(-(truth * g.log() + (1 - truth) * (1 - g).log()).sum()))
        hit = (g > 0.5).to(DTYPE) == truth
        self.accuracy.append(float(hit.to(DTYPE).mean()))
        self.exact_set.append(float(hit.all()))
        self.seconds.append(seconds)
        self.predicted.append(g.numpy())
        self.actual.append(truth.numpy())

    def add_fidelity(self, joint_kl: float, marginal_kl: float) -> None:
        self.joint_kl.append(joint_kl)
        self.marginal_kl.append(marginal_kl)

    def summary(self) -> dict[str, float]:
        predicted = np.stack(self.predicted)
        actual = np.stack(self.actual)
        per_category_ap = [
            _average_precision(predicted[:, i], actual[:, i]) for i in range(predicted.shape[1])
        ]
        threshold = predicted > 0.5
        f1 = []
        for i in range(predicted.shape[1]):
            tp = float((threshold[:, i] & (actual[:, i] > 0.5)).sum())
            fp = float((threshold[:, i] & (actual[:, i] < 0.5)).sum())
            fn = float((~threshold[:, i] & (actual[:, i] > 0.5)).sum())
            denominator = 2 * tp + fp + fn
            f1.append(2 * tp / denominator if denominator > 0 else float("nan"))
        row = {
            "nll": float(np.mean(self.nll)),
            "accuracy": float(np.mean(self.accuracy)),
            "exact_set": float(np.mean(self.exact_set)),
            "macro_f1": float(np.nanmean(f1)),
            "mean_ap": float(np.nanmean(per_category_ap)),
            "ece": _expected_calibration_error(predicted.ravel(), actual.ravel()),
            "ms": 1000.0 * float(np.mean(self.seconds)),
        }
        if self.joint_kl:
            row["joint_kl"] = float(np.mean(self.joint_kl))
            row["marginal_kl"] = float(np.mean(self.marginal_kl))
        return row


DEFAULT_RULES = (
    "prior",
    "heisenberg",
    "heisenberg-gauge",
    "heisenberg-pe",
    "adf",
    "exact",
    "exact-empirical-prior",
)


def evaluate(
    layer: IndexLayer,
    corpus: Corpus,
    *,
    num_symbols: int,
    rules: tuple[str, ...] = DEFAULT_RULES,
    seed: int = 0,
    limit: int | None = None,
    fidelity: bool = True,
    pre: Precomputed | None = None,
    return_per_image: bool = False,
):
    """Score every rule after absorbing ``num_symbols`` revealed words per image."""

    import time

    pre = pre or precompute(layer)
    correction = affine_correction(pre)
    rng = np.random.default_rng(seed)

    accumulators = {name: Accumulator.empty() for name in rules}
    order = np.arange(corpus.num_images)
    if limit is not None and limit < len(order):
        order = rng.choice(order, size=limit, replace=False)

    for index in order:
        available = corpus.symbols[int(index)]
        if len(available) < num_symbols:
            continue
        revealed = rng.permutation(len(available))[:num_symbols]
        symbols = [int(available[j]) for j in sorted(revealed)]
        truth = torch.as_tensor(corpus.presence[int(index)], dtype=DTYPE)

        if fidelity:
            reference = exact_joint(pre, symbols)
            reference_marginals = reference @ pre.states

        for name in rules:
            start = time.perf_counter()
            gamma = run_rules(pre, symbols, correction=correction, rules=(name,))[name]
            elapsed = time.perf_counter() - start
            accumulators[name].add(gamma, truth, elapsed)
            if fidelity:
                approximate = _gamma_to_joint(gamma, pre.states)
                joint_kl = float(
                    (reference * (reference.clamp_min(1e-300).log()
                                  - approximate.clamp_min(1e-300).log())).sum()
                )
                g = gamma.clamp(1e-12, 1 - 1e-12)
                r = reference_marginals.clamp(1e-12, 1 - 1e-12)
                marginal_kl = float(
                    (r * (r.log() - g.log()) + (1 - r) * ((1 - r).log() - (1 - g).log())).sum()
                )
                accumulators[name].add_fidelity(joint_kl, marginal_kl)

    summary = {name: acc.summary() for name, acc in accumulators.items() if acc.nll}
    if not return_per_image:
        return summary
    per_image = {
        name: {
            "nll": np.asarray(acc.nll),
            "accuracy": np.asarray(acc.accuracy),
            "joint_kl": np.asarray(acc.joint_kl) if acc.joint_kl else np.empty(0),
        }
        for name, acc in accumulators.items() if acc.nll
    }
    return summary, per_image


def paired_difference(
    per_image: dict[str, dict[str, np.ndarray]],
    left: str,
    right: str,
    *,
    metric: str = "nll",
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the paired ``left - right`` difference over images.

    The rules see identical evidence on identical images in identical order, so
    the comparison is paired and the per-image difference is the right unit. For
    NLL, negative means ``left`` is better.
    """

    a, b = per_image[left][metric], per_image[right][metric]
    delta = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(resamples, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "left_better_fraction": float((delta < 0).mean()),
        "images": int(len(delta)),
    }


def load_reference(root: str | None = None):
    """Import the analysis modules from the ``bayes-approximation`` worktree.

    Both worktrees ship a top-level ``experiments`` package, so the local one
    always shadows the other on ``PYTHONPATH``. Bind the analysis package under
    its own name instead, which keeps its relative imports intact.
    """

    import importlib
    import os
    import sys
    import types
    from pathlib import Path

    root = root or os.environ.get("TB_BAYES_ROOT")
    if not root:
        raise RuntimeError("set TB_BAYES_ROOT to the bayes-approximation worktree")

    package_dir = Path(root).expanduser().resolve() / "experiments" / "bayes_approximation"
    if not package_dir.is_dir():
        raise RuntimeError(f"no bayes_approximation package under {package_dir}")

    if "tb_bayes" not in sys.modules:
        package = types.ModuleType("tb_bayes")
        package.__path__ = [str(package_dir)]
        sys.modules["tb_bayes"] = package
        source_src = str(Path(root).expanduser().resolve() / "src")
        if source_src not in sys.path:
            sys.path.insert(0, source_src)

    return (
        importlib.import_module("tb_bayes.general"),
        importlib.import_module("tb_bayes.inference"),
        importlib.import_module("tb_bayes.model"),
    )


def cross_check(
    layer: IndexLayer, symbols: list[int], *, atol: float = 1e-9, root: str | None = None
) -> dict[str, float]:
    """Verify the fast path against the reference implementation."""

    general, inference, reference = load_reference(root)

    model = reference.GenerativeModel(q_prior=layer.q_prior, A=layer.A, a0=layer.a0)
    pre = precompute(layer)
    correction = affine_correction(pre)
    n = layer.state_dim
    mine = run_rules(
        pre, symbols, correction=correction,
        rules=("heisenberg", "heisenberg-gauge", "heisenberg-pe", "adf", "exact"),
    )
    theirs = {
        "heisenberg": inference.tb(model, symbols).marginals(n),
        "heisenberg-gauge": general.tb_affine(
            model, symbols, general.affine_correction(model)[0]
        ).marginals(n),
        "heisenberg-pe": inference.tb_predictive_error(
            model, symbols, mode="gradient"
        ).marginals(n),
        "adf": inference.assumed_density_filter(model, symbols).marginals(n),
        "exact": inference.exact(model, symbols).marginals(n),
    }
    deltas = {name: float((mine[name] - theirs[name]).abs().max()) for name in theirs}
    worst = max(deltas.values())
    if worst > atol:
        raise AssertionError(f"fast path disagrees with the reference: {deltas}")
    return deltas
