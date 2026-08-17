"""Tests for the COCO learned-index-layer experiment.

These use a small synthetic corpus so they run without the COCO download. The
one test that needs the real analysis code (the cross-check against
``bayes_approximation``) is skipped unless ``TB_BAYES_ROOT`` is set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from experiments.coco_heisenberg import data as D
from experiments.coco_heisenberg import evaluation as E
from experiments.coco_heisenberg import model as Mo


def make_corpus(*, images: int = 600, categories: int = 6, symbols: int = 30) -> D.Corpus:
    """A corpus where each symbol is evidence for one category, plus noise."""

    rng = np.random.default_rng(0)
    presence = (rng.random((images, categories)) < 0.35).astype(np.uint8)
    owner = rng.integers(0, categories, size=symbols)
    rows: list[list[int]] = []
    for i in range(images):
        active = np.flatnonzero(presence[i])
        pool = [k for k in range(symbols) if owner[k] in active]
        if len(pool) < 2:  # keep the latent well defined
            presence[i, rng.integers(0, categories)] = 1
            active = np.flatnonzero(presence[i])
            pool = [k for k in range(symbols) if owner[k] in active]
        take = rng.choice(pool, size=min(4, len(pool)), replace=False)
        rows.append(sorted(int(k) for k in take))
    return D.Corpus(
        presence=presence,
        symbols=rows,
        vocabulary=[f"w{k}" for k in range(symbols)],
        categories=[f"c{i}" for i in range(categories)],
    )


def test_tokenizer_drops_stopwords_and_short_words():
    assert D._tokenize("A man is sitting on the TOP of a red bus.") == [
        "man", "sitting", "top", "red", "bus"
    ]


def test_sufficient_statistics_match_the_expanded_pairs():
    corpus = make_corpus()
    states, counts = Mo.sufficient_statistics(corpus)
    presence, symbols = Mo._design(corpus)

    assert counts.sum() == len(symbols)
    # every (x, k) pair must land in the row for its own presence pattern
    weights = (2 ** np.arange(corpus.num_categories - 1, -1, -1)).astype(np.int64)
    pairs = zip(presence.numpy().astype(np.int64)[:50], symbols.numpy()[:50], strict=True)
    for row, k in pairs:
        assert counts[int(row @ weights), int(k)] > 0
    assert torch.equal(states[int(presence[0].numpy().astype(np.int64) @ weights)], presence[0])


def test_fitting_beats_the_uniform_symbol_model():
    corpus = make_corpus()
    layer = Mo.fit_index_layer(corpus, steps=200, learning_rate=0.3)
    assert Mo.held_out_nll(layer, corpus) < Mo.uniform_symbol_nll(corpus) - 0.2
    assert layer.A.shape == (corpus.num_categories, corpus.num_symbols)
    assert torch.isfinite(layer.joint_prior).all()
    assert abs(float(layer.joint_prior.sum()) - 1.0) < 1e-9


def test_exact_posterior_is_a_proper_distribution_and_uses_the_evidence():
    corpus = make_corpus()
    layer = Mo.fit_index_layer(corpus, steps=200, learning_rate=0.3)
    pre = E.precompute(layer)
    joint = E.exact_joint(pre, [0, 1, 2])
    assert abs(float(joint.sum()) - 1.0) < 1e-12
    # absorbing evidence must move the belief away from the prior
    prior_gamma = torch.sigmoid(pre.q_prior)
    posterior_gamma = joint @ pre.states
    assert float((posterior_gamma - prior_gamma).abs().max()) > 1e-3


def test_heisenberg_is_exactly_order_invariant_and_corrections_are_not():
    corpus = make_corpus()
    layer = Mo.fit_index_layer(corpus, steps=200, learning_rate=0.3)
    pre = E.precompute(layer)
    correction = E.affine_correction(pre)
    symbols = [3, 9, 14, 21]

    def run(order):
        return E.run_rules(
            pre, order, correction=correction,
            rules=("heisenberg", "heisenberg-gauge", "heisenberg-pe"),
        )

    forward, backward = run(symbols), run(symbols[::-1])
    assert float((forward["heisenberg"] - backward["heisenberg"]).abs().max()) < 1e-12
    assert float((forward["heisenberg-gauge"] - backward["heisenberg-gauge"]).abs().max()) < 1e-12
    # the state-dependent correction is the one that gives up order invariance
    assert float((forward["heisenberg-pe"] - backward["heisenberg-pe"]).abs().max()) > 1e-9


def test_affine_correction_reduces_log_partition_variance():
    corpus = make_corpus()
    layer = Mo.fit_index_layer(corpus, steps=200, learning_rate=0.3)
    stats = E.log_partition_stats(E.precompute(layer))
    assert stats["var_residual"] <= stats["var_log_partition"] + 1e-12
    assert 0.0 <= stats["fraction_affine"] <= 1.0


def test_average_precision_matches_a_hand_worked_case():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    # hits at ranks 1 and 3: (1/1 + 2/3) / 2
    assert E._average_precision(scores, labels) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_paired_difference_recovers_a_known_shift():
    per_image = {
        "a": {"nll": np.zeros(500), "accuracy": np.zeros(500), "joint_kl": np.empty(0)},
        "b": {"nll": np.full(500, 0.25), "accuracy": np.zeros(500), "joint_kl": np.empty(0)},
    }
    stat = E.paired_difference(per_image, "a", "b")
    assert stat["mean"] == pytest.approx(-0.25)
    assert stat["left_better_fraction"] == 1.0
    assert stat["ci_low"] <= -0.25 <= stat["ci_high"]


def test_evaluation_runs_and_orders_the_prior_last():
    corpus = make_corpus()
    layer = Mo.fit_index_layer(corpus, steps=200, learning_rate=0.3)
    rows = E.evaluate(layer, corpus, num_symbols=3, limit=120, seed=1)
    assert {"prior", "heisenberg", "exact"} <= set(rows)
    # absorbing evidence must beat ignoring it
    assert rows["heisenberg"]["nll"] < rows["prior"]["nll"]
    assert rows["exact"]["marginal_kl"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not os.environ.get("TB_BAYES_ROOT"), reason="needs TB_BAYES_ROOT")
def test_fast_path_matches_the_reference_implementation():
    corpus = make_corpus()
    layer = Mo.fit_index_layer(corpus, steps=200, learning_rate=0.3)
    deltas = E.cross_check(layer, [1, 5, 11])
    assert max(deltas.values()) < 1e-9
