"""Tests for the stage-0 diffusion probe.

A tiny stand-in model exercises the trajectory walk without downloading weights,
so the whole file runs offline in well under a second.
"""

from __future__ import annotations

import pytest
import torch

from experiments.diffusion_heisenberg import probe as P

VOCAB = 24
MASK = 23


class ToyModel:
    """A masked LM whose only rule is that committing token 5 favours token 6.

    That gives the trajectory something real to find: an interaction between
    positions that a token-only additive correction can, in principle, capture.
    """

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        length = x.shape[1]
        logits = torch.zeros(1, length, VOCAB)
        logits[0, :, 1] = 1.0  # a bland favourite everywhere
        if (x[0] == 5).any():
            logits[0, :, 6] += 3.0
        for position in range(length):
            logits[0, position, 2] += 0.1 * position  # break ties by position
        return self._Out(logits)


def test_kl_is_zero_for_identical_rows_and_positive_otherwise():
    a = torch.tensor([0.5, 0.3, 0.2]).log()
    b = torch.tensor([0.2, 0.3, 0.5]).log()
    assert float(P._kl(a, a)) == pytest.approx(0.0, abs=1e-12)
    assert float(P._kl(a, b)) > 0.0


def test_best_scaled_kl_never_loses_to_doing_nothing():
    reference = torch.tensor([0.7, 0.2, 0.1]).log()
    baseline = torch.tensor([0.3, 0.3, 0.4]).log()
    direction = torch.tensor([5.0, -2.0, -3.0])  # deliberately far too large
    scales = torch.linspace(0.0, 1.5, 16)

    value, scale = P.best_scaled_kl(reference, baseline, direction, scales)
    assert value <= float(P._kl(reference, baseline)) + 1e-9
    assert 0.0 <= scale <= 1.5


def test_best_scaled_kl_finds_a_useful_direction():
    baseline = torch.tensor([0.3, 0.3, 0.4]).log()
    direction = torch.tensor([1.0, 0.0, -1.0])
    reference = (baseline + 0.8 * direction).log_softmax(-1)
    scales = torch.linspace(0.0, 1.5, 16)

    value, scale = P.best_scaled_kl(reference, baseline, direction, scales)
    assert value < 1e-3
    assert scale == pytest.approx(0.8, abs=0.11)


def test_accumulator_leave_one_out_excludes_the_observation():
    acc = P.Accumulators(vocab_size=3, tokens=[7])
    first = torch.tensor([1.0, 0.0, -1.0])
    second = torch.tensor([3.0, 0.0, -3.0])
    acc.add(7, first)
    acc.add(7, second)

    assert torch.allclose(acc.leave_one_out(7, first), second)
    assert torch.allclose(acc.leave_one_out(7, second), first)
    assert float(acc.count[0]) == 2


def test_accumulator_declines_a_single_observation():
    acc = P.Accumulators(vocab_size=3, tokens=[7])
    acc.add(7, torch.ones(3))
    assert acc.leave_one_out(7, torch.ones(3)) is None


def test_trajectory_commits_and_yields_before_and_after():
    model = ToyModel()
    x = torch.full((1, 8), MASK, dtype=torch.long)
    block = slice(0, 8)

    events = list(P.trajectory(model, x, MASK, block, steps=3,
                               max_targets=4, exclude=frozenset({MASK})))
    assert events, "the walk should produce at least one commit"
    for token, position, targets, before, after in events:
        assert token != MASK
        assert position not in targets
        assert before.shape == after.shape == (8, VOCAB)
        # the committed position is really written into the sequence
        assert int(x[0, position]) == token

    # one forward per commit, plus the initial one -- not a fresh decode each time
    assert model.calls == len(events) + 1


def test_trajectory_detects_a_real_interaction():
    """Committing token 5 must visibly move the other positions' beliefs."""

    model = ToyModel()
    x = torch.full((1, 6), MASK, dtype=torch.long)
    x[0, 0] = 5  # the trigger is already present
    block = slice(1, 6)
    before = model(x).logits[0, block].log_softmax(-1)

    clean = torch.full((1, 6), MASK, dtype=torch.long)
    baseline = model(clean).logits[0, block].log_softmax(-1)
    assert float(P._kl(before[0], baseline[0])) > 0.1


def test_special_tokens_collects_the_scaffolding_ids():
    class ToyTokenizer:
        all_special_ids = [1, 2]

        def convert_tokens_to_ids(self, token):
            return {"<|endoftext|>": 3, "<|im_end|>": 4}.get(token, -1)

    ids = P.special_tokens(ToyTokenizer())
    assert {1, 2, 3, 4} <= ids
    assert -1 not in ids


def test_report_averages_only_what_it_was_given():
    report = P.Report()
    report.add("a", 1.0)
    report.add("a", 3.0)
    summary = report.summary()
    assert summary == {"a": 2.0}
