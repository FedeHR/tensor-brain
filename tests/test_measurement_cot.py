"""Tests for the measurement chain-of-thought experiment."""

import pytest
import torch

from experiments.measurement_cot.analysis import jensen_gap, monte_carlo_convergence, step_report
from experiments.measurement_cot.collapse import CollapseSpec, collapse_weights, index_entropy
from experiments.measurement_cot.data import build_queries, shortcut_baselines
from experiments.measurement_cot.graph import GraphSpec, LayeredDAG
from experiments.measurement_cot.model import (
    MeasurementChain,
    frontier_distribution,
    uniform_schedule,
)
from experiments.measurement_cot.train import TrainConfig, stage_schedule, train_chain


@pytest.fixture(scope="module")
def graph() -> LayeredDAG:
    return LayeredDAG(GraphSpec(layer_sizes=(40, 12, 12, 12), branching=2, seed=0))


@pytest.fixture(scope="module")
def queries(graph):
    return build_queries(graph, seed=0)


def test_children_respect_branching(graph):
    for layer, children in enumerate(graph.children):
        assert children.shape == (graph.spec.layer_sizes[layer], graph.spec.branching)
        # Sampling without replacement means every parent has distinct children.
        assert (children[:, 1:] > children[:, :-1]).all()


def test_frontier_masks_agree_with_reachability(graph):
    terminal = graph.frontier_masks[graph.spec.num_hops]
    assert torch.equal(terminal, graph.reachable)


def test_frontier_masks_are_closed_under_children(graph):
    for layer in range(graph.spec.num_hops):
        current = graph.frontier_masks[layer]
        nxt = graph.frontier_masks[layer + 1]
        for start in range(current.shape[0]):
            for position in current[start].nonzero(as_tuple=True)[0].tolist():
                assert nxt[start][graph.children[layer][position]].all()


def test_gold_path_is_a_real_path(graph):
    start, terminal = 0, int(graph.reachable[0].nonzero()[0])
    path = graph.gold_path(start, terminal)
    assert path[0] == start and path[-1] == terminal
    for layer, (here, there) in enumerate(zip(path[:-1], path[1:], strict=True)):
        assert there in graph.children[layer][here].tolist()


def test_exactly_one_terminal_is_reachable(graph, queries):
    train, test, _ = queries
    for split in (train, test):
        offset = graph.layer_offsets[graph.spec.num_hops]
        correct = torch.where(split.answer == 0, split.terminal_a, split.terminal_b) - offset
        other = torch.where(split.answer == 0, split.terminal_b, split.terminal_a) - offset
        assert graph.reachable[split.start_position, correct].all()
        assert not graph.reachable[split.start_position, other].any()


def test_split_shares_no_reachable_pair(graph, queries):
    train, test, _ = queries
    offset = graph.layer_offsets[graph.spec.num_hops]

    def pairs(split):
        correct = torch.where(split.answer == 0, split.terminal_a, split.terminal_b) - offset
        return set(zip(split.start_position.tolist(), correct.tolist(), strict=True))

    assert not (pairs(train) & pairs(test))


def test_terminal_identity_carries_no_signal(graph, queries):
    train, test, _ = queries
    baselines = shortcut_baselines(train, test, graph)
    # Quota-balanced negatives are supposed to leave a terminal-only rule at chance;
    # a rule that beat this bound would mean the task can be solved without traversing.
    assert baselines["shortcut_terminal_only"] < 0.6
    assert 0.4 < baselines["shortcut_slot_only"] < 0.6


def test_collapse_weights_are_distributions():
    scores = torch.randn(16, 12)
    for spec in (
        CollapseSpec(mode="expected"),
        CollapseSpec(mode="expected", temperature=0.3),
        CollapseSpec(mode="argmax"),
        CollapseSpec(mode="sample", samples=4),
    ):
        weights, probabilities = collapse_weights(scores, spec)
        assert torch.allclose(weights.sum(-1), torch.ones(16), atol=1e-5)
        assert torch.allclose(probabilities.sum(-1), torch.ones(16), atol=1e-5)
        assert (weights >= -1e-6).all()


def test_sample_average_converges_to_expected():
    torch.manual_seed(0)
    scores = torch.randn(1, 8)
    exact, _ = collapse_weights(scores, CollapseSpec(mode="expected"))
    generator = torch.Generator().manual_seed(0)
    draws = torch.stack(
        [
            collapse_weights(
                scores, CollapseSpec(mode="sample", samples=512), generator=generator
            )[0]
            for _ in range(16)
        ]
    )
    assert (draws.mean(0) - exact).abs().max() < 0.03


def test_reported_probabilities_ignore_temperature():
    scores = torch.randn(8, 10)
    _, cold = collapse_weights(scores, CollapseSpec(mode="expected", temperature=0.2))
    _, warm = collapse_weights(scores, CollapseSpec(mode="expected", temperature=5.0))
    # The reported belief is the model's own, so analyses are comparable across
    # conditions that happened to feed back a sharpened distribution.
    assert torch.allclose(cold, warm)


def test_straight_through_keeps_a_gradient():
    scores = torch.randn(4, 6, requires_grad=True)
    weights, _ = collapse_weights(scores, CollapseSpec(mode="argmax"))
    assert weights.sum().requires_grad
    weights.pow(2).sum().backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0


def test_hard_modes_are_one_hot_in_the_forward_pass():
    scores = torch.randn(32, 9)
    weights, _ = collapse_weights(scores, CollapseSpec(mode="argmax"))
    assert ((weights == 0) | (weights == 1)).all()


def test_entropy_matches_uniform_bound():
    uniform = torch.full((3, 16), 1 / 16)
    assert torch.allclose(index_entropy(uniform), torch.full((3,), torch.tensor(16.0).log()))


def test_retain_gate_zero_erases_the_previous_state(graph, queries):
    train, _, _ = queries
    torch.manual_seed(0)
    model = MeasurementChain(graph, state_dim=32, hidden_dim=64, retain_gate=0.0)
    batch = train.index(torch.arange(8))
    schedule = uniform_schedule(CollapseSpec(mode="argmax"), graph.spec.num_hops)
    trace = model(batch, schedule, record=True)
    # With alpha = 0 the post-feedback state is the feedback alone, so it must lie in
    # the span of a single index embedding.
    post = trace.post_feedback_state[0]
    assert torch.allclose(post.norm(dim=-1), post.norm(dim=-1)[0].expand(8), atol=1e-4)


def test_target_slots_enter_symmetrically(graph, queries):
    train, _, _ = queries
    torch.manual_seed(0)
    model = MeasurementChain(graph, state_dim=32, hidden_dim=64)
    batch = train.index(torch.arange(8))
    swapped = type(batch)(
        start=batch.start,
        terminal_a=batch.terminal_b,
        terminal_b=batch.terminal_a,
        answer=1 - batch.answer,
        start_position=batch.start_position,
        gold_path=batch.gold_path,
    )
    # Swapping the two candidate slots must not change what is written in, so the
    # question cannot be answered from the write alone.
    assert torch.allclose(model.write_query(batch), model.write_query(swapped), atol=1e-5)


def test_answer_logits_swap_with_the_slots(graph, queries):
    train, _, _ = queries
    torch.manual_seed(0)
    model = MeasurementChain(graph, state_dim=32, hidden_dim=64)
    batch = train.index(torch.arange(8))
    q = model.write_query(batch)
    swapped = type(batch)(
        start=batch.start,
        terminal_a=batch.terminal_b,
        terminal_b=batch.terminal_a,
        answer=1 - batch.answer,
        start_position=batch.start_position,
        gold_path=batch.gold_path,
    )
    assert torch.allclose(
        model.answer_logits(q, batch), model.answer_logits(q, swapped).flip(-1), atol=1e-5
    )


def test_frozen_index_bank_gets_no_gradient(graph, queries):
    train, _, _ = queries
    model = MeasurementChain(graph, state_dim=32, hidden_dim=64, learn_index_bank=False)
    schedule = uniform_schedule(CollapseSpec(mode="expected"), graph.spec.num_hops)
    trace = model(train.index(torch.arange(8)), schedule)
    trace.logits.sum().backward()
    assert model.tb.A.grad is None
    assert model.tb.evolution.V.grad is not None


def test_frontier_distribution_is_supported_on_the_frontier(graph, queries):
    train, _, _ = queries
    batch = train.index(torch.arange(16))
    for hop in range(1, graph.spec.num_hops):
        distribution = frontier_distribution(graph, batch, hop, torch.device("cpu"))
        mask = graph.frontier_masks[hop][batch.start_position]
        assert torch.allclose(distribution.sum(-1), torch.ones(16))
        assert (distribution[~mask] == 0).all()


def test_schedule_length_is_validated(graph, queries):
    train, _, _ = queries
    model = MeasurementChain(graph, state_dim=32, hidden_dim=64)
    with pytest.raises(ValueError, match="one spec per intermediate hop"):
        model(train.index(torch.arange(4)), [CollapseSpec(mode="expected")])


def test_curriculum_replaces_hops_front_to_back():
    schedule = [CollapseSpec(mode="argmax")] * 3
    for stage, expected_teacher in ((0, 3), (1, 2), (2, 1), (3, 0)):
        staged, supervised = stage_schedule(schedule, stage, total_stages=3)
        assert sum(s.mode == "teacher" for s in staged) == expected_teacher
        assert len(supervised) == expected_teacher
    # The final stage is the condition itself, with no teacher forcing left.
    staged, supervised = stage_schedule(schedule, 3, total_stages=3)
    assert all(s.mode == "argmax" for s in staged) and supervised == []


def test_tiny_data_overfits(graph, queries):
    """The required tiny-data sanity check: the chain can fit what it is shown."""

    train, _, _ = queries
    tiny = train.index(torch.arange(32))
    config = TrainConfig(
        steps=2500, batch_size=32, learning_rate=1e-2, supervision="none",
        curriculum_stages=0, seed=0, eval_every=2500,
    )
    _, result = train_chain(
        graph, tiny, tiny, uniform_schedule(CollapseSpec(mode="expected"), graph.spec.num_hops),
        config, model_kwargs={"state_dim": 128, "hidden_dim": 256, "learn_index_bank": True},
        eval_repeats=1,
    )
    assert result.train_accuracy == 1.0


def test_analysis_runs_and_reports_sane_ranges(graph, queries):
    train, test, _ = queries
    torch.manual_seed(0)
    model = MeasurementChain(graph, state_dim=32, hidden_dim=64, retain_gate=0.0)
    schedule = uniform_schedule(CollapseSpec(mode="expected"), graph.spec.num_hops)
    probe = test.index(torch.arange(min(32, len(test))))

    reports = step_report(model, probe, schedule)
    assert len(reports) == graph.spec.num_hops - 1
    for report in reports:
        assert 0.0 <= report.frontier_mass <= 1.0
        assert 0.0 <= report.entropy <= report.max_entropy + 1e-5

    rows = monte_carlo_convergence(model, probe, hop=1, sample_counts=(1, 64), trials=4)
    # More draws must not be a worse estimate of the same expectation.
    assert rows[-1]["distance"] < rows[0]["distance"]

    gaps = jensen_gap(model, probe, hop=1, feedback_gates=(0.5, 1.0))
    assert all(row["gap"] >= 0 for row in gaps)
    # The second-order term predicts a quadratic rise in the feedback gate.
    assert gaps[-1]["gap"] > gaps[0]["gap"]
