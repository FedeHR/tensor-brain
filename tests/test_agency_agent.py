"""Tests for the Tensor Brain agent schedule, rollout, and learning stages."""

import dataclasses

import pytest
import torch

from experiments.agency.agent import AgentConfig, GridAgent
from experiments.agency.baselines import GRUPolicy
from experiments.agency.conditions import CONDITIONS
from experiments.agency.diagnostics import (
    action_alignment,
    index_similarity,
    narrate_episode,
    value_landscape,
)
from experiments.agency.gridworld import (
    ACTION_NAMES,
    GridConfig,
    SymbolicForaging,
    train_cues,
)
from experiments.agency.rollout import evaluate, run_episodes, summarize
from experiments.agency.training import CloneConfig, ReinforceConfig, clone_from_oracle, reinforce
from experiments.agency.vocabulary import NOTHING, build_vocabulary

TASK = GridConfig(size=4, num_objects=3, view_radius=1, max_steps=8)


def make_agent(**overrides) -> GridAgent:
    torch.manual_seed(0)
    return GridAgent(TASK, AgentConfig(state_dim=16, hidden_dim=16, **overrides))


def make_environment(num_envs: int = 6) -> SymbolicForaging:
    return SymbolicForaging(TASK, num_envs, seed=0, allowed_cues=train_cues(3, 3))


def test_action_group_positions_match_environment_action_ids() -> None:
    """`get_candidate_positions` must return exactly the environment's action ids."""

    vocabulary = build_vocabulary(TASK)
    assert vocabulary.group_labels("action") == ACTION_NAMES


def test_percept_groups_extend_the_cue_groups_by_one_index() -> None:
    """The same column is the cue 'find red' and the percept 'I see red'."""

    vocabulary = build_vocabulary(TASK)
    for factor in ("color", "shape"):
        cue = vocabulary.indices(factor).tolist()
        percept = vocabulary.indices(f"percept_{factor}").tolist()
        assert percept[: len(cue)] == cue
        assert percept[len(cue) :] == [vocabulary.index(NOTHING)]


def test_window_cycle_returns_an_action_from_the_action_group() -> None:
    agent = make_agent()
    environment = make_environment()
    trace = agent.window_cycle(
        *agent.initial_state(environment.num_envs, torch.device("cpu")),
        environment.observation(),
        torch.zeros(environment.num_envs),
        agent.color_indices[environment.state.cue_color()],
        agent.shape_indices[environment.state.cue_shape()],
        is_first_step=torch.ones(environment.num_envs, dtype=torch.bool),
    )
    assert bool(torch.isin(trace.action_index, agent.action_indices).all())
    assert torch.equal(agent.action_indices[trace.action_position], trace.action_index)
    assert trace.action_probabilities.shape == (environment.num_envs, len(ACTION_NAMES))
    assert torch.allclose(
        trace.action_probabilities.sum(dim=-1), torch.ones(environment.num_envs), atol=1e-5
    )


def test_action_feedback_gate_controls_the_written_back_embedding() -> None:
    """`q' = alpha q + beta a_k` must be visible at the agent boundary."""

    environment = make_environment(4)
    arguments = (
        environment.observation(),
        torch.zeros(environment.num_envs),
        torch.zeros(environment.num_envs, dtype=torch.long),
        torch.zeros(environment.num_envs, dtype=torch.long),
    )
    kwargs = {"is_first_step": torch.ones(environment.num_envs, dtype=torch.bool)}

    torch.manual_seed(1)
    with_feedback = make_agent(measure_percepts=False)
    torch.manual_seed(0)
    trace_on = with_feedback.window_cycle(
        *with_feedback.initial_state(environment.num_envs, torch.device("cpu")),
        *arguments,
        **kwargs,
    )
    without = make_agent(measure_percepts=False, action_feedback_gate=0.0)
    torch.manual_seed(0)
    trace_off = without.window_cycle(
        *without.initial_state(environment.num_envs, torch.device("cpu")),
        *arguments,
        **kwargs,
    )
    expected = trace_off.q + with_feedback.brain.A.T[trace_on.action_index]
    assert torch.allclose(trace_on.q, expected, atol=1e-5)


def test_teacher_forced_action_is_the_supplied_index() -> None:
    agent = make_agent()
    environment = make_environment()
    teacher = agent.action_indices[environment.oracle_action()]
    trace = agent.window_cycle(
        *agent.initial_state(environment.num_envs, torch.device("cpu")),
        environment.observation(),
        torch.zeros(environment.num_envs),
        agent.color_indices[environment.state.cue_color()],
        agent.shape_indices[environment.state.cue_shape()],
        is_first_step=torch.ones(environment.num_envs, dtype=torch.bool),
        action_teacher=teacher,
    )
    assert torch.equal(trace.action_index, teacher)


def test_deliberation_windows_apply_the_evolution_operator_repeatedly() -> None:
    """More windows must change the state trajectory, not merely the runtime."""

    environment = make_environment(4)
    arguments = (
        environment.observation(),
        torch.zeros(environment.num_envs),
        torch.zeros(environment.num_envs, dtype=torch.long),
        torch.zeros(environment.num_envs, dtype=torch.long),
    )
    shallow = make_agent(measure_percepts=False, deliberation_windows=1)
    deep = make_agent(measure_percepts=False, deliberation_windows=3)
    torch.manual_seed(0)
    shallow_trace = shallow.window_cycle(
        *shallow.initial_state(4, torch.device("cpu")), *arguments,
        is_first_step=torch.ones(4, dtype=torch.bool),
    )
    torch.manual_seed(0)
    deep_trace = deep.window_cycle(
        *deep.initial_state(4, torch.device("cpu")), *arguments,
        is_first_step=torch.ones(4, dtype=torch.bool),
    )
    assert not torch.allclose(shallow_trace.q, deep_trace.q)


def test_percept_targets_report_nothing_when_no_object_is_in_view() -> None:
    agent = make_agent()
    environment = make_environment(2)
    visible = torch.tensor([-1, 0])
    color, shape = agent.percept_targets(
        visible, environment.state.object_color, environment.state.object_shape
    )
    assert int(color[0]) == agent.nothing_index
    assert int(shape[0]) == agent.nothing_index
    assert int(color[1]) == int(agent.color_indices[environment.state.object_color[1, 0]])


def test_reward_index_value_is_a_score_of_the_reward_column() -> None:
    agent = make_agent()
    q = torch.randn(5, agent.config.state_dim)
    expected = agent.brain.index_scores(q, agent.reward_indices).squeeze(-1)
    assert torch.allclose(agent.value_of(q), expected)


def test_run_episodes_masks_finished_environments() -> None:
    agent = make_agent()
    environment = make_environment()
    batch = run_episodes(environment, agent)
    assert bool(batch.alive[0].all())
    # `alive` may only turn off, never back on.
    assert bool((batch.alive[1:].int() <= batch.alive[:-1].int()).all())
    assert torch.equal(batch.episode_length, batch.alive.sum(dim=0))


def test_returns_to_go_are_exact_for_a_single_episode() -> None:
    agent = make_agent()
    environment = make_environment(3)
    batch = run_episodes(environment, agent)
    returns = batch.returns_to_go(0.9)
    for env_slot in range(environment.num_envs):
        length = int(batch.episode_length[env_slot])
        expected = 0.0
        for step in reversed(range(length)):
            expected = float(batch.reward[step, env_slot]) + 0.9 * expected
            assert returns[step, env_slot] == pytest.approx(expected, abs=1e-5)


def test_behavioural_cloning_reduces_the_teacher_forced_loss() -> None:
    agent = make_agent()
    environment = make_environment(32)
    log = clone_from_oracle(
        environment, agent, CloneConfig(updates=60, learning_rate=3e-3), evaluate_every=59
    )
    assert log.loss[-1] < log.loss[0]


def test_reinforce_runs_and_produces_gradients_for_the_index_matrix() -> None:
    agent = make_agent()
    environment = make_environment(16)
    before = agent.brain.A.detach().clone()
    reinforce(environment, agent, ReinforceConfig(updates=5, evaluate_every=100))
    assert not torch.allclose(before, agent.brain.A.detach())


def test_gru_control_shares_the_rollout_contract() -> None:
    torch.manual_seed(0)
    policy = GRUPolicy(TASK, state_dim=16)
    environment = make_environment(8)
    metrics = summarize(run_episodes(environment, policy))
    assert 0.0 <= metrics.success_rate <= 1.0
    reinforce(environment, policy, ReinforceConfig(updates=3, evaluate_every=100))


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_every_condition_completes_an_update(condition: str) -> None:
    """Each named ablation must run end to end on the tiny task."""

    torch.manual_seed(0)
    agent_config = CONDITIONS[condition]
    policy = (
        GRUPolicy(TASK, state_dim=16)
        if agent_config is None
        else GridAgent(TASK, dataclasses.replace(agent_config, state_dim=16, hidden_dim=16))
    )
    environment = make_environment(8)
    reinforce(environment, policy, ReinforceConfig(updates=2, evaluate_every=100))


def test_diagnostics_produce_a_complete_narration() -> None:
    agent = make_agent()
    environment = make_environment(4)
    episode = narrate_episode(environment, agent)
    assert len(episode.action_name) == len(episode.agent_row) == len(episode.value)
    assert len(episode.named_color) == len(episode.action_name)
    assert episode.cue[0] and episode.cue[1]

    landscape = value_landscape(environment, agent)
    assert landscape.shape == (TASK.size, TASK.size)
    # The layout must be restored so the environment stays usable afterwards.
    assert evaluate(environment, agent, repeats=1).success_rate >= 0.0

    similarity, labels = index_similarity(agent)
    assert similarity.shape == (len(labels), len(labels))
    assert torch.allclose(similarity.diagonal(), torch.ones(len(labels)), atol=1e-5)

    scores, cue_labels, action_labels = action_alignment(agent)
    assert scores.shape == (len(cue_labels), len(action_labels))


def test_planned_selection_picks_the_highest_imagined_value() -> None:
    """One-step imagination must agree with the action it claims to choose."""

    agent = make_agent(action_selection="planned")
    environment = make_environment(5)
    q, context = agent.initial_state(environment.num_envs, torch.device("cpu"))
    values = agent.imagined_action_values(q, context)
    assert values.shape == (environment.num_envs, len(ACTION_NAMES))
    chosen = agent.plan_action(q, context)
    assert torch.equal(chosen, agent.action_indices[values.argmax(dim=-1)])

    # Imagining must not commit the agent to anything.
    before = q.clone()
    agent.imagined_action_values(q, context)
    assert torch.equal(before, q)

    trace = agent.window_cycle(
        q,
        context,
        environment.observation(),
        torch.zeros(environment.num_envs),
        agent.color_indices[environment.state.cue_color()],
        agent.shape_indices[environment.state.cue_shape()],
        is_first_step=torch.ones(environment.num_envs, dtype=torch.bool),
    )
    assert bool(torch.isin(trace.action_index, agent.action_indices).all())


def test_planning_requires_an_evolution_operator() -> None:
    agent = make_agent(evolution="none", action_selection="planned")
    q, _ = agent.initial_state(3, torch.device("cpu"))
    with pytest.raises(RuntimeError):
        agent.plan_action(q, None)


def test_analysis_separates_escaped_from_trapped_seeds() -> None:
    """Bimodal outcomes must not be averaged into a meaningless middle."""

    from experiments.agency.analysis import summarize_condition

    per_seed = {
        "eval": [
            {"success_rate": 0.0, "first_choice_accuracy": 0.0},
            {"success_rate": 0.98, "first_choice_accuracy": 0.9},
            {"success_rate": 0.96, "first_choice_accuracy": 1.0},
        ],
        "holdout": [
            {"success_rate": 0.0, "first_choice_accuracy": 0.0},
            {"success_rate": 0.95, "first_choice_accuracy": 0.8},
            {"success_rate": 0.94, "first_choice_accuracy": 0.9},
        ],
    }
    summary = summarize_condition(
        "example", per_seed, metrics=("success_rate", "first_choice_accuracy")
    )
    assert summary.seeds == 3
    assert summary.escaped == 2
    assert summary.escape_rate == pytest.approx(2 / 3)
    # The trapped seed must be excluded from the conditional metric.
    assert summary.value("eval", "first_choice_accuracy") == pytest.approx(0.95)
    assert summary.value("holdout", "first_choice_accuracy") == pytest.approx(0.85)


def test_analysis_reports_a_never_escaping_condition_as_such() -> None:
    from experiments.agency.analysis import markdown_table, summarize_condition

    per_seed = {
        "eval": [{"success_rate": 0.0, "first_choice_accuracy": 0.0}],
        "holdout": [{"success_rate": 0.0, "first_choice_accuracy": 0.0}],
    }
    summary = summarize_condition(
        "trapped", per_seed, metrics=("success_rate", "first_choice_accuracy")
    )
    assert summary.escaped == 0
    assert "never escaped" in markdown_table({"trapped": summary})
