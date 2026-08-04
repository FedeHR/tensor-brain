"""Tests for the BabyAI / MiniGrid integration."""

import pytest
import torch

pytest.importorskip("minigrid", reason="install the `minigrid` extra to run these")

from experiments.agency.agent import AgentConfig  # noqa: E402
from experiments.agency.minigrid.agent import (  # noqa: E402
    MiniGridAgent,
    RecurrentControl,
    SymbolicViewEncoder,
)
from experiments.agency.minigrid.conditions import CONDITIONS, LEVEL_CONDITIONS  # noqa: E402
from experiments.agency.minigrid.diagnostics import narrate_episode  # noqa: E402
from experiments.agency.minigrid.env import (  # noqa: E402
    VIEW_SIZE,
    VectorMiniGrid,
    cue_combinations,
    diagonal_holdout,
)
from experiments.agency.minigrid.ppo import (  # noqa: E402
    PPOConfig,
    collect,
    evaluate,
    generalized_advantage,
    replay,
    train,
)
from experiments.agency.minigrid.vocabulary import (  # noqa: E402
    ANY_COLOR,
    NOTHING,
    build_vocabulary,
    parse_mission,
)

LEVEL = "BabyAI-GoToLocal-v0"


def small_config(**overrides) -> AgentConfig:
    return AgentConfig(state_dim=16, hidden_dim=16, **overrides)


# ------------------------------------------------------------------ vocabulary


def test_cue_and_percept_groups_share_their_colour_columns() -> None:
    """The column that names a colour must be the column that requests it."""

    vocabulary = build_vocabulary()
    for name in ("red", "green", "blue"):
        assert vocabulary.index(name) in vocabulary.indices("color").tolist()
        assert vocabulary.index(name) in vocabulary.indices("percept_color").tolist()
    # Only the percept groups carry the "nothing observed" label.
    assert vocabulary.index(NOTHING) in vocabulary.indices("percept_color").tolist()
    assert vocabulary.index(NOTHING) not in vocabulary.indices("color").tolist()


@pytest.mark.parametrize(
    ("mission", "color", "object_type"),
    [
        ("go to the red ball", "red", "ball"),
        ("go to a green key", "green", "key"),
        ("open the yellow door", "yellow", "door"),
        ("go to the box", ANY_COLOR, "box"),
        # A compound mission is matched at its first referable phrase. This one
        # is the same in every DoorKey episode, so the cue is a constant.
        ("use the key to open the door and then get to the goal", ANY_COLOR, "door"),
        ("wander around aimlessly", ANY_COLOR, "key"),
    ],
)
def test_mission_parsing(mission: str, color: str, object_type: str) -> None:
    cue = parse_mission(mission)
    assert (cue.color, cue.object_type) == (color, object_type)


def test_diagonal_holdout_keeps_every_factor_in_training() -> None:
    """A held-out mission must be unseen only as a combination."""

    combinations = {(c, o) for c in ("red", "green", "blue") for o in ("key", "ball", "box")}
    holdout = diagonal_holdout(combinations)
    training = combinations - holdout
    assert holdout and holdout < combinations
    assert {c for c, _ in training} == {c for c, _ in combinations}
    assert {o for _, o in training} == {o for _, o in combinations}


# ------------------------------------------------------------------ environment


def test_environment_exposes_the_agent_contract() -> None:
    environment = VectorMiniGrid(LEVEL, 4, seed=0)
    assert environment.observation().shape == (4, environment.observation_dim)
    assert environment.observation_dim == VIEW_SIZE * VIEW_SIZE * 3 + 1
    color, object_type = environment.cue_indices()
    assert color.shape == object_type.shape == (4,)
    assert environment.max_steps > 0
    environment.close()


def test_cue_restriction_excludes_held_out_missions() -> None:
    combinations = cue_combinations(LEVEL, samples=120)
    holdout = diagonal_holdout(combinations)
    environment = VectorMiniGrid(
        LEVEL, 8, seed=1, allowed_cues=frozenset(combinations) - holdout
    )
    vocabulary = environment.vocabulary
    for _ in range(6):
        colors, objects = environment.cue_indices()
        seen = {
            (vocabulary.label(int(c)), vocabulary.label(int(o)))
            for c, o in zip(colors.tolist(), objects.tolist(), strict=True)
        }
        assert seen.isdisjoint(holdout)
        environment.step(torch.zeros(8, dtype=torch.long))
    environment.close()


def test_percept_targets_come_from_the_visible_view_only() -> None:
    """Naming must be solvable from the observation, never from simulator state."""

    environment = VectorMiniGrid(LEVEL, 4, seed=2)
    colors, objects = environment.percept_targets()
    vocabulary = environment.vocabulary
    assert colors.shape == objects.shape == (4,)
    for index in objects.tolist():
        assert index in vocabulary.indices("percept_object").tolist()
    # Blanking the view must make every slot report "nothing".
    environment._images[:] = 1  # the `empty` object code
    colors, objects = environment.percept_targets()
    assert set(objects.tolist()) == {vocabulary.index(NOTHING)}
    environment.close()


def test_finished_episodes_reset_immediately() -> None:
    environment = VectorMiniGrid(LEVEL, 4, seed=3, max_steps=3)
    for _ in range(3):
        result = environment.step(torch.zeros(4, dtype=torch.long))
    assert bool(result.done.all())
    assert environment.observation().shape == (4, environment.observation_dim)
    environment.close()


# ----------------------------------------------------------------------- agent


def test_symbolic_encoder_maps_a_packed_view_into_pre_cbs() -> None:
    encoder = SymbolicViewEncoder(16, embed_dim=4)
    environment = VectorMiniGrid(LEVEL, 3, seed=4)
    drive = encoder(environment.observation())
    assert drive.shape == (3, 16)
    environment.close()


def test_agent_measures_actions_from_the_action_group() -> None:
    environment = VectorMiniGrid(LEVEL, 3, seed=5)
    agent = MiniGridAgent(small_config())
    trace = agent.window_cycle(
        *agent.initial_state(3, torch.device("cpu")),
        environment.observation(),
        torch.zeros(3),
        *environment.cue_indices(),
    )
    assert bool(torch.isin(trace.action_index, agent.action_indices).all())
    assert trace.action_probabilities.shape == (3, environment.num_actions)
    environment.close()


def test_agent_and_control_share_the_rollout_contract() -> None:
    environment = VectorMiniGrid(LEVEL, 3, seed=6)
    for policy in (
        MiniGridAgent(small_config()),
        RecurrentControl(small_config(), cell="gru"),
        RecurrentControl(small_config(), cell="lstm"),
    ):
        state, context = policy.initial_state(3, torch.device("cpu"))
        trace = policy.window_cycle(
            state, context, environment.observation(), torch.zeros(3),
            *environment.cue_indices(),
        )
        state, context = policy.reset_finished(
            trace.q, trace.context, torch.ones(3, dtype=torch.bool)
        )
        assert float(state.abs().sum()) == 0.0
    environment.close()


# ------------------------------------------------------------------------- ppo


def test_replay_reproduces_the_collected_log_probabilities() -> None:
    """PPO is only valid if a stored segment can be re-run exactly.

    Teacher-forcing the action *and* the two perceptual samples must reproduce
    the same recurrent trajectory, so an unchanged policy must return exactly the
    log-probabilities recorded during collection.
    """

    torch.manual_seed(0)
    environment = VectorMiniGrid(LEVEL, 4, seed=7)
    agent = MiniGridAgent(small_config())
    state, context = agent.initial_state(4, torch.device("cpu"))
    from experiments.agency.minigrid.ppo import EpisodeTracker

    segment, *_ = collect(
        environment, agent, state, context, torch.zeros(4), EpisodeTracker(4), 12
    )
    with torch.no_grad():
        log_probabilities, _, values = replay(agent, segment)
    assert torch.allclose(log_probabilities, segment.log_probability, atol=1e-5)
    assert torch.allclose(values, segment.value, atol=1e-5)
    environment.close()


def test_generalized_advantage_matches_a_direct_computation() -> None:
    torch.manual_seed(0)
    environment = VectorMiniGrid(LEVEL, 2, seed=8)
    agent = MiniGridAgent(small_config())
    from experiments.agency.minigrid.ppo import EpisodeTracker

    segment, *_ = collect(
        environment, agent, *agent.initial_state(2, torch.device("cpu")),
        torch.zeros(2), EpisodeTracker(2), 6,
    )
    final = torch.zeros(2)
    advantages, targets = generalized_advantage(
        segment, final, discount=0.9, gae_lambda=0.8
    )
    expected = torch.zeros_like(advantages)
    running = torch.zeros(2)
    next_value = final
    for step in reversed(range(6)):
        alive = (~segment.done[step]).float()
        delta = segment.reward[step] + 0.9 * next_value * alive - segment.value[step]
        running = delta + 0.9 * 0.8 * alive * running
        expected[step] = running
        next_value = segment.value[step]
    assert torch.allclose(advantages, expected, atol=1e-6)
    assert torch.allclose(targets, expected + segment.value, atol=1e-6)
    environment.close()


def test_ppo_updates_the_index_matrix() -> None:
    torch.manual_seed(0)
    environment = VectorMiniGrid(LEVEL, 4, seed=9)
    agent = MiniGridAgent(small_config())
    before = agent.brain.A.detach().clone()
    train(environment, agent, PPOConfig(updates=2, segment_steps=8, epochs=2, evaluate_every=99))
    assert not torch.allclose(before, agent.brain.A.detach())
    environment.close()


def test_ppo_rejects_a_non_replayable_deliberation_mode() -> None:
    environment = VectorMiniGrid(LEVEL, 2, seed=10)
    agent = MiniGridAgent(small_config(deliberation_windows=2, deliberation_mode="measure"))
    with pytest.raises(ValueError, match="replayed exactly"):
        train(environment, agent, PPOConfig(updates=1, segment_steps=4))
    environment.close()


def test_evaluate_reports_bounded_metrics() -> None:
    environment = VectorMiniGrid(LEVEL, 4, seed=11, max_steps=8)
    agent = MiniGridAgent(small_config())
    metrics = evaluate(environment, agent, episodes=8)
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert metrics["episodes"] >= 8
    environment.close()


# ----------------------------------------------------------------- diagnostics


def test_narrated_episode_records_symbols_and_frames() -> None:
    environment = VectorMiniGrid(LEVEL, 1, seed=12, render=True, max_steps=10)
    agent = MiniGridAgent(small_config())
    episode = narrate_episode(environment, agent)
    assert episode.length == len(episode.action_name) == len(episode.value)
    assert len(episode.named_color) == episode.length
    assert episode.frames and next(iter(episode.frames.values())).ndim == 3
    assert episode.mission
    environment.close()


def test_every_level_condition_is_defined() -> None:
    for level, names in LEVEL_CONDITIONS.items():
        assert names, level
        for name in names:
            assert name in CONDITIONS
