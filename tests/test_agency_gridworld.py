"""Behavioural tests for the symbolic-foraging gridworld."""

import torch

from experiments.agency.gridworld import (
    COLLECT,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_WEST,
    GridConfig,
    SymbolicForaging,
    latin_square_holdout,
    train_cues,
)


def make_environment(**overrides) -> SymbolicForaging:
    config = GridConfig(size=5, num_objects=3, view_radius=2, max_steps=20, **overrides)
    return SymbolicForaging(config, 64, seed=0)


def test_no_distractor_shares_both_cue_factors() -> None:
    """The cue conjunction must be necessary and sufficient to identify the goal."""

    environment = make_environment()
    state = environment.state
    target = state.target_slot[:, None]
    matches = (state.object_color == state.object_color.gather(1, target)) & (
        state.object_shape == state.object_shape.gather(1, target)
    )
    assert torch.equal(matches.sum(dim=1), torch.ones(environment.num_envs, dtype=torch.long))


def test_objects_and_agent_occupy_distinct_cells() -> None:
    environment = make_environment()
    state = environment.state
    cells = torch.cat(
        [
            (state.agent_row * environment.config.size + state.agent_col)[:, None],
            state.object_row * environment.config.size + state.object_col,
        ],
        dim=1,
    )
    for row in cells:
        assert len(set(row.tolist())) == row.numel()


def test_allowed_cues_restrict_only_the_instruction() -> None:
    """Held-out pairs never become the target but still appear as distractors."""

    holdout = latin_square_holdout(3, 3)
    config = GridConfig(size=5, num_objects=3, view_radius=2, max_steps=20)
    environment = SymbolicForaging(config, 512, seed=1, allowed_cues=train_cues(3, 3))
    cues = set(
        zip(
            environment.state.cue_color().tolist(),
            environment.state.cue_shape().tolist(),
            strict=True,
        )
    )
    assert cues.isdisjoint(holdout)

    present = set(
        zip(
            environment.state.object_color.reshape(-1).tolist(),
            environment.state.object_shape.reshape(-1).tolist(),
            strict=True,
        )
    )
    assert present & holdout, "held-out pairs must still occur as distractors"


def test_movement_clamps_at_the_border() -> None:
    environment = make_environment()
    environment.state.agent_row[:] = 0
    environment.state.agent_col[:] = 0
    environment.step(torch.full((environment.num_envs,), MOVE_NORTH))
    environment.step(torch.full((environment.num_envs,), MOVE_WEST))
    assert torch.equal(environment.state.agent_row, torch.zeros_like(environment.state.agent_row))
    assert torch.equal(environment.state.agent_col, torch.zeros_like(environment.state.agent_col))


def test_collecting_the_target_terminates_with_positive_reward() -> None:
    environment = make_environment()
    state = environment.state
    target = state.target_slot[:, None]
    state.agent_row = state.object_row.gather(1, target).squeeze(1)
    state.agent_col = state.object_col.gather(1, target).squeeze(1)
    result = environment.step(torch.full((environment.num_envs,), COLLECT))
    assert bool(result.collected_target.all())
    assert bool(result.terminated.all())
    expected = environment.config.target_reward - environment.config.step_penalty
    assert torch.allclose(result.reward, torch.full_like(result.reward, expected))


def test_collecting_a_distractor_is_penalised_without_ending_the_episode() -> None:
    environment = make_environment()
    state = environment.state
    distractor = ((state.target_slot + 1) % environment.config.num_objects)[:, None]
    state.agent_row = state.object_row.gather(1, distractor).squeeze(1)
    state.agent_col = state.object_col.gather(1, distractor).squeeze(1)
    result = environment.step(torch.full((environment.num_envs,), COLLECT))
    assert bool(result.collected_distractor.all())
    assert not bool(result.terminated.any())
    expected = environment.config.distractor_reward - environment.config.step_penalty
    assert torch.allclose(result.reward, torch.full_like(result.reward, expected))


def test_observation_marks_out_of_bounds_and_the_attended_object() -> None:
    config = GridConfig(size=5, num_objects=1, num_colors=3, num_shapes=3, view_radius=1)
    environment = SymbolicForaging(config, 1, seed=3)
    environment.state.agent_row[:] = 0
    environment.state.agent_col[:] = 0
    environment.state.object_row[:] = 0
    environment.state.object_col[:] = 1
    view = environment.observation().reshape(1, 3, 3, config.num_channels)
    # The whole first viewed row and column lie outside the grid.
    assert bool(view[0, 0, :, -1].all()) and bool(view[0, :, 0, -1].all())
    # The object sits one cell east of the agent: view row 1, column 2.
    present_channel = config.num_colors + config.num_shapes
    assert float(view[0, 1, 2, present_channel]) == 1.0
    assert float(view[0, 1, 2, int(environment.state.object_color[0, 0])]) == 1.0


def test_visible_object_slot_reports_the_nearest_object_or_none() -> None:
    config = GridConfig(size=5, num_objects=2, view_radius=1)
    environment = SymbolicForaging(config, 1, seed=4)
    environment.state.agent_row[:] = 2
    environment.state.agent_col[:] = 2
    environment.state.object_row[0] = torch.tensor([2, 4])
    environment.state.object_col[0] = torch.tensor([3, 4])
    assert int(environment.visible_object_slot()[0]) == 0
    environment.state.object_row[0] = torch.tensor([0, 4])
    environment.state.object_col[0] = torch.tensor([0, 4])
    assert int(environment.visible_object_slot()[0]) == -1


def test_oracle_reaches_and_collects_every_target() -> None:
    environment = make_environment()
    alive = torch.ones(environment.num_envs, dtype=torch.bool)
    succeeded = torch.zeros(environment.num_envs, dtype=torch.bool)
    for _ in range(environment.config.max_steps):
        result = environment.step(environment.oracle_action())
        succeeded |= result.collected_target & alive
        alive &= ~result.done
        if not bool(alive.any()):
            break
    assert bool(succeeded.all())


def test_oracle_moves_towards_the_target() -> None:
    config = GridConfig(size=5, num_objects=1, view_radius=1)
    environment = SymbolicForaging(config, 1, seed=5)
    environment.state.agent_row[:] = 0
    environment.state.agent_col[:] = 0
    environment.state.object_row[:] = 3
    environment.state.object_col[:] = 1
    assert int(environment.oracle_action()[0]) == MOVE_SOUTH
    environment.state.object_row[:] = 0
    environment.state.object_col[:] = 4
    assert int(environment.oracle_action()[0]) == MOVE_EAST
    environment.state.object_col[:] = 0
    assert int(environment.oracle_action()[0]) == COLLECT


def test_reset_done_only_resamples_finished_episodes() -> None:
    environment = make_environment()
    before = environment.state.agent_row.clone()
    done = torch.zeros(environment.num_envs, dtype=torch.bool)
    done[:8] = True
    environment.reset_done(done)
    assert torch.equal(environment.state.agent_row[8:], before[8:])
