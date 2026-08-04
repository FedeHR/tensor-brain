"""Tests for the Memory Maze integration.

Memory Maze itself needs Python 3.12 and the legacy ``gym``, which the default
development environment does not have. ``env.py`` imports ``gym`` lazily for
exactly this reason, so everything except environment construction -- the
vocabulary, the encoder, all three policies, and the probe mathematics -- is
tested here on whatever Python the suite happens to run on. The handful of tests
that need a live maze are skipped when it is absent rather than silently
dropped.
"""

from __future__ import annotations

import pytest
import torch

from experiments.agency.memorymaze.agent import (
    MemoryMazeAgent,
    PixelEncoder,
    RecurrentControl,
)
from experiments.agency.memorymaze.conditions import CONDITIONS, LEVELS, REFERENCE
from experiments.agency.memorymaze.env import (
    ACTION_NAMES,
    COLOR_NAMES,
    IMAGE_SIDE,
    build_vocabulary,
)
from experiments.agency.memorymaze.probe import (
    Recording,
    native_readout,
    probe_colors,
    probe_regression,
)


def _has_memory_maze() -> bool:
    try:
        import memory_maze  # noqa: F401
    except Exception:
        return False
    return True


needs_maze = pytest.mark.skipif(
    not _has_memory_maze(), reason="memory_maze requires Python 3.12 and legacy gym"
)


# --------------------------------------------------------------- vocabulary


def test_colour_columns_are_shared_between_instruction_and_percept() -> None:
    """The instruction and the perceptual naming must address the *same* index.

    This is the property the whole study rests on: "the maze is asking for red"
    and "the nearest thing I see is red" have to be the same symbol, or the
    native readout is not reading what the instruction wrote.
    """

    vocabulary = build_vocabulary()
    for name in COLOR_NAMES:
        assert vocabulary.index(name) in vocabulary.indices("color").tolist()
        assert vocabulary.index(name) in vocabulary.indices("percept_color").tolist()


def test_percept_colour_group_has_a_nothing_visible_option() -> None:
    vocabulary = build_vocabulary()
    assert len(vocabulary.group_labels("percept_color")) == len(COLOR_NAMES) + 1


# ------------------------------------------------------------------ encoder


def test_encoder_maps_a_flat_image_batch_into_state_coordinates() -> None:
    encoder = PixelEncoder(state_dim=32)
    flat = torch.zeros(5, IMAGE_SIDE * IMAGE_SIDE * 3)
    assert encoder(flat).shape == (5, 32)


# ----------------------------------------------------------------- policies


def test_all_three_policies_share_an_identically_sized_encoder() -> None:
    """The comparison is about what happens *after* perception."""

    policies = [
        MemoryMazeAgent(REFERENCE),
        RecurrentControl(REFERENCE, cell="gru"),
        RecurrentControl(REFERENCE, cell="lstm"),
    ]
    sizes = {sum(p.numel() for p in policy.encoder.parameters()) for policy in policies}
    assert len(sizes) == 1


def test_controls_are_not_handicapped_relative_to_the_tensor_brain() -> None:
    """Capacity comparability, as a test rather than a claim in a docstring."""

    def count(policy) -> int:
        return sum(parameter.numel() for parameter in policy.parameters())

    tensor_brain = count(MemoryMazeAgent(REFERENCE))
    for cell in ("gru", "lstm"):
        control = count(RecurrentControl(REFERENCE, cell=cell))
        assert control >= tensor_brain
        assert control < 2 * tensor_brain


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_control_window_cycle_returns_a_usable_trace(cell: str) -> None:
    policy = RecurrentControl(REFERENCE, cell=cell)
    envs = 3
    state, context = policy.initial_state(envs, torch.device("cpu"))
    cue = policy.vocabulary.indices("color")[0].repeat(envs)
    trace = policy.window_cycle(
        state,
        context,
        torch.zeros(envs, IMAGE_SIDE * IMAGE_SIDE * 3),
        torch.zeros(envs),
        cue,
        cue,
    )
    assert trace.q.shape == (envs, policy.state_width)
    assert trace.action_probabilities.shape == (envs, len(ACTION_NAMES))
    assert torch.allclose(trace.action_probabilities.sum(dim=1), torch.ones(envs))
    # A control has no index layer, so it has nothing to read natively.
    assert trace.percept_color_probabilities is None


def test_lstm_state_is_twice_as_wide_as_the_gru_state() -> None:
    gru = RecurrentControl(REFERENCE, cell="gru")
    lstm = RecurrentControl(REFERENCE, cell="lstm")
    assert lstm.state_width == 2 * gru.state_width


def test_reset_finished_zeroes_only_the_finished_environments() -> None:
    policy = RecurrentControl(REFERENCE, cell="gru")
    state = torch.ones(3, policy.state_width)
    done = torch.tensor([True, False, True])
    reset, _ = policy.reset_finished(state, None, done)
    assert float(reset[0].abs().sum()) == 0.0
    assert float(reset[2].abs().sum()) == 0.0
    assert torch.equal(reset[1], torch.ones(policy.state_width))


# --------------------------------------------------------------- conditions


def test_every_condition_is_either_a_tensor_brain_or_a_named_control() -> None:
    assert CONDITIONS["tb-full"] is not None
    assert CONDITIONS["gru-control"] is None
    assert CONDITIONS["lstm-control"] is None


def test_level_budget_is_reported_in_frames() -> None:
    level = LEVELS["9x9"]
    assert level.frames == level.updates * level.segment_steps * level.num_envs


# -------------------------------------------------------------------- probe


def _synthetic(samples: int, width: int, *, seed: int) -> Recording:
    """A recording whose ground truth is an exact linear function of the state.

    A correct probe must recover it almost perfectly; this is what separates
    "the probe works" from "the representation is good".
    """

    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(samples, width, generator=generator)
    projection = torch.randn(width, 6, generator=torch.Generator().manual_seed(0))
    labels = torch.randint(0, 4, (samples,), generator=generator)
    return Recording(
        state=state,
        ground_truth={
            "targets_pos": state @ projection,
            "agent_pos": state @ projection[:, :2],
            "target_vec": state @ projection[:, :2],
        },
        percept_color=torch.nn.functional.one_hot(labels, 4).float(),
        percept_label=labels,
    )


def test_probe_recovers_a_linear_ground_truth() -> None:
    train = _synthetic(400, 16, seed=1)
    test = _synthetic(400, 16, seed=2)
    result = probe_regression(train, test, "targets_pos", penalty=1e-6)
    assert result["r2"] > 0.99
    assert result["dims"] == 6


def test_probe_scores_zero_on_ground_truth_the_state_cannot_explain() -> None:
    """An unlearnable target must score about 0, not something impressive."""

    train = _synthetic(400, 16, seed=1)
    test = _synthetic(400, 16, seed=2)
    noise = torch.randn(len(test), 6, generator=torch.Generator().manual_seed(7))
    scrambled = Recording(
        state=test.state,
        ground_truth={**test.ground_truth, "targets_pos": noise},
        percept_color=test.percept_color,
        percept_label=test.percept_label,
    )
    result = probe_regression(train, scrambled, "targets_pos", penalty=1.0)
    assert result["r2"] < 0.1


def test_colour_probe_reports_its_own_majority_baseline() -> None:
    train = _synthetic(400, 16, seed=1)
    test = _synthetic(400, 16, seed=2)
    result = probe_colors(train, test)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["chance"] == pytest.approx(0.25)
    assert result["majority_baseline"] >= result["chance"] - 0.15


def test_native_readout_is_absent_rather_than_poor_for_a_control() -> None:
    """The structural point: a GRU has no index layer to read, not a bad one."""

    recording = _synthetic(50, 8, seed=3)
    control = Recording(
        state=recording.state,
        ground_truth=recording.ground_truth,
        percept_color=None,
        percept_label=recording.percept_label,
    )
    assert native_readout(control) is None


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_write_test_declines_a_control_rather_than_writing_into_its_action_head(
    cell: str,
) -> None:
    """A control exposes `.vocabulary` and an `.A`, so duck typing is not enough.

    Its `A` is the action head -- shape `actions x state` -- not the
    `state x indices` embedding whose column `k` *is* the symbol `k`. A
    duck-typed guard let a GRU into the write test and only failed because the
    shapes happened to disagree.
    """

    from experiments.agency.memorymaze.probe import write_test

    control = RecurrentControl(REFERENCE, cell=cell)
    assert hasattr(control, "vocabulary") and hasattr(control.brain, "A")
    assert write_test(None, control, color=COLOR_NAMES[0]) is None


def test_native_readout_needs_no_fitted_parameters() -> None:
    recording = _synthetic(50, 8, seed=3)
    result = native_readout(recording)
    assert result is not None
    assert result["probe_parameters"] == 0.0
    # The synthetic recording names the colour exactly, by construction.
    assert result["accuracy"] == pytest.approx(1.0)


# --------------------------------------------------- live environment (3.12)


def test_the_adapter_does_not_import_gym() -> None:
    """`gym` is installed only to satisfy `memory_maze`'s own import.

    The adapter talks to Memory Maze through its native `dm_env` interface, so
    no module in this package may reach for `gym`. Migrating to `gymnasium` is
    not possible -- memory_maze registers into gym's registry and subclasses
    `gym.Env` -- so the next best thing is to depend on neither.
    """

    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "experiments" / "agency" / "memorymaze"
    for source in package.glob("*.py"):
        # Parsed rather than grepped: the docstrings discuss `gym` at length,
        # and a text scan would match the explanation instead of the code.
        tree = ast.parse(source.read_text())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        offenders = [
            name for name in imported if name == "gym" or name.startswith("gym.")
        ]
        assert not offenders, f"{source.name} imports {offenders}"


@needs_maze
def test_levels_are_maze_sizes_rather_than_gym_ids() -> None:
    from experiments.agency.memorymaze.env import LEVEL_TASKS

    assert LEVELS["9x9"].env_id in LEVEL_TASKS


@needs_maze
def test_unknown_level_fails_loudly() -> None:
    from experiments.agency.memorymaze.env import VectorMemoryMaze

    with pytest.raises(KeyError):
        VectorMemoryMaze(1, level="memory_maze:MemoryMaze-9x9-ExtraObs-v0")


@needs_maze
def test_same_seed_builds_the_same_maze_and_different_seeds_do_not() -> None:
    """Seeding now goes through the task factory, not the global NumPy RNG."""

    from experiments.agency.memorymaze.env import VectorMemoryMaze

    first = VectorMemoryMaze(1, seed=3)
    second = VectorMemoryMaze(1, seed=3)
    other = VectorMemoryMaze(1, seed=4)
    try:
        layout = first._observations[0]["maze_layout"]
        assert (layout == second._observations[0]["maze_layout"]).all()
        assert (layout != other._observations[0]["maze_layout"]).any()
    finally:
        for environment in (first, second, other):
            environment.close()


@needs_maze
def test_environment_exposes_ground_truth_the_agent_never_sees() -> None:
    from experiments.agency.memorymaze.env import VectorMemoryMaze

    environment = VectorMemoryMaze(1, seed=0)
    try:
        truth = environment.ground_truth()
        assert truth["targets_pos"].shape == (1, len(COLOR_NAMES) * 2)
        assert environment.observation().shape == (1, environment.observation_dim)
    finally:
        environment.close()
