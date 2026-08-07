"""Tests for the Memory Maze filter study.

Only corpus recording needs a live maze; the filters, the decoder, the explorer,
the visibility geometry and the probe mathematics are all pure torch and are
tested wherever the suite runs.

The tests that matter most here are the ones pinning down claims the study will
make in prose: that the drift-corrected write is the raw write minus its mean,
that the correction vanishes as the softmax sharpens (which bounds what that
variant can possibly do), and that a masked step delivers no observation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from experiments.agency.memorymaze.explorer import ExplorerConfig, ScriptedExplorer
from experiments.agency.memorymaze.filter import (
    FilterConfig,
    FrameDecoder,
    RecurrentFilter,
    TensorBrainFilter,
    build_filter,
    build_filter_vocabulary,
    count_parameters,
    observation_mask,
)
from experiments.agency.memorymaze.filter_conditions import (
    CONDITIONS,
    MASK_PROBABILITIES,
    condition_config,
    grid,
)
from experiments.agency.memorymaze.filter_diagnostics import (
    index_entropy,
    log_partition_variance,
    saturated_fraction,
)
from experiments.agency.memorymaze.filter_probe import nearest_visible_color
from experiments.agency.memorymaze.horizon import (
    NEVER,
    horizon_curve,
    line_of_sight,
    steps_since_visible,
    visibility,
)
from experiments.agency.memorymaze.linear_probe import (
    classification_probe,
    mutual_information,
    regression_probe,
)


def _batch(size: int = 4, config: FilterConfig | None = None) -> tuple:
    config = config or FilterConfig()
    model = build_filter(config)
    image = torch.rand(size, 64 * 64 * 3)
    action = torch.randint(0, 6, (size,))
    observed = torch.ones(size, dtype=torch.bool)
    state, context = model.initial_state(size, torch.device("cpu"))
    return model, state, context, image, action, observed


# ------------------------------------------------------------------ explorer


def test_explorer_holds_an_action_for_its_dwell() -> None:
    """Momentum is the whole point: a redraw every step would jitter in place."""

    explorer = ScriptedExplorer(1, ExplorerConfig(dwell_mean=20.0), seed=0)
    actions = [int(explorer.act()) for _ in range(10)]
    assert len(set(actions)) == 1, "a long dwell should hold one action"


def test_explorer_eventually_uses_the_whole_action_set() -> None:
    explorer = ScriptedExplorer(16, ExplorerConfig(dwell_mean=2.0), seed=1)
    seen: set[int] = set()
    for _ in range(200):
        seen.update(explorer.act().tolist())
    assert seen == set(range(6))


def test_explorer_is_reproducible() -> None:
    def run(seed: int) -> list[int]:
        explorer = ScriptedExplorer(4, seed=seed)
        return [int(value) for _ in range(30) for value in explorer.act()]

    assert run(3) == run(3)
    assert run(3) != run(4)


def test_explorer_rejects_weights_that_do_not_cover_the_action_set() -> None:
    with pytest.raises(ValueError):
        ExplorerConfig(weights=(0.5, 0.5))


# ----------------------------------------------------------------- vocabulary


def test_filter_vocabulary_has_no_privileged_labels() -> None:
    """Colour indices in the filter's own vocabulary would make probing circular."""

    vocabulary = build_filter_vocabulary(16)
    assert set(vocabulary.groups) == {"latent", "action"}


# --------------------------------------------------------------- filter steps


@pytest.mark.parametrize("cell", ["tb", "gru", "lstm"])
def test_every_filter_returns_a_usable_trace(cell: str) -> None:
    model, state, context, image, action, observed = _batch(config=FilterConfig(cell=cell))
    trace = model.step(state, context, image, action, observed)
    assert trace.readout.shape == (4, model.state_dim)
    assert torch.isfinite(trace.readout).all()


def test_lstm_state_is_twice_as_wide_as_the_gru_state() -> None:
    assert RecurrentFilter(FilterConfig(cell="lstm")).state_width == 2 * FilterConfig().state_dim
    assert RecurrentFilter(FilterConfig(cell="gru")).state_width == FilterConfig().state_dim


def test_none_feedback_draws_no_symbol_and_soft_feedback_draws_no_outcome() -> None:
    """The native readout exists only where a symbol is actually emitted."""

    model, state, context, image, action, observed = _batch(
        config=FilterConfig(feedback="none")
    )
    trace = model.step(state, context, image, action, observed)
    assert trace.index_probabilities is None and trace.index_outcome is None

    model, state, context, image, action, observed = _batch(
        config=FilterConfig(feedback="soft")
    )
    trace = model.step(state, context, image, action, observed)
    assert trace.index_probabilities is not None
    assert trace.index_outcome is None, "attention collapses nothing, so names nothing"


def test_masked_step_delivers_no_observation() -> None:
    """With every observation withheld the encoder must not reach the state."""

    config = FilterConfig(feedback="none", action_write=False, evolution="none")
    model = build_filter(config)
    size = 3
    state, context = model.initial_state(size, torch.device("cpu"))
    image = torch.rand(size, 64 * 64 * 3)
    action = torch.zeros(size, dtype=torch.long)
    masked = model.step(state, context, image, action, torch.zeros(size, dtype=torch.bool))
    assert torch.allclose(masked.q, state), "a masked step changed the state"

    seen = model.step(state, context, image, action, torch.ones(size, dtype=torch.bool))
    assert not torch.allclose(seen.q, state)


def test_observation_mask_matches_its_probability() -> None:
    generator = torch.Generator().manual_seed(0)
    delivered = sum(
        int(observation_mask(64, 0.75, generator, torch.device("cpu")).sum())
        for _ in range(50)
    )
    assert 0.2 < delivered / (64 * 50) < 0.3
    always = observation_mask(8, 0.0, generator, torch.device("cpu"))
    assert bool(always.all())


# ---------------------------------------------------- the corrected write


def test_corrected_write_is_the_raw_write_minus_its_mean() -> None:
    """`a_k - E_p[a]`, computed by hand against `TensorBrain.measure`."""

    torch.manual_seed(0)
    brain = TensorBrainFilter(FilterConfig(num_latent_indices=8)).brain
    q = torch.randn(5, FilterConfig().state_dim)
    candidates = torch.arange(8)

    torch.manual_seed(7)
    raw, outcome, probabilities = brain.measure(q, candidates, selection="sample")
    torch.manual_seed(7)
    corrected, other, _ = brain.measure(
        q, candidates, selection="sample", drift_correction=True
    )
    assert torch.equal(outcome, other), "the same seed must draw the same index"

    expected = probabilities @ brain.A[:, candidates].T
    assert torch.allclose(corrected, raw - expected, atol=1e-6)


def test_the_correction_vanishes_as_the_softmax_sharpens() -> None:
    """The bound on what `corrected` can do, and the reason its predicted
    ordering in the original proposal cannot hold in the sharp regime.

    Scaling the embeddings up sharpens the candidate softmax; in the limit
    ``p -> delta_k`` and ``a_k - E_p[a] -> 0``, so the corrected write decays to
    no feedback at all.
    """

    torch.manual_seed(0)
    model = TensorBrainFilter(FilterConfig(num_latent_indices=8))
    q = torch.randn(64, FilterConfig().state_dim)
    candidates = torch.arange(8)

    magnitudes = []
    for scale in (1.0, 10.0, 100.0):
        with torch.no_grad():
            brain = TensorBrainFilter(FilterConfig(num_latent_indices=8)).brain
            brain.A.copy_(model.brain.A * scale)
        torch.manual_seed(1)
        corrected, _, _ = brain.measure(
            q, candidates, selection="sample", drift_correction=True
        )
        # Normalised by the embedding scale, so this compares the *shape* of the
        # write rather than its units.
        magnitudes.append(float((corrected - q).norm(dim=-1).mean()) / scale)

    assert magnitudes[0] > magnitudes[1] > magnitudes[2]
    assert magnitudes[-1] < 0.05 * magnitudes[0]


def test_alpha_zero_discards_the_prior_at_the_action_write() -> None:
    config = FilterConfig(feedback="none", evolution="none", action_retain_gate=0.0)
    model = build_filter(config)
    state = torch.randn(2, config.state_dim)
    image = torch.zeros(2, 64 * 64 * 3)
    action = torch.tensor([1, 1])
    trace = model.step(
        state, None, image, action, torch.zeros(2, dtype=torch.bool)
    )
    # Every row saw the same action and no observation, so with the prior
    # discarded the two states must coincide.
    assert torch.allclose(trace.q[0], trace.q[1], atol=1e-6)


def test_no_action_write_leaves_the_action_out_of_the_state() -> None:
    config = FilterConfig(feedback="none", evolution="none", action_write=False)
    model = build_filter(config)
    state = torch.zeros(2, config.state_dim)
    image = torch.zeros(2, 64 * 64 * 3)
    first = model.step(state, None, image, torch.tensor([0, 0]), torch.zeros(2, dtype=torch.bool))
    second = model.step(state, None, image, torch.tensor([5, 3]), torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(first.q, second.q)


# -------------------------------------------------------------------- decoder


def test_decoder_returns_a_frame_in_pixel_range() -> None:
    decoder = FrameDecoder(FilterConfig().state_dim)
    frame = decoder(torch.rand(3, FilterConfig().state_dim))
    assert frame.shape == (3, 64 * 64 * 3)
    assert float(frame.min()) >= 0.0 and float(frame.max()) <= 1.0


# ----------------------------------------------------------------- capacity


def test_the_control_is_not_handicapped_relative_to_the_tensor_brain() -> None:
    """The standing rule for this project: strong baselines at comparable scale."""

    brain = count_parameters(build_filter(FilterConfig(cell="tb")))
    control = count_parameters(build_filter(FilterConfig(cell="gru")))
    assert brain["encoder"] == control["encoder"], "the encoder must be shared"
    assert control["post_perception"] >= brain["post_perception"]


# --------------------------------------------------------------- conditions


def test_every_condition_builds_and_the_grid_is_the_advertised_size() -> None:
    for name in CONDITIONS:
        assert build_filter(condition_config(name, 0.5)) is not None
    assert len(grid()) == len(CONDITIONS) * len(MASK_PROBABILITIES)


def test_unknown_condition_fails_loudly() -> None:
    with pytest.raises(KeyError):
        condition_config("tb-imaginary", 0.0)


def test_zero_masking_is_in_the_grid_as_the_control() -> None:
    assert 0.0 in MASK_PROBABILITIES


# -------------------------------------------------------------- diagnostics


def test_entropy_is_maximal_for_a_uniform_distribution() -> None:
    uniform = torch.full((2, 8), 1 / 8)
    assert index_entropy(uniform) == pytest.approx(math.log(8), abs=1e-5)
    sharp = torch.zeros(2, 8)
    sharp[:, 0] = 1.0
    assert index_entropy(sharp) == pytest.approx(0.0, abs=1e-5)


def test_saturation_counts_units_past_the_flat_region() -> None:
    q = torch.tensor([[0.0, 5.0, -5.0, 1.0]])
    assert saturated_fraction(q) == pytest.approx(0.5)


def test_log_partition_variance_is_zero_when_the_state_is_deterministic() -> None:
    """With gamma at 0 or 1 there is nothing to sample, so log Z cannot vary."""

    q = torch.full((4, 16), 40.0)
    embeddings = torch.randn(16, 6)
    variance = log_partition_variance(q, embeddings, torch.zeros(6), samples=16)
    assert variance == pytest.approx(0.0, abs=1e-8)


# ----------------------------------------------------------------- geometry


def test_line_of_sight_is_blocked_by_a_wall() -> None:
    layout = np.ones((5, 5), dtype=np.uint8)
    assert line_of_sight(layout, np.array([0.5, 0.5]), np.array([4.5, 0.5]))
    layout[0, 2] = 0  # a wall at row 0, column 2
    assert not line_of_sight(layout, np.array([0.5, 0.5]), np.array([4.5, 0.5]))


def test_visibility_requires_the_target_to_be_in_front() -> None:
    layout = torch.ones(5, 5, dtype=torch.uint8)
    agent = torch.tensor([[2.5, 2.5]])
    targets = torch.tensor([[[4.5, 2.5], [0.5, 2.5]]])
    # Forward is component 1: the first target ahead, the second behind.
    vectors = torch.tensor([[[0.0, 2.0], [0.0, -2.0]]])
    seen = visibility(agent, targets, vectors, layout)
    assert bool(seen[0, 0]) and not bool(seen[0, 1])


def test_steps_since_visible_counts_from_the_last_sighting() -> None:
    visible = torch.tensor([[True], [False], [False], [True], [False]])
    gaps = steps_since_visible(visible)
    assert gaps.flatten().tolist() == [0, 1, 2, 0, 1]


def test_steps_since_visible_reports_never_before_the_first_sighting() -> None:
    visible = torch.tensor([[False], [False], [True]])
    gaps = steps_since_visible(visible)
    assert gaps.flatten().tolist() == [NEVER, NEVER, 0]


def test_horizon_curve_buckets_every_sample_exactly_once() -> None:
    error = torch.rand(50, 3)
    gaps = torch.randint(0, 400, (50, 3))
    curve = horizon_curve(error, gaps)
    assert sum(int(entry["samples"]) for entry in curve) == 150


def test_nearest_visible_colour_reports_nothing_when_nothing_is_in_view() -> None:
    vectors = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    nothing = nearest_visible_color(vectors, torch.zeros(1, 3, dtype=torch.bool))
    assert int(nothing[0]) == 3
    seen = nearest_visible_color(vectors, torch.tensor([[False, True, True]]))
    assert int(seen[0]) == 1, "the nearest *visible* target, not the nearest target"


# ------------------------------------------------------------------- probes


def test_probe_recovers_a_linear_ground_truth() -> None:
    torch.manual_seed(0)
    state = torch.randn(400, 12)
    weights = torch.randn(12, 2)
    result = regression_probe(state, state @ weights, state, state @ weights, penalty=1e-6)
    assert result["r2"] > 0.99


def test_probe_scores_about_zero_on_unexplainable_ground_truth() -> None:
    torch.manual_seed(0)
    train_state, test_state = torch.randn(300, 8), torch.randn(300, 8)
    result = regression_probe(
        train_state, torch.randn(300, 2), test_state, torch.randn(300, 2), penalty=10.0
    )
    assert result["r2"] < 0.1


def test_classification_probe_reports_its_own_majority_baseline() -> None:
    """The colour label is dominated by 'nothing visible', so accuracy alone lies."""

    torch.manual_seed(0)
    state = torch.randn(200, 5)
    labels = torch.zeros(200, dtype=torch.long)
    labels[:20] = 1
    result = classification_probe(state, labels, state, labels, classes=4)
    assert result["majority_baseline"] == pytest.approx(0.9)
    assert result["chance"] == pytest.approx(0.25)


def test_mutual_information_is_maximal_when_the_symbol_determines_the_label() -> None:
    labels = torch.tensor([0, 1, 2, 3] * 25)
    assert mutual_information(
        labels, labels, num_symbols=4, num_labels=4
    )["normalized"] == pytest.approx(1.0, abs=1e-6)


def test_mutual_information_is_zero_for_an_uninformative_symbol() -> None:
    labels = torch.tensor([0, 1, 2, 3] * 25)
    constant = torch.zeros(100, dtype=torch.long)
    result = mutual_information(constant, labels, num_symbols=4, num_labels=4)
    assert result["normalized"] == pytest.approx(0.0, abs=1e-6)


def test_mutual_information_is_absent_rather_than_poor_without_an_index_layer() -> None:
    """A GRU emits no symbol; the honest report is no measurement, not a low one."""

    from experiments.agency.memorymaze.filter_probe import FilterRecording, probe_all

    def recording() -> FilterRecording:
        return FilterRecording(
            state=torch.randn(60, 4),
            ground_truth={
                "targets_pos": torch.randn(60, 6),
                "agent_pos": torch.randn(60, 2),
                "target_vec": torch.randn(60, 2),
            },
            nearest_color=torch.randint(0, 4, (60,)),
            gaps=torch.randint(0, 50, (60, 3)),
            index_outcome=None,
            diagnostics={},
        )

    results = probe_all(recording(), recording(), num_latent_indices=8)
    assert results["native_readout"] is None
