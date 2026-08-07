import math

import torch
from torch.nn import functional as F

from experiments.delayed_recall import (
    build_model,
    make_item_bank,
    run_gate,
    sample_trials,
    train_condition,
)

SMALL = {
    "num_items": 8,
    "feature_dim": 16,
    "state_dim": 32,
    "hidden_dim": 32,
    "gru_hidden_dim": 16,
}
BANK = make_item_bank(SMALL["num_items"], SMALL["feature_dim"])


def _small_model(condition: str, *, feedback_gate: float = 1.0):
    return build_model(
        condition,
        SMALL["num_items"],
        SMALL["feature_dim"],
        state_dim=SMALL["state_dim"],
        hidden_dim=SMALL["hidden_dim"],
        gru_hidden_dim=SMALL["gru_hidden_dim"],
        evolution="qtb-relu",
        score_mode="direct",
        feedback_gate=feedback_gate,
    )


def test_item_bank_has_unit_per_component_rms() -> None:
    assert BANK.shape == (8, 16)
    assert torch.allclose(BANK.norm(dim=-1), torch.full((8,), 4.0), atol=1e-5)


def test_distractors_are_never_the_cue() -> None:
    trials = sample_trials(BANK, 6, 256, generator=torch.Generator().manual_seed(0))

    assert trials.presented.shape == (256, 7)
    assert trials.views.shape == (256, 7, SMALL["feature_dim"])
    assert torch.equal(trials.presented[:, 0], trials.cue)
    assert bool((trials.presented[:, 1:] != trials.cue.unsqueeze(-1)).all())


def test_only_the_first_step_is_flagged_as_the_cue() -> None:
    trials = sample_trials(BANK, 3, 16, generator=torch.Generator().manual_seed(1))

    assert bool((trials.cue_flag[:, 0] == 1.0).all())
    assert bool((trials.cue_flag[:, 1:] == 0.0).all())


def test_noiseless_views_are_exactly_the_prototypes() -> None:
    trials = sample_trials(BANK, 3, 16, generator=torch.Generator().manual_seed(2))

    assert torch.allclose(trials.views, BANK[trials.presented])
    assert trials.probe_view is None and trials.lure is None


def test_noisy_views_differ_per_presentation_but_keep_the_input_scale() -> None:
    trials = sample_trials(
        BANK, 5, 64, view_noise=0.5, generator=torch.Generator().manual_seed(3)
    )
    expected = math.sqrt(SMALL["feature_dim"])

    assert not torch.allclose(trials.views, BANK[trials.presented])
    assert torch.allclose(
        trials.views.norm(dim=-1), torch.full((64, 6), expected), atol=1e-4
    )


def test_the_probe_is_equidistant_between_the_cue_and_its_lure() -> None:
    trials = sample_trials(
        BANK, 2, 512, recall_mode="probe", generator=torch.Generator().manual_seed(4)
    )

    assert trials.lure is not None and trials.probe_view is not None
    assert bool((trials.lure != trials.cue).all())
    to_cue = F.cosine_similarity(trials.probe_view, BANK[trials.cue], dim=-1)
    to_lure = F.cosine_similarity(trials.probe_view, BANK[trials.lure], dim=-1)
    assert torch.allclose(to_cue, to_lure, atol=1e-5)


def test_probe_mode_reports_the_two_candidate_floor() -> None:
    probe = train_condition(
        "gru", delay=0, recall_mode="probe", training_steps=1, evaluation_trials=64, **SMALL
    )
    silent = train_condition(
        "gru", delay=0, training_steps=1, evaluation_trials=64, **SMALL
    )

    assert probe.chance == 0.5
    assert silent.chance == 1.0 / SMALL["num_items"]


def test_both_models_decode_every_step_and_the_recall() -> None:
    trials = sample_trials(BANK, 3, 5, generator=torch.Generator().manual_seed(5))
    for condition in ("feedback", "gru"):
        recall_logits, identity_logits = _small_model(condition)(trials)

        assert recall_logits.shape == (5, SMALL["num_items"])
        assert identity_logits.shape == (5, 4, SMALL["num_items"])


def test_both_models_read_the_probe_when_there_is_one() -> None:
    """A probe that changed nothing would make probe mode a silent run."""

    generator = torch.Generator().manual_seed(6)
    silent = sample_trials(BANK, 2, 32, generator=generator)
    probed = sample_trials(
        BANK, 2, 32, recall_mode="probe", generator=torch.Generator().manual_seed(6)
    )
    assert torch.equal(silent.views, probed.views)

    for condition in ("feedback", "gru"):
        torch.manual_seed(11)
        model = _small_model(condition)
        recall_silent, _ = model(silent)
        recall_probed, _ = model(probed)

        assert not torch.allclose(recall_silent, recall_probed)


def test_the_ablation_removes_the_injection_and_changes_nothing_else() -> None:
    """The two Tensor Brain conditions must differ only by the injected embedding."""

    torch.manual_seed(7)
    with_feedback = _small_model("feedback")
    torch.manual_seed(7)
    without_feedback = _small_model("no-feedback")

    for left, right in zip(
        with_feedback.parameters(), without_feedback.parameters(), strict=True
    ):
        assert torch.equal(left, right)
    assert with_feedback.feedback_gate == 1.0
    assert without_feedback.feedback_gate == 0.0

    trials = sample_trials(BANK, 3, 16, generator=torch.Generator().manual_seed(8))
    recall_with, _identity_with = with_feedback(trials)
    recall_without, _identity_without = without_feedback(trials)

    assert not torch.allclose(recall_with, recall_without)


def test_the_recall_readout_is_not_the_last_identity_readout() -> None:
    """Recall must name the cue, so it cannot reuse the final item's state."""

    trials = sample_trials(BANK, 2, 16, generator=torch.Generator().manual_seed(9))
    for condition in ("feedback", "gru"):
        recall_logits, identity_logits = _small_model(condition)(trials)

        assert not torch.allclose(recall_logits, identity_logits[:, -1])


def test_the_tensor_brain_learns_delayed_recall_above_chance() -> None:
    result = train_condition(
        "feedback", delay=2, training_steps=1_500, seed=0, **SMALL
    )

    assert result.identity_accuracy > 0.95
    assert result.recall_accuracy > 4 * result.chance


def test_the_baseline_learns_delayed_recall_above_chance() -> None:
    result = train_condition("gru", delay=2, training_steps=1_500, seed=0, **SMALL)

    assert result.recall_accuracy > 4 * result.chance


def test_the_gate_aggregates_seeds_into_a_solve_rate() -> None:
    gate = run_gate("gru", delay=1, seeds=(0, 1), training_steps=800, **SMALL)

    assert gate.seeds == (0, 1)
    assert 0.0 <= gate.solve_rate <= 1.0
    assert gate.mean_recall_accuracy <= gate.best_recall_accuracy
    assert gate.chance == 1.0 / SMALL["num_items"]


def test_parameter_counts_are_comparable_at_the_default_dimensions() -> None:
    brain = train_condition(
        "feedback", delay=0, training_steps=1, evaluation_trials=8
    ).parameter_count
    gru = train_condition(
        "gru", delay=0, training_steps=1, evaluation_trials=8
    ).parameter_count

    assert abs(gru - brain) / brain < 0.1
