import torch

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


def _small_model(condition: str, *, feedback_gate: float = 1.0):
    return build_model(
        condition,
        make_item_bank(SMALL["num_items"], SMALL["feature_dim"]),
        state_dim=SMALL["state_dim"],
        hidden_dim=SMALL["hidden_dim"],
        gru_hidden_dim=SMALL["gru_hidden_dim"],
        evolution="qtb-relu",
        score_mode="direct",
        feedback_gate=feedback_gate,
    )


def test_item_bank_has_unit_per_component_rms() -> None:
    bank = make_item_bank(8, 16)

    assert bank.shape == (8, 16)
    assert torch.allclose(bank.norm(dim=-1), torch.full((8,), 4.0), atol=1e-5)


def test_distractors_are_never_the_cue() -> None:
    trials = sample_trials(4, 6, 256, generator=torch.Generator().manual_seed(0))

    assert trials.presented.shape == (256, 7)
    assert torch.equal(trials.presented[:, 0], trials.cue)
    assert bool((trials.presented[:, 1:] != trials.cue.unsqueeze(-1)).all())


def test_only_the_first_step_is_flagged_as_the_cue() -> None:
    trials = sample_trials(8, 3, 16, generator=torch.Generator().manual_seed(1))

    assert bool((trials.cue_flag[:, 0] == 1.0).all())
    assert bool((trials.cue_flag[:, 1:] == 0.0).all())


def test_zero_delay_presents_only_the_cue() -> None:
    trials = sample_trials(8, 0, 16, generator=torch.Generator().manual_seed(2))

    assert trials.presented.shape == (16, 1)


def test_both_models_decode_every_step_and_the_recall() -> None:
    trials = sample_trials(
        SMALL["num_items"], 3, 5, generator=torch.Generator().manual_seed(3)
    )
    for condition in ("feedback", "gru"):
        recall_logits, identity_logits = _small_model(condition)(trials)

        assert recall_logits.shape == (5, SMALL["num_items"])
        assert identity_logits.shape == (5, 4, SMALL["num_items"])


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

    trials = sample_trials(
        SMALL["num_items"], 3, 16, generator=torch.Generator().manual_seed(8)
    )
    recall_with, _identity_with = with_feedback(trials)
    recall_without, _identity_without = without_feedback(trials)

    assert not torch.allclose(recall_with, recall_without)


def test_the_recall_readout_is_not_the_last_identity_readout() -> None:
    """Recall must name the cue, so it cannot reuse the final item's state."""

    trials = sample_trials(
        SMALL["num_items"], 2, 16, generator=torch.Generator().manual_seed(9)
    )
    for condition in ("feedback", "gru"):
        recall_logits, identity_logits = _small_model(condition)(trials)

        assert not torch.allclose(recall_logits, identity_logits[:, -1])


def test_the_tensor_brain_learns_delayed_recall_above_chance() -> None:
    result = train_condition(
        "feedback",
        delay=2,
        evolution="qtb-relu",
        training_steps=1_500,
        seed=0,
        **SMALL,
    )

    assert result.identity_accuracy > 0.95
    assert result.recall_accuracy > 4 * result.chance


def test_the_baseline_learns_delayed_recall_above_chance() -> None:
    result = train_condition(
        "gru", delay=2, training_steps=1_500, seed=0, **SMALL
    )

    assert result.recall_accuracy > 4 * result.chance


def test_the_gate_aggregates_seeds_into_a_solve_rate() -> None:
    gate = run_gate(
        "gru",
        delay=1,
        seeds=(0, 1),
        training_steps=800,
        **SMALL,
    )

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
