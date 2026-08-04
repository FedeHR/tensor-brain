"""Tests for the noisy-perception / volatile-target environment."""

import math

import pytest
import torch

from experiments.agency.gridworld import GridConfig, SymbolicForaging
from experiments.agency.noisy import NoisyConfig, NoisyForaging

TASK = GridConfig(size=5, num_objects=3, view_radius=2, max_steps=20)


def make(noise: float = 0.0, hazard: float = 0.0, num_envs: int = 256, seed: int = 0):
    return NoisyForaging(
        TASK, num_envs, seed=seed, noisy=NoisyConfig(noise, hazard)
    )


def test_noiseless_is_identical_to_the_plain_environment() -> None:
    """The added machinery must not perturb the original task at eps = h = 0."""

    torch.manual_seed(0)
    noisy = NoisyForaging(TASK, 8, seed=3, noisy=NoisyConfig(0.0, 0.0))
    torch.manual_seed(0)
    plain = SymbolicForaging(TASK, 8, seed=3)
    assert torch.equal(noisy.observation(), plain.observation())
    assert torch.equal(noisy.state.target_slot, plain.state.target_slot)


def test_reading_disagreement_matches_the_noise_model() -> None:
    """A reading should disagree with the truth at rate eps * (1 - 1/K)."""

    environment = make(noise=0.5, num_envs=512)
    disagreements = []
    for _ in range(30):
        reported, _ = environment.observed_attributes()
        disagreements.append(float((reported != environment.state.object_color).float().mean()))
        environment.observation()
        environment.step(torch.randint(0, 5, (512,)))
    expected = 0.5 * (1.0 - 1.0 / TASK.num_colors)
    assert abs(sum(disagreements) / len(disagreements) - expected) < 0.03


def test_the_renderer_sees_the_same_reading_that_was_recorded() -> None:
    """Evidence counts must describe the view the agent actually received."""

    environment = make(noise=0.9, num_envs=64)
    before = environment.color_counts.clone()
    view = environment.observation().reshape(64, TASK.view_side, TASK.view_side, -1)
    delta = environment.color_counts - before
    # Every recorded reading belongs to an object inside the view window.
    assert float(delta.sum()) == pytest.approx(float(view[..., -2].sum()))


def test_posterior_is_normalised_and_informative() -> None:
    environment = make(noise=0.4, num_envs=512)
    for _ in range(25):
        environment.observation()
        environment.step(torch.randint(0, 4, (512,)))
    posterior = environment.exact_posterior()
    assert torch.allclose(posterior.sum(-1), torch.ones(512), atol=1e-5)
    on_target = posterior.gather(1, environment.state.target_slot[:, None]).mean()
    assert float(on_target) > 1.0 / TASK.num_objects + 0.05


def test_posterior_matches_a_direct_likelihood_ratio_computation() -> None:
    """The vectorized posterior must equal a per-object log-likelihood ratio."""

    environment = make(noise=0.3, num_envs=16, seed=5)
    for _ in range(12):
        environment.observation()
        environment.step(torch.randint(0, 4, (16,)))
    epsilon, colours = 0.3, TASK.num_colors
    shapes = TASK.num_shapes
    cue_color = environment.state.cue_color()
    cue_shape = environment.state.cue_shape()
    expected = torch.zeros(16, TASK.num_objects)
    for env in range(16):
        for slot in range(TASK.num_objects):
            total = 0.0
            for counts, cue, values in (
                (environment.color_counts[env, slot], int(cue_color[env]), colours),
                (environment.shape_counts[env, slot], int(cue_shape[env]), shapes),
            ):
                hit = (1.0 - epsilon) + epsilon / values
                miss = epsilon / values
                other = values - 1
                for value in range(values):
                    positive = hit if value == cue else miss
                    negative = (
                        miss if value == cue else (hit + (other - 1) * miss) / other
                    )
                    total += float(counts[value]) * (
                        math.log(positive) - math.log(negative)
                    )
            expected[env, slot] = total
    assert torch.allclose(
        environment.exact_posterior(), torch.softmax(expected, dim=-1), atol=1e-4
    )


def test_zero_noise_posterior_identifies_an_observed_target() -> None:
    """With a perfect sensor, one look at the target settles the question.

    The condition is that the *target* has been observed. If it has not, a
    distractor that matches the cue on one factor legitimately carries more
    evidence than an object never seen, and preferring it is correct Bayesian
    behaviour rather than a defect.
    """

    environment = make(noise=0.0, num_envs=128, seed=7)
    for _ in range(20):
        environment.observation()
        environment.step(torch.randint(0, 4, (128,)))
    target = environment.state.target_slot
    rows = torch.arange(128)
    observed = environment.color_counts[rows, target].sum(-1) > 0
    assert bool(observed.any())
    chosen = environment.exact_posterior().argmax(dim=-1)
    assert torch.equal(chosen[observed], target[observed])


def test_hazard_switches_the_target_and_clears_stale_evidence() -> None:
    environment = make(hazard=1.0, num_envs=64, seed=9)
    environment.observation()
    assert float(environment.color_counts.sum()) > 0.0
    before = environment.state.target_slot.clone()
    environment.step(torch.zeros(64, dtype=torch.long))
    assert not bool((environment.state.target_slot == before).all())
    assert float(environment.color_counts.sum()) == 0.0


def test_zero_hazard_keeps_the_target_fixed() -> None:
    environment = make(hazard=0.0, num_envs=64, seed=11)
    before = environment.state.target_slot.clone()
    alive = torch.ones(64, dtype=torch.bool)
    for _ in range(5):
        result = environment.step(torch.zeros(64, dtype=torch.long))
        alive &= ~result.done
    assert torch.equal(environment.state.target_slot[alive], before[alive])


def test_learned_retain_gate_is_a_trained_parameter() -> None:
    """`alpha` must actually receive gradient when it is made learnable."""

    from experiments.agency.agent import AgentConfig, GridAgent
    from experiments.agency.rollout import run_episodes
    from experiments.agency.training import _masked_mean

    torch.manual_seed(0)
    agent = GridAgent(
        TASK, AgentConfig(state_dim=16, hidden_dim=16, learn_action_retain_gate=True)
    )
    assert agent.action_retain_gate is not None
    environment = make(noise=0.3, num_envs=8)
    batch = run_episodes(environment, agent)
    loss = -_masked_mean(batch.action_log_probability, batch.alive.float())
    loss.backward()
    assert agent.action_retain_gate.grad is not None
    assert float(agent.action_retain_gate.grad.abs()) > 0.0


def test_unobserved_objects_carry_neutral_evidence() -> None:
    """Never having looked at an object must not make it the favourite.

    A bare likelihood scores an unobserved object at zero, which beats any
    accumulated negative log-likelihood; the posterior must use a ratio so that
    no evidence leaves an object at the prior.
    """

    environment = make(noise=0.3, num_envs=32, seed=13)
    # Give one object plenty of readings that agree with the cue and leave the
    # rest unobserved.
    environment.color_counts.zero_()
    environment.shape_counts.zero_()
    target = environment.state.target_slot
    rows = torch.arange(32)
    environment.color_counts[rows, target, environment.state.cue_color()] = 20.0
    environment.shape_counts[rows, target, environment.state.cue_shape()] = 20.0
    posterior = environment.exact_posterior()
    assert torch.equal(posterior.argmax(dim=-1), target)
    assert float(posterior.gather(1, target[:, None]).mean()) > 0.95


def test_contrary_evidence_pushes_an_object_below_the_prior() -> None:
    """Readings that disagree with the cue must count against an object."""

    environment = make(noise=0.3, num_envs=16, seed=17)
    environment.color_counts.zero_()
    environment.shape_counts.zero_()
    wrong = (environment.state.cue_color() + 1) % TASK.num_colors
    environment.color_counts[torch.arange(16), 0, wrong] = 15.0
    posterior = environment.exact_posterior()
    assert float(posterior[:, 0].mean()) < 1.0 / TASK.num_objects
