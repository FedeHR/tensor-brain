"""E1/E2: does the Bayes-filter reading of the measurement update buy anything?

E1 (``--noise``)
    Unreliable perception makes identifying the cued object an evidence
    accumulation problem, whose optimal solution is summing log-likelihood
    ratios -- the form of ``q <- q + a_k``. Prediction: the Tensor Brain's
    advantage over recurrent controls grows with the noise level, and is absent
    at zero noise, which is what the earlier studies measured.

E2 (``--hazard``)
    A switching target makes accumulated evidence go stale, so the optimal
    weight on the log-prior falls. Prediction: ``alpha = 1`` (HB-POVM) wins at
    zero hazard and loses to smaller ``alpha`` as hazard rises, with a learned
    ``alpha`` tracking the environment.

Usage::

    python -m experiments.agency.run_noisy --condition tb-full \
        --noise 0.5 --hazard 0.0 --seed 0 --output-root runs/agency/noisy
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiments.agency.agent import AgentConfig, GridAgent
from experiments.agency.baselines import GRUPolicy, LSTMPolicy
from experiments.agency.conditions import REFERENCE, TASK
from experiments.agency.gridworld import latin_square_holdout, train_cues
from experiments.agency.noisy import NoisyConfig, NoisyForaging
from experiments.agency.rollout import evaluate
from experiments.agency.training import ReinforceConfig, reinforce, split_evaluator

CONDITIONS: dict[str, AgentConfig | None] = {
    # --- E1: architecture comparison under unreliable perception ----------
    "tb-full": REFERENCE,
    "gru-control": None,
    "lstm-control": None,
    # Breaks the link between the column that *reads* an attribute and the
    # column that *writes* evidence for it, which is what makes the write a
    # likelihood term. Should now cost something, unlike in the noiseless study.
    "decoupled-feedback": replace(REFERENCE, decouple_feedback=True),
    "no-percept-measure": replace(REFERENCE, measure_percepts=False),
    # --- E2: the weight on the log-prior ----------------------------------
    "alpha-1.0": REFERENCE,
    "alpha-0.5": replace(REFERENCE, action_retain_gate=0.5),
    "alpha-0.0": replace(REFERENCE, action_retain_gate=0.0),
    "alpha-learned": replace(REFERENCE, learn_action_retain_gate=True),
}


def build_policy(condition: str, config: AgentConfig | None, grid) -> object:
    if config is not None:
        return GridAgent(grid, config)
    control = LSTMPolicy if condition.startswith("lstm") else GRUPolicy
    return control(grid, state_dim=REFERENCE.state_dim)


@torch.no_grad()
def calibration(environment: NoisyForaging, policy, *, batches: int = 6) -> dict[str, float]:
    """Compare the agent's readiness to commit against the exact Bayes posterior.

    At every step where the agent stands on an object, record the exact
    posterior that this object is the cued one and the agent's probability of
    choosing ``collect``. A Bayes-rational agent commits when the posterior is
    high, so the two should be positively correlated, and ``P(collect)`` should
    discriminate targets from distractors.
    """

    device = policy.brain.A.device
    posteriors, collect_probability, is_target = [], [], []
    # `collect` is the last action index in the gridworld action group.
    collect_position = len(policy.vocabulary.group_labels("action")) - 1
    for _ in range(batches):
        environment.reset()
        state, context = policy.initial_state(environment.num_envs, device)
        previous_reward = torch.zeros(environment.num_envs, device=device)
        alive = torch.ones(environment.num_envs, dtype=torch.bool, device=device)
        for step in range(environment.config.max_steps):
            grid = environment.state
            on = (grid.object_row == grid.agent_row[:, None]) & (
                grid.object_col == grid.agent_col[:, None]
            )
            standing = on.any(dim=1) & alive
            slot = on.float().argmax(dim=1)
            posterior = environment.exact_posterior().gather(1, slot[:, None]).squeeze(1)
            trace = policy.window_cycle(
                state,
                context,
                environment.observation(),
                previous_reward,
                policy.color_indices[grid.cue_color()],
                policy.shape_indices[grid.cue_shape()],
                is_first_step=torch.full(
                    (environment.num_envs,), step == 0, dtype=torch.bool, device=device
                ),
            )
            if bool(standing.any()):
                posteriors.append(posterior[standing])
                collect_probability.append(trace.action_probabilities[standing, collect_position])
                is_target.append((slot == grid.target_slot)[standing].float())
            result = environment.step(trace.action_position)
            state, context = trace.q, trace.context
            alive = alive & ~result.done
            previous_reward = result.reward
            if not bool(alive.any()):
                break
    if not posteriors:
        return {"belief_correlation": float("nan"), "commit_auc": float("nan"), "samples": 0.0}
    posterior = torch.cat(posteriors)
    probability = torch.cat(collect_probability)
    target = torch.cat(is_target)
    centred_a = posterior - posterior.mean()
    centred_b = probability - probability.mean()
    correlation = float(
        (centred_a * centred_b).sum()
        / (centred_a.norm() * centred_b.norm()).clamp_min(1e-9)
    )
    # AUC via the rank-sum identity.
    order = probability.argsort()
    ranks = torch.empty_like(probability)
    ranks[order] = torch.arange(len(probability), dtype=probability.dtype) + 1
    positives, negatives = target.sum(), (1 - target).sum()
    auc = float(
        ((ranks * target).sum() - positives * (positives + 1) / 2)
        / (positives * negatives).clamp_min(1.0)
    )
    return {
        "belief_correlation": correlation,
        "commit_auc": auc,
        "samples": float(len(posterior)),
    }


def run(
    condition: str,
    seed: int,
    output_root: Path,
    *,
    noise: float,
    hazard: float,
    updates: int,
    num_envs: int = 128,
) -> dict:
    """Train one condition at one seed under one noise/hazard setting."""

    tag = f"noise{noise:g}_hazard{hazard:g}"
    directory = output_root / tag / condition / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    noisy = NoisyConfig(observation_noise=noise, hazard_rate=hazard)
    colours, shapes = TASK.num_colors, TASK.num_shapes
    train_environment = NoisyForaging(
        TASK, num_envs, seed=seed, allowed_cues=train_cues(colours, shapes), noisy=noisy
    )
    holdout_environment = NoisyForaging(
        TASK,
        num_envs,
        seed=seed + 10_000,
        allowed_cues=latin_square_holdout(colours, shapes),
        noisy=noisy,
    )
    policy = build_policy(condition, CONDITIONS[condition], TASK)
    config = ReinforceConfig(
        updates=updates, learning_rate=3e-3, entropy_weight=0.01,
        evaluate_every=max(1, updates // 20), evaluate_repeats=2,
    )
    started = time.time()
    log = reinforce(
        train_environment, policy, config,
        evaluation=split_evaluator(policy, train_environment, holdout_environment, repeats=2),
    )
    final = evaluate(train_environment, policy, repeats=8).as_dict()
    belief = calibration(train_environment, policy)
    learned_alpha = (
        float(policy.action_retain_gate.detach())
        if getattr(policy, "action_retain_gate", None) is not None
        else None
    )
    result = {
        "condition": condition, "seed": seed, "noise": noise, "hazard": hazard,
        "seconds": round(time.time() - started, 1),
        "final": final, "calibration": belief, "learned_alpha": learned_alpha,
        "log": log.as_dict(),
    }
    (directory / "config.json").write_text(
        json.dumps(
            {"condition": condition, "seed": seed, "noisy": asdict(noisy),
             "task": asdict(TASK), "reinforce": asdict(config)},
            indent=2, sort_keys=True,
        )
    )
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    torch.save({"model_state_dict": policy.state_dict()}, directory / "checkpoint.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--hazard", type=float, default=0.0)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--output-root", type=Path, default=Path("runs/agency/noisy"))
    arguments = parser.parse_args()
    result = run(
        arguments.condition, arguments.seed, arguments.output_root,
        noise=arguments.noise, hazard=arguments.hazard, updates=arguments.updates,
    )
    alpha = result["learned_alpha"]
    print(
        f"noise={result['noise']:g} hazard={result['hazard']:g} "
        f"{result['condition']} seed{result['seed']}: "
        f"first_choice={result['final']['first_choice_accuracy']:.3f} "
        f"return={result['final']['mean_return']:+.3f} "
        f"belief_r={result['calibration']['belief_correlation']:+.3f} "
        f"auc={result['calibration']['commit_auc']:.3f}"
        + (f" alpha={alpha:.3f}" if alpha is not None else "")
        + f" ({result['seconds']:.0f}s)"
    )


if __name__ == "__main__":
    main()
