"""Run one condition and seed of the agency ablation grid.

Usage::

    python -m experiments.agency.run_grid --condition tb-full --seed 0 \
        --output-root runs/agency/grid

Every run writes ``config.json``, ``result.json`` (the complete learning-curve
log plus the large-sample final evaluation), and ``checkpoint.pt`` so that the
qualitative figures can be regenerated without retraining.
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
from experiments.agency.conditions import CONDITIONS, REINFORCE_OVERRIDES, TASK
from experiments.agency.gridworld import (
    GridConfig,
    SymbolicForaging,
    latin_square_holdout,
    train_cues,
)
from experiments.agency.training import (
    ReinforceConfig,
    final_metrics,
    reinforce,
    split_evaluator,
)


def build_policy(
    condition: str, task: GridConfig, agent_config: AgentConfig | None, state_dim: int
) -> GridAgent | GRUPolicy:
    """Construct the Tensor Brain agent, or the control policy for ``None``."""

    if agent_config is None:
        control = LSTMPolicy if condition.startswith("lstm") else GRUPolicy
        return control(task, state_dim=state_dim)
    return GridAgent(task, agent_config)


def run_condition(
    condition: str,
    seed: int,
    output_root: Path,
    *,
    task: GridConfig = TASK,
    reinforce_config: ReinforceConfig | None = None,
    num_envs: int = 128,
) -> dict:
    """Train one condition at one seed and write its artifacts."""

    if condition not in CONDITIONS:
        raise KeyError(f"unknown condition: {condition}")
    reinforce_config = reinforce_config or ReinforceConfig(
        updates=2500,
        learning_rate=3e-3,
        entropy_weight=0.01,
        evaluate_every=100,
        evaluate_repeats=2,
    )
    reinforce_config = replace(reinforce_config, **REINFORCE_OVERRIDES.get(condition, {}))
    directory = output_root / condition / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    agent_config = CONDITIONS[condition]
    train_environment = SymbolicForaging(
        task, num_envs, seed=seed, allowed_cues=train_cues(task.num_colors, task.num_shapes)
    )
    holdout_environment = SymbolicForaging(
        task,
        num_envs,
        seed=seed + 10_000,
        allowed_cues=latin_square_holdout(task.num_colors, task.num_shapes),
    )
    policy = build_policy(condition, task, agent_config, state_dim=64)

    started = time.time()
    log = reinforce(
        train_environment,
        policy,
        reinforce_config,
        evaluation=split_evaluator(
            policy,
            train_environment,
            holdout_environment,
            repeats=reinforce_config.evaluate_repeats,
        ),
    )
    finals = final_metrics(policy, train_environment, holdout_environment, repeats=16)
    result = {
        "condition": condition,
        "seed": seed,
        "seconds": round(time.time() - started, 1),
        "num_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "log": log.as_dict(),
        "final": {split: metrics.as_dict() for split, metrics in finals.items()},
    }
    (directory / "config.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "seed": seed,
                "num_envs": num_envs,
                "task": asdict(task),
                "agent": asdict(agent_config) if agent_config is not None else None,
                "reinforce": asdict(reinforce_config),
            },
            indent=2,
            sort_keys=True,
        )
    )
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    torch.save({"model_state_dict": policy.state_dict()}, directory / "checkpoint.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/agency/grid"))
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--num-envs", type=int, default=128)
    arguments = parser.parse_args()

    result = run_condition(
        arguments.condition,
        arguments.seed,
        arguments.output_root,
        reinforce_config=ReinforceConfig(
            updates=arguments.updates,
            learning_rate=3e-3,
            entropy_weight=0.01,
            evaluate_every=100,
            evaluate_repeats=2,
        ),
        num_envs=arguments.num_envs,
    )
    print(
        f"{result['condition']} seed{result['seed']}: "
        f"eval={result['final']['eval']['success_rate']:.3f} "
        f"holdout={result['final']['holdout']['success_rate']:.3f} "
        f"({result['seconds']:.0f}s)"
    )


if __name__ == "__main__":
    main()
