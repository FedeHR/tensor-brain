"""Run one condition and seed of the MiniGrid study.

Usage::

    python -m experiments.agency.minigrid.run --level gotolocal \
        --condition tb-full --seed 0 --output-root runs/agency/minigrid
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.agency.agent import AgentConfig
from experiments.agency.minigrid.agent import MiniGridAgent, RecurrentControl
from experiments.agency.minigrid.conditions import (
    CONDITIONS,
    LEVEL_CONDITIONS,
    LEVELS,
    Level,
)
from experiments.agency.minigrid.env import (
    VectorMiniGrid,
    cue_combinations,
    diagonal_holdout,
)
from experiments.agency.minigrid.ppo import PPOConfig, evaluate, train


def build_policy(
    condition: str, agent_config: AgentConfig | None
) -> MiniGridAgent | RecurrentControl:
    """Construct the Tensor Brain agent, or a control policy for ``None``."""

    if agent_config is not None:
        return MiniGridAgent(agent_config)
    cell = "lstm" if condition.startswith("lstm") else "gru"
    return RecurrentControl(CONDITIONS["tb-full"], cell=cell)


def cue_split(level: Level) -> tuple[frozenset | None, frozenset | None]:
    """Training and held-out mission sets, or ``(None, None)`` if not applicable."""

    if not level.compositional:
        return None, None
    combinations = cue_combinations(level.env_id, samples=400)
    holdout = diagonal_holdout(combinations)
    return frozenset(combinations) - holdout, holdout


def run_condition(
    level_name: str,
    condition: str,
    seed: int,
    output_root: Path,
    *,
    updates: int | None = None,
) -> dict:
    """Train one condition at one seed on one level and write its artifacts."""

    level = LEVELS[level_name]
    if condition not in LEVEL_CONDITIONS[level_name]:
        raise KeyError(f"{condition} is not defined for level {level_name}")
    directory = output_root / level_name / condition / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    train_cues, holdout_cues = cue_split(level)
    environment = VectorMiniGrid(
        level.env_id, level.num_envs, seed=seed, allowed_cues=train_cues
    )
    policy = build_policy(condition, CONDITIONS[condition])
    ppo_config = PPOConfig(
        updates=updates or level.updates,
        segment_steps=level.segment_steps,
        evaluate_every=max(1, (updates or level.updates) // 25),
    )

    evaluation = None
    if holdout_cues:
        train_eval = VectorMiniGrid(
            level.env_id, level.num_envs, seed=seed + 5_000, allowed_cues=train_cues
        )
        holdout_eval = VectorMiniGrid(
            level.env_id, level.num_envs, seed=seed + 9_000, allowed_cues=holdout_cues
        )

        def evaluation() -> dict[str, dict[str, float]]:  # noqa: F811
            return {
                "eval": evaluate(train_eval, policy, episodes=64),
                "holdout": evaluate(holdout_eval, policy, episodes=64),
            }

    started = time.time()
    log = train(environment, policy, ppo_config, evaluation=evaluation)
    result = {
        "level": level_name,
        "env_id": level.env_id,
        "condition": condition,
        "seed": seed,
        "seconds": round(time.time() - started, 1),
        "frames": ppo_config.updates * ppo_config.segment_steps * level.num_envs,
        "num_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "log": log.as_dict(),
    }
    (directory / "config.json").write_text(
        json.dumps(
            {
                "level": asdict(level),
                "condition": condition,
                "seed": seed,
                "agent": asdict(CONDITIONS[condition]) if CONDITIONS[condition] else None,
                "ppo": asdict(ppo_config),
                "train_cues": sorted(train_cues) if train_cues else None,
                "holdout_cues": sorted(holdout_cues) if holdout_cues else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    torch.save({"model_state_dict": policy.state_dict()}, directory / "checkpoint.pt")
    environment.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True, choices=sorted(LEVELS))
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/agency/minigrid"))
    parser.add_argument("--updates", type=int)
    arguments = parser.parse_args()

    result = run_condition(
        arguments.level,
        arguments.condition,
        arguments.seed,
        arguments.output_root,
        updates=arguments.updates,
    )
    final = result["log"]["train_metrics"][-1]
    print(
        f"{result['level']}/{result['condition']} seed{result['seed']}: "
        f"success={final['success_rate']:.3f} return={final['mean_return']:.3f} "
        f"({result['frames']} frames, {result['seconds']:.0f}s)"
    )


if __name__ == "__main__":
    main()
