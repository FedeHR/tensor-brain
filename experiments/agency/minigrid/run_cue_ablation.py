"""Does a trained policy actually use the instruction?

`tb-full` scoring the same as `no-cue` admits two explanations: the agent ignores
the mission, or the level does not need it. They are distinguishable without any
retraining -- evaluate a *trained* policy with the missions shuffled across the
batch, so each environment receives another environment's instruction while its
own world is unchanged. A policy that uses the instruction must get worse; a
policy that ignores it cannot.

This is the control that decides whether the gridworld's C4 result transfers.

Usage::

    python -m experiments.agency.minigrid.run_cue_ablation \
        --grid-root runs/agency/minigrid --output runs/agency/minigrid_cue_ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.agency.minigrid.conditions import CONDITIONS, LEVELS
from experiments.agency.minigrid.env import VectorMiniGrid
from experiments.agency.minigrid.ppo import EpisodeTracker
from experiments.agency.minigrid.run import build_policy, cue_split


@torch.no_grad()
def evaluate_with_cues(
    environment: VectorMiniGrid,
    policy,
    *,
    shuffle: bool,
    episodes: int = 128,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Evaluate, optionally permuting the missions across the batch."""

    device = policy.brain.A.device
    environment.reset()
    state, context = policy.initial_state(environment.num_envs, device)
    previous_reward = torch.zeros(environment.num_envs, device=device)
    tracker = EpisodeTracker(environment.num_envs, window=10_000)
    while len(tracker.finished_successes) < episodes:
        cue_color, cue_object = environment.cue_indices()
        if shuffle:
            # A derangement is not required: any permutation that mostly moves
            # missions is enough, and the residual fixed points only make the
            # measured drop conservative.
            order = torch.randperm(environment.num_envs, generator=generator)
            cue_color, cue_object = cue_color[order], cue_object[order]
        trace = policy.window_cycle(
            state,
            context,
            environment.observation().to(device),
            previous_reward,
            cue_color.to(device),
            cue_object.to(device),
        )
        result = environment.step(trace.action_position)
        tracker.update(result.reward, result.done, result.success)
        state, context = policy.reset_finished(
            trace.q, trace.context, result.done.to(device)
        )
        previous_reward = result.reward.to(device) * (~result.done).float().to(device)
    return tracker.metrics()


def run(grid_root: Path, level: str, seeds: list[int], episodes: int) -> dict:
    specification = LEVELS[level]
    train_cues, _ = cue_split(specification)
    results: dict[str, dict[str, list[float]]] = {}
    for condition in sorted(CONDITIONS):
        for seed in seeds:
            checkpoint = grid_root / level / condition / f"seed{seed}" / "checkpoint.pt"
            if not checkpoint.exists():
                continue
            policy = build_policy(condition, CONDITIONS[condition])
            policy.load_state_dict(
                torch.load(checkpoint, weights_only=True)["model_state_dict"]
            )
            policy.eval()
            entry = results.setdefault(condition, {"matched": [], "shuffled": []})
            for name, shuffle in (("matched", False), ("shuffled", True)):
                torch.manual_seed(4_000 + seed)
                environment = VectorMiniGrid(
                    specification.env_id, 16, seed=seed + 3_000, allowed_cues=train_cues
                )
                metrics = evaluate_with_cues(
                    environment, policy, shuffle=shuffle, episodes=episodes
                )
                entry[name].append(metrics["success_rate"])
                environment.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=Path("runs/agency/minigrid"))
    parser.add_argument("--level", default="gotolocal")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/agency/minigrid_cue_ablation.json")
    )
    arguments = parser.parse_args()

    results = run(arguments.grid_root, arguments.level, arguments.seeds, arguments.episodes)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2, sort_keys=True))

    print(f"{'condition':>22}  {'matched':>8} {'shuffled':>9} {'drop':>7}")
    for condition, entry in sorted(
        results.items(),
        key=lambda item: -(
            sum(item[1]["matched"]) - sum(item[1]["shuffled"])
        ) / max(1, len(item[1]["matched"])),
    ):
        matched = sum(entry["matched"]) / len(entry["matched"])
        shuffled = sum(entry["shuffled"]) / len(entry["shuffled"])
        print(
            f"{condition:>22}  {matched:8.3f} {shuffled:9.3f} {matched - shuffled:+7.3f}"
        )


if __name__ == "__main__":
    main()
