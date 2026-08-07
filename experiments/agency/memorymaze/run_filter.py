"""One cell of the filter grid: fit a filter, freeze it, probe it.

Usage::

    PYTHONPATH=src:. python -m experiments.agency.memorymaze.run_filter \
        --corpus $MEMORYMAZE_CORPUS --condition tb-raw --mask 0.5 --seed 0 \
        --output-root runs/agency/filter

No environment is constructed and nothing is rendered: this reads the recorded
corpus, so it needs neither MuJoCo nor a display, and it runs wherever a GPU is.
That is the whole reason the corpus exists.

Both phases run here rather than in separate jobs, because Phase 2 is closed
form and takes seconds -- splitting them would cost more in scheduling than it
saves. The trained weights are still written out, so a probe can be re-specified
later without refitting.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.agency.memorymaze.corpus import OfflineCorpus
from experiments.agency.memorymaze.filter import count_parameters
from experiments.agency.memorymaze.filter_conditions import (
    CONDITIONS,
    MASK_PROBABILITIES,
    condition_config,
)
from experiments.agency.memorymaze.filter_probe import probe_all, record_filter
from experiments.agency.memorymaze.filter_training import FilterModel, TrainConfig, train_filter


def run_cell(
    corpus_root: Path,
    condition: str,
    mask_probability: float,
    seed: int,
    output_root: Path,
    *,
    train_config: TrainConfig | None = None,
    # The whole split by default. Target positions are constant within an
    # episode, so the number of *episodes* -- not of steps -- is what bounds how
    # many distinct labels the probe can generalise over.
    probe_episodes: int = 100,
    device: torch.device | None = None,
    in_memory: bool = False,
) -> dict:
    """Fit and probe one condition at one masking level and one seed."""

    config = condition_config(condition, mask_probability)
    train_config = train_config or TrainConfig(seed=seed)
    directory = output_root / f"rho{mask_probability:g}" / condition / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    model = FilterModel(config)
    parameters = count_parameters(model.filter)

    train_corpus = OfflineCorpus(corpus_root, "train", in_memory=in_memory)
    started = time.time()
    print(
        f"start {condition} rho={mask_probability:g} seed{seed} "
        f"device={device} params={parameters}",
        flush=True,
    )
    log = train_filter(model, train_corpus, train_config, device=device)
    trained_seconds = time.time() - started

    # Both probe splits are held out from Phase 1, and they are different mazes
    # from each other, so the score is fit on one world and tested on another.
    probe_corpus = OfflineCorpus(corpus_root, "probe", in_memory=in_memory)
    test_corpus = OfflineCorpus(corpus_root, "test", in_memory=in_memory)
    settings = dict(mask_probability=mask_probability, seed=seed + 31_000, device=device)
    train_recording = record_filter(model, probe_corpus, episodes=probe_episodes, **settings)
    test_recording = record_filter(model, test_corpus, episodes=probe_episodes, **settings)
    results = probe_all(
        train_recording, test_recording, num_latent_indices=config.num_latent_indices
    )

    output = {
        "condition": condition,
        "mask_probability": mask_probability,
        "seed": seed,
        "parameters": parameters,
        "seconds": {"train": trained_seconds, "total": time.time() - started},
        "filter": asdict(config),
        "training": asdict(train_config),
        "probe": results,
        "log": log.as_dict(),
    }
    (directory / "result.json").write_text(json.dumps(output, indent=2))
    torch.save(model.state_dict(), directory / "filter.pt")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--mask", type=float, default=0.0, choices=MASK_PROBABILITIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/agency/filter"))
    parser.add_argument("--steps", type=int, default=None, help="override Phase-1 steps")
    parser.add_argument("--probe-episodes", type=int, default=100)
    parser.add_argument("--in-memory", action="store_true", help="load the corpus into RAM")
    arguments = parser.parse_args()

    train_config = TrainConfig(seed=arguments.seed)
    if arguments.steps is not None:
        train_config = TrainConfig(seed=arguments.seed, steps=arguments.steps)

    output = run_cell(
        arguments.corpus,
        arguments.condition,
        arguments.mask,
        arguments.seed,
        arguments.output_root,
        train_config=train_config,
        probe_episodes=arguments.probe_episodes,
        in_memory=arguments.in_memory,
    )
    probe = output["probe"]
    native = probe["native_readout"]
    colour = probe["nearest_color"]
    print(
        f"{output['condition']} rho={output['mask_probability']:g} "
        f"seed{output['seed']}: "
        f"targets_r2={probe['regression']['targets_pos']['r2']:+.3f} "
        f"agent_r2={probe['regression']['agent_pos']['r2']:+.3f} "
        # Reported as a gap: most steps have no target in view, so the majority
        # baseline sits near 0.89 and a bare accuracy says almost nothing.
        f"colour_over_majority={colour['accuracy'] - colour['majority_baseline']:+.3f} "
        + (f"native_nmi={native['normalized']:.3f} " if native else "native=none ")
        + f"({output['seconds']['total']:.0f}s)"
    )


if __name__ == "__main__":
    main()
