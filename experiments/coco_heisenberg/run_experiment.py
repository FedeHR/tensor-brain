"""Run the COCO learned-index-layer experiment.

    TB_BAYES_ROOT=../tensor-brain-bayes-approximation \
    PYTHONPATH=".:src" python -m experiments.coco_heisenberg.run_experiment \
        --corpus data/coco/corpus_train2017_k1000.npz --output output/coco_heisenberg

Reports downstream decision quality first and posterior fidelity second, because
the scoping work showed the two can rank rules differently and the decision is
what a consumer of the state experiences.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import evaluation as E
from . import model as Mo

DOWNSTREAM = ("mean_ap", "macro_f1", "micro_f1", "accuracy", "exact_set", "nll", "ece")
OPERATING = ("macro_precision", "macro_recall", "predicted_positive_rate", "true_positive_rate")
FIDELITY = ("joint_kl", "marginal_kl")


def _revision(root: Path) -> dict[str, str]:
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    return {"revision": head}


def _print_table(title: str, rows: dict[str, dict[str, float]], columns: tuple[str, ...]) -> None:
    present = [c for c in columns if any(c in row for row in rows.values())]
    print(f"\n{title}")
    header = f"{'rule':<24}" + "".join(f"{c:>12}" for c in present)
    print(header)
    print("-" * len(header))
    for name, row in rows.items():
        line = f"{name:<24}"
        for c in present:
            line += f"{row[c]:>12.4f}" if c in row else f"{'':>12}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/coco_heisenberg"))
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--symbol-counts", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--limit", type=int, default=4000, help="evaluation images per point")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-cross-check", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    corpus = D.load(args.corpus)
    train_index, test_index = D.split_indices(
        corpus.num_images, holdout=args.holdout, seed=args.seed
    )
    train, test = corpus.subset(train_index), corpus.subset(test_index)
    print(f"train {train.num_images}  test {test.num_images}  "
          f"n={corpus.num_categories}  K={corpus.num_symbols}")

    started = time.time()
    layer = Mo.fit_index_layer(
        train, steps=args.steps, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, seed=args.seed,
    )
    fit_seconds = time.time() - started

    symbol_nll = Mo.held_out_nll(layer, test)
    uniform_nll = Mo.uniform_symbol_nll(test)
    pre = E.precompute(layer)
    stats = E.log_partition_stats(pre)

    print(f"\nfitted in {fit_seconds:.1f}s")
    print(f"held-out symbol NLL {symbol_nll:.4f}  (uniform {uniform_nll:.4f}, "
          f"information gained {uniform_nll - symbol_nll:.4f} nats)")
    print(f"mean column norm {float(layer.A.norm(dim=0).mean()):.4f}")
    print("log-partition: " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))

    checks = {}
    if not args.skip_cross_check:
        checks = E.cross_check(layer, [int(k) for k in test.symbols[0][:4]])
        print("cross-check vs reference: max abs diff "
              f"{max(checks.values()):.2e} over {len(checks)} rules")

    # Contrasts that decide the chapter's claims, each paired image by image.
    CONTRASTS = (
        ("heisenberg", "exact"),
        ("heisenberg-gauge", "heisenberg"),
        ("heisenberg-pe", "heisenberg"),
        ("heisenberg", "exact-empirical-prior"),
    )

    sweep: dict[str, dict[str, dict[str, float]]] = {}
    paired: dict[str, dict[str, dict[str, float]]] = {}
    posterior_variance: dict[str, float] = {}
    for count in args.symbol_counts:
        started = time.time()
        rows, per_image = E.evaluate(
            layer, test, num_symbols=count, seed=args.seed, limit=args.limit, pre=pre,
            return_per_image=True,
        )
        sweep[str(count)] = rows
        _print_table(
            f"M = {count} symbols absorbed   (multi-label recognition of the 12 supercategories)",
            rows, DOWNSTREAM,
        )
        _print_table(f"M = {count}   (where the threshold sits)", rows, OPERATING)
        _print_table(f"M = {count}   (fidelity to the exact posterior, nats)", rows, FIDELITY)

        # the error law wants the posterior-weighted variance, not the prior one
        posterior_variance[str(count)] = E.posterior_partition_variance(
            pre, test, num_symbols=count, seed=args.seed, limit=min(args.limit, 2000)
        )

        contrasts = {}
        print(f"\nM = {count}   paired per-image differences (negative favours the first rule)")
        print(f"{'contrast':<42}{'dNLL':>10}{'95% CI':>20}{'wins':>8}")
        for left, right in CONTRASTS:
            if left not in per_image or right not in per_image:
                continue
            stat = E.paired_difference(per_image, left, right, seed=args.seed)
            contrasts[f"{left} - {right}"] = stat
            ci = f"[{stat['ci_low']:+.4f},{stat['ci_high']:+.4f}]"
            print(f"{left + ' - ' + right:<42}{stat['mean']:>+10.4f}{ci:>20}"
                  f"{stat['left_better_fraction']:>8.2f}")
        paired[str(count)] = contrasts
        print(f"[{time.time() - started:.1f}s]")

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "corpus": str(args.corpus),
            "holdout": args.holdout,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "symbol_counts": args.symbol_counts,
            "limit": args.limit,
            "seed": args.seed,
            "num_categories": corpus.num_categories,
            "num_symbols": corpus.num_symbols,
            "train_images": train.num_images,
            "test_images": test.num_images,
        },
        "fit": {
            "seconds": fit_seconds,
            "symbol_nll": symbol_nll,
            "uniform_symbol_nll": uniform_nll,
            "mean_column_norm": float(layer.A.norm(dim=0).mean()),
            "history": layer.history,
        },
        "log_partition": stats,
        "cross_check": checks,
        "sweep": sweep,
        "paired": paired,
        "posterior_partition_variance": posterior_variance,
        "categories": corpus.categories,
        "source": _revision(Path(__file__).resolve().parents[2]),
    }
    destination = args.output / "results.json"
    destination.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        args.output / "index_layer.npz",
        A=layer.A.numpy(), a0=layer.a0.numpy(),
        q_prior=layer.q_prior.numpy(), joint_prior=layer.joint_prior.numpy(),
    )
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
