"""Figure for the stage-0 diffusion probe.

    PYTHONPATH=".:src" python -m experiments.diffusion_heisenberg.figures

One figure, one claim: committing a token really does move the other positions,
the effect is strongly local, and an additive correction that depends only on the
committed token recovers almost none of it even where the effect is largest.

Print figure for a thesis, so it commits to a light surface and paints it
explicitly. Palette is the validated categorical order; every series is
direct-labeled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
ORDER = ["1", "2", "3", "4", "<=6", "<=9", ">9"]
PRETTY = {"1": "1", "2": "2", "3": "3", "4": "4", "<=6": "5–6", "<=9": "7–9", ">9": "10+"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)


def figure_verdict(payload: dict, out: Path) -> None:
    rows = payload["by_distance"]
    keys = [k for k in ORDER if k in rows]
    x = np.arange(len(keys))
    labels = [PRETTY[k] for k in keys]
    nothing = np.array([rows[k]["do_nothing"] for k in keys])
    captured = np.array([rows[k]["captured"] for k in keys]) * 100
    oracle = np.array([1 - rows[k]["oracle"] / rows[k]["do_nothing"] for k in keys]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), facecolor=SURFACE)

    ax = axes[0]
    _style(ax)
    ax.bar(x, nothing, width=0.62, color="#2a78d6", zorder=3)
    for position, value in zip(x, nothing, strict=True):
        ax.annotate(f"{value:.2f}", xy=(position, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", color="#2a78d6",
                    fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, nothing.max() * 1.2)
    ax.set_title("The interaction is real, and local", color=INK, fontsize=11,
                 loc="left", pad=10)
    ax.set_xlabel("distance from the committed position, tokens",
                  color=INK_SOFT, fontsize=9.5)
    ax.set_ylabel("KL moved by the commit, nats", color=INK_SOFT, fontsize=9.5)

    ax = axes[1]
    _style(ax)
    ax.bar(x - 0.16, captured, width=0.30, color="#eb6834", zorder=3)
    ax.bar(x + 0.16, oracle, width=0.30, color="#1baf7a", zorder=3)
    ax.axhline(0, color=INK, linewidth=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(oracle.max(), captured.max()) * 1.35)
    ax.set_title("…but an additive correction recovers almost none of it",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("distance from the committed position, tokens",
                  color=INK_SOFT, fontsize=9.5)
    ax.set_ylabel("% of the movement captured", color=INK_SOFT, fontsize=9.5)
    ax.annotate("global gain", xy=(x[0] - 0.16, captured[0]), xytext=(-2, 24),
                textcoords="offset points", ha="center", color="#eb6834",
                fontsize=9, fontweight="bold")
    ax.annotate("per-event oracle", xy=(x[0] + 0.16, oracle[0]), xytext=(10, 18),
                textcoords="offset points", ha="left", color="#1baf7a",
                fontsize=9, fontweight="bold")
    ax.annotate("an upper bound, not a method", xy=(x[1] + 0.15, oracle.max() * 1.20),
                color=INK_MUTED, fontsize=8.5)

    summary = payload["summary"]
    pairs = payload["counts"]["kl_do_nothing"]
    fig.suptitle("Stage 0: the additive correction does not transfer to diffusion decoding",
                 color=INK, fontsize=13.5, x=0.04, ha="left", y=0.985, fontweight="bold")
    fig.text(
        0.04, 0.925,
        f"{pairs:,} (commit, target) pairs from a 0.6B masked diffusion LM on GSM8K, "
        "leave-one-out throughout.\nPooled over all distances the global-gain rule "
        f"captures {payload['captured']['additive_global_gain']:.1%}; the "
        "parameter-free variant's fitted gain is exactly zero.",
        color=INK_SOFT, fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    del summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("output/diffusion_heisenberg/stage0.json"))
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    out = args.results.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figure_verdict(payload, out / "01_stage0_verdict.png")
    print(f"wrote 1 figure to {out}")


if __name__ == "__main__":
    main()
