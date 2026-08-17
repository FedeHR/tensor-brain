"""Figures for the COCO learned-index-layer experiment.

    PYTHONPATH=".:src" python -m experiments.coco_heisenberg.figures \
        --results output/coco_heisenberg/results.json

Four figures, one claim each:

    01_dissociation   fidelity and downstream quality rank the rules differently
    02_contrasts      the paired differences, with bootstrap CIs
    03_error_law      the M^2 law, measured against its own prediction
    04_error_budget   what is left is mostly the prior, not the update rule

These are print figures for a thesis, so they commit to a light surface and paint
it explicitly rather than inheriting one. Palette is the validated categorical
order (blue, orange, aqua, violet); every series is direct-labeled, which is also
the required relief for aqua's contrast against a light surface.
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

SERIES = {
    "heisenberg": "#2a78d6",
    "heisenberg-gauge": "#eb6834",
    "heisenberg-pe": "#1baf7a",
    "adf": "#4a3aa7",
}
LABEL = {
    "heisenberg": "Heisenberg",
    "heisenberg-gauge": "Heisenberg + gauge fix",
    "heisenberg-pe": "Heisenberg + prediction error",
    "adf": "ADF",
    "exact": "exact Bayes",
    "prior": "prior",
    "exact-empirical-prior": "exact Bayes, empirical joint prior",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)


def _label_end(ax, x, y, text, color, *, dx=0.12, va="center", size=9, weight="normal"):
    ax.annotate(
        text, xy=(x, y), xytext=(x + dx, y), textcoords="data",
        color=color, fontsize=size, va=va, ha="left", fontweight=weight,
        annotation_clip=False,
    )


def _label_column(ax, x, entries, *, dx=0.16, gap_frac=0.062, size=9, log=False):
    """Direct-label several series at their line ends without letting text collide.

    Labels are pushed apart to a minimum vertical gap and joined to their true
    endpoint by a faint leader, so identity never depends on colour alone -- which
    is also the relief the palette's contrast warning requires.
    """

    lo, hi = ax.get_ylim()
    if log:  # push apart in the space the eye actually sees
        lo, hi = np.log10(lo), np.log10(hi)
    gap = (hi - lo) * gap_frac
    entries = sorted(entries, key=lambda e: e[0])
    ys = [np.log10(float(e[0])) if log else float(e[0]) for e in entries]
    for _ in range(400):
        moved = False
        for i in range(1, len(ys)):
            overlap = gap - (ys[i] - ys[i - 1])
            if overlap > 1e-12:
                ys[i - 1] -= overlap / 2
                ys[i] += overlap / 2
                moved = True
        if not moved:
            break
    for (y_true, text, color, weight), y in zip(entries, ys, strict=True):
        target_x = x * dx if log else x + dx
        target_y = 10**y if log else y
        ax.annotate(
            text, xy=(x, y_true), xytext=(target_x, target_y), textcoords="data",
            color=color, fontsize=size, va="center", ha="left", fontweight=weight,
            annotation_clip=False,
            # a surface-coloured plate keeps a rule line from striking through its label
            bbox={"boxstyle": "round,pad=0.18", "facecolor": SURFACE,
                  "edgecolor": "none"},
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7,
                        "alpha": 0.45, "shrinkA": 0, "shrinkB": 2},
        )


def figure_dissociation(payload: dict, out: Path) -> None:
    """Downstream quality and posterior fidelity order the rules differently."""

    counts = payload["config"]["symbol_counts"]
    sweep = payload["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), facecolor=SURFACE)

    order = ["heisenberg", "heisenberg-gauge", "heisenberg-pe", "adf"]

    # Both panels are measured against exact Bayes, which is the zero line in each.
    # Plotting absolute NLL would hide the effect: the spread between rules is a
    # few hundredths of a nat against ~4 nats of absolute NLL.
    panels = (
        (axes[0], "marginal_kl", "Fidelity: KL to the exact posterior of the same model",
         "KL from exact Bayes, nats", False),
        (axes[1], "nll", "Downstream: NLL of the true supercategory set",
         "excess NLL vs exact Bayes, nats", True),
    )
    for ax, metric, title, ylabel, relative in panels:
        _style(ax)
        entries = []
        for rule in order:
            base = [sweep[str(m)]["exact"][metric] for m in counts] if relative else 0
            y = np.array([sweep[str(m)][rule][metric] for m in counts])
            if relative:
                y = y - np.array(base)
            ax.plot(counts, y, color=SERIES[rule], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            entries.append((y[-1], LABEL[rule], SERIES[rule], "normal"))
        ax.axhline(0, color=INK, linewidth=1.5, zorder=2)
        entries.append((0.0, "exact Bayes", INK, "bold"))
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("symbols absorbed  (M)", color=INK_SOFT, fontsize=9.5)
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9.5)
        ax.set_xticks(counts)
        ax.set_xlim(counts[0] - 0.3, counts[-1] + 4.6)
        margin = 0.18 * (max(e[0] for e in entries) - min(e[0] for e in entries))
        ax.set_ylim(min(e[0] for e in entries) - margin, max(e[0] for e in entries) + margin)
        _label_column(ax, counts[-1], entries)

    lo, hi = axes[1].get_ylim()
    axes[1].annotate("better than\nexact Bayes", xy=(counts[0] - 0.1, lo + (hi - lo) * 0.10),
                     color=INK_MUTED, fontsize=8.5, va="center")

    fig.suptitle(
        "The two things you might measure disagree",
        color=INK, fontsize=13.5, x=0.045, ha="left", y=0.985, fontweight="bold",
    )
    fig.text(
        0.045, 0.925,
        "Both panels are measured against exact Bayes, the black line at zero.\n"
        "Up to M≈5 the Heisenberg belief is the furthest of the four from the exact "
        "posterior — and predicts the true categories better than exact Bayes does.",
        color=INK_SOFT, fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_contrasts(payload: dict, out: Path) -> None:
    """Paired per-image differences with bootstrap confidence intervals."""

    counts = payload["config"]["symbol_counts"]
    paired = payload["paired"]
    contrasts = [
        ("heisenberg - exact", "#2a78d6", "Heisenberg  −  exact Bayes"),
        ("heisenberg-gauge - heisenberg", "#eb6834", "gauge fix  −  Heisenberg"),
        ("heisenberg-pe - heisenberg", "#1baf7a", "prediction error  −  Heisenberg"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.0), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK, linewidth=1.4, zorder=2)

    entries = []
    offsets = np.linspace(-0.13, 0.13, len(contrasts))
    for (key, color, label), shift in zip(contrasts, offsets, strict=True):
        xs = np.asarray(counts, dtype=float) * (1 + shift * 0.14)
        mean = np.array([paired[str(m)][key]["mean"] for m in counts])
        low = np.array([paired[str(m)][key]["ci_low"] for m in counts])
        high = np.array([paired[str(m)][key]["ci_high"] for m in counts])
        ax.errorbar(
            xs, mean, yerr=[mean - low, high - mean], color=color, linewidth=2,
            marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
            capsize=3, elinewidth=1.6, zorder=3,
        )
        entries.append((mean[-1], label, color, "normal"))

    ax.set_xticks(counts)
    ax.set_xlim(counts[0] - 0.4, counts[-1] + 5.4)
    ax.set_xlabel("symbols absorbed  (M)", color=INK_SOFT, fontsize=9.5)
    ax.set_ylabel("paired difference in NLL, nats", color=INK_SOFT, fontsize=9.5)
    _label_column(ax, counts[-1], entries)
    lo, hi = ax.get_ylim()
    ax.annotate("first rule better", xy=(counts[0] - 0.2, lo + (hi - lo) * 0.16),
                color=INK_MUTED, fontsize=8.5, rotation=90, va="center")

    fig.suptitle(
        "Every comparison, paired image by image",
        color=INK, fontsize=13.5, x=0.045, ha="left", y=0.985, fontweight="bold",
    )
    fig.text(
        0.045, 0.925,
        "Bars are 95% bootstrap intervals over held-out images; all exclude zero.\n"
        "Heisenberg beats exact Bayes until M≈5. The gauge fix wins everywhere.",
        color=INK_SOFT, fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_error_law(payload: dict, out: Path) -> None:
    """The measured error against the M^2 law that predicts it."""

    counts = np.asarray(payload["config"]["symbol_counts"], dtype=float)
    measured = np.array(
        [payload["sweep"][str(int(m))]["heisenberg"]["marginal_kl"] for m in counts]
    )
    prior_var = payload["log_partition"]["var_log_partition"]
    posterior_var = payload.get("posterior_partition_variance", {})

    fig, ax = plt.subplots(figsize=(9.6, 5.4), facecolor=SURFACE)
    _style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")

    entries = []
    ax.plot(counts, 0.5 * counts**2 * prior_var, color="#1baf7a", linewidth=2,
            linestyle=(0, (1, 2)), zorder=2)
    entries.append((0.5 * counts[-1] ** 2 * prior_var,
                    "½M²·Var$_{prior}$[log Z]", "#1baf7a", "normal"))

    if posterior_var:
        pv = np.array([posterior_var[str(int(m))] for m in counts])
        ax.plot(counts, 0.5 * counts**2 * pv, color="#eb6834", linewidth=2,
                linestyle=(0, (5, 2)), zorder=2)
        entries.append((0.5 * counts[-1] ** 2 * pv[-1],
                        "½M²·Var$_{posterior}$[log Z]", "#eb6834", "normal"))

    ax.plot(counts, measured, color="#2a78d6", linewidth=2.4, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    entries.append((measured[-1], "measured", "#2a78d6", "bold"))

    slope_all = float(np.polyfit(np.log(counts), np.log(measured), 1)[0])
    low = counts <= 4
    slope_low = float(np.polyfit(np.log(counts[low]), np.log(measured[low]), 1)[0])
    ax.annotate(
        f"fitted exponent\n{slope_low:.2f} over M ≤ 4\n{slope_all:.2f} over the full range",
        xy=(1.06, measured[-1] * 0.62), color=INK, fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": SURFACE,
              "edgecolor": GRID, "linewidth": 1},
    )

    ax.set_xticks(counts)
    ax.set_xticklabels([str(int(m)) for m in counts])
    ax.set_xlim(counts[0] * 0.92, counts[-1] * 3.4)
    ax.set_xlabel("symbols absorbed  (M),  log scale", color=INK_SOFT, fontsize=9.5)
    ax.set_ylabel("marginal KL to the exact posterior, nats", color=INK_SOFT, fontsize=9.5)
    _label_column(ax, counts[-1], entries, dx=1.09, gap_frac=0.075, log=True)

    fig.suptitle(
        "The M² error law survives a learned index layer",
        color=INK, fontsize=13.5, x=0.045, ha="left", y=0.985, fontweight="bold",
    )
    fig.text(
        0.045, 0.925,
        "Derived on synthetic Gaussian A, tested here on A fitted to COCO captions.\n"
        "The law needs the posterior-weighted variance; the prior-weighted one over-predicts.",
        color=INK_SOFT, fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_error_budget(payload: dict, out: Path) -> None:
    """How the remaining downstream error splits between rule and prior."""

    counts = payload["config"]["symbol_counts"]
    sweep = payload["sweep"]
    rule_cost = np.array(
        [sweep[str(m)]["heisenberg"]["nll"] - sweep[str(m)]["exact"]["nll"] for m in counts]
    )
    prior_cost = np.array(
        [sweep[str(m)]["exact"]["nll"] - sweep[str(m)]["exact-empirical-prior"]["nll"]
         for m in counts]
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.0), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK, linewidth=1.4, zorder=2)

    for values, color in ((prior_cost, "#eb6834"), (rule_cost, "#2a78d6")):
        ax.plot(counts, values, color=color, linewidth=2.4, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)

    ax.set_xticks(counts)
    ax.set_xlim(counts[0] - 0.3, counts[-1] + 5.2)
    _label_column(ax, counts[-1], [
        (prior_cost[-1], "cost of the factorized prior", "#eb6834", "normal"),
        (rule_cost[-1], "cost of the update rule", "#2a78d6", "normal"),
    ])
    ax.set_xlabel("symbols absorbed  (M)", color=INK_SOFT, fontsize=9.5)
    ax.set_ylabel("excess NLL against the best reachable belief, nats",
                  color=INK_SOFT, fontsize=9.5)

    fig.suptitle(
        "Most of what is left is the prior, not the update rule",
        color=INK, fontsize=13.5, x=0.045, ha="left", y=0.985, fontweight="bold",
    )
    fig.text(
        0.045, 0.925,
        "The factorized prior costs 0.24–0.38 nats at every M, two to twenty times the "
        "update rule.\nNo agent whose state is a product of Bernoullis can recover that, "
        "whatever its update rule.",
        color=INK_SOFT, fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_classification_metrics(payload: dict, out: Path) -> None:
    """The task in the metrics a multi-label recogniser is normally judged by."""

    counts = payload["config"]["symbol_counts"]
    sweep = payload["sweep"]
    order = ["heisenberg", "heisenberg-gauge", "heisenberg-pe", "adf"]
    metrics = (
        ("mean_ap", "mean average precision  (mAP)", "ranking quality, threshold-free"),
        ("macro_f1", "macro-F1", "unweighted mean of per-category F1, at 0.5"),
        ("accuracy", "Hamming accuracy", "fraction of the 12 label decisions correct"),
        ("exact_set", "subset accuracy", "all 12 labels correct at once"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), facecolor=SURFACE)
    for ax, (metric, title, note) in zip(axes.ravel(), metrics, strict=True):
        _style(ax)
        entries = []
        for rule in order:
            y = [sweep[str(m)][rule][metric] for m in counts]
            ax.plot(counts, y, color=SERIES[rule], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            entries.append((y[-1], LABEL[rule], SERIES[rule], "normal"))
        y = [sweep[str(m)]["exact"][metric] for m in counts]
        ax.plot(counts, y, color=INK, linewidth=2, linestyle=(0, (5, 2)), zorder=2)
        entries.append((y[-1], "exact Bayes", INK, "bold"))
        y = [sweep[str(m)]["prior"][metric] for m in counts]
        ax.plot(counts, y, color=INK_MUTED, linewidth=1.6, linestyle=(0, (1, 2)), zorder=2)
        entries.append((y[-1], "prior (no evidence)", INK_MUTED, "normal"))

        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=16)
        ax.annotate(note, xy=(0, 1.015), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=8.5, va="bottom")
        ax.set_xlabel("symbols absorbed  (M)", color=INK_SOFT, fontsize=9.5)
        ax.set_xticks(counts)
        ax.set_xlim(counts[0] - 0.3, counts[-1] + 5.6)
        _label_column(ax, counts[-1], entries, gap_frac=0.075)

    fig.suptitle(
        "The task, in the metrics it is normally judged by",
        color=INK, fontsize=14, x=0.038, ha="left", y=0.99, fontweight="bold",
    )
    fig.text(
        0.038, 0.945,
        "Higher is better in all four panels. mAP is rank-based, so it cannot see the "
        "gauge fix at all —\nthat correction is a constant shift and leaves every "
        "within-category ranking untouched.",
        color=INK_SOFT, fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_metric_disagreement(payload: dict, out: Path) -> None:
    """Why the standard metrics disagree: the additive rule trades precision for recall."""

    counts = payload["config"]["symbol_counts"]
    sweep = payload["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), facecolor=SURFACE)

    ax = axes[0]
    _style(ax)
    entries = []
    for rule, color in (("heisenberg", "#2a78d6"), ("exact", INK)):
        for metric, dash in (("macro_recall", None), ("macro_precision", (0, (5, 2)))):
            y = [sweep[str(m)][rule][metric] for m in counts]
            ax.plot(counts, y, color=color, linewidth=2, linestyle=dash or "solid",
                    marker="o", markersize=4.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.1, zorder=3)
            entries.append((y[-1], f"{LABEL[rule]} · {metric.split('_')[1]}", color,
                            "bold" if rule == "exact" else "normal"))
    ax.set_title("The additive rule predicts more positives", color=INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_ylabel("macro precision / recall", color=INK_SOFT, fontsize=9.5)
    ax.set_xlabel("symbols absorbed  (M)", color=INK_SOFT, fontsize=9.5)
    ax.set_xticks(counts)
    ax.set_xlim(counts[0] - 0.3, counts[-1] + 8.4)
    _label_column(ax, counts[-1], entries, gap_frac=0.08, size=8.5)

    ax = axes[1]
    _style(ax)
    ax.axhline(0, color=INK, linewidth=1.5, zorder=2)
    entries = [(0.0, "exact Bayes", INK, "bold")]
    for metric, color, label in (
        ("macro_f1", "#eb6834", "macro-F1"),
        ("accuracy", "#4a3aa7", "Hamming accuracy"),
        ("exact_set", "#1baf7a", "subset accuracy"),
    ):
        y = [sweep[str(m)]["heisenberg"][metric] - sweep[str(m)]["exact"][metric]
             for m in counts]
        ax.plot(counts, y, color=color, linewidth=2.2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        entries.append((y[-1], label, color, "normal"))
    ax.set_title("…so the verdict depends on the metric", color=INK,
                 fontsize=11, loc="left", pad=10)
    ax.set_ylabel("Heisenberg − exact Bayes", color=INK_SOFT, fontsize=9.5)
    ax.set_xlabel("symbols absorbed  (M)", color=INK_SOFT, fontsize=9.5)
    ax.set_xticks(counts)
    ax.set_xlim(counts[0] - 0.3, counts[-1] + 6.4)
    _label_column(ax, counts[-1], entries, gap_frac=0.075)

    fig.suptitle(
        "One rule, three standard metrics, three different answers",
        color=INK, fontsize=13.5, x=0.04, ha="left", y=0.985, fontweight="bold",
    )
    fig.text(
        0.04, 0.925,
        "Dropping log Z makes the belief overconfident, which at a 0.5 threshold means "
        "more positive predictions:\nrecall rises, precision falls. Macro-F1 rewards that "
        "on rare categories; accuracy, dominated by true negatives, punishes it.",
        color=INK_SOFT, fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_per_category(payload: dict, out: Path, *, at_m: int = 4) -> None:
    """Which supercategories the task is actually hard for."""

    categories = payload["categories"]
    row = payload["sweep"][str(at_m)]["heisenberg"]
    ap = np.array(row["per_category_ap"])
    prevalence = np.array(row["per_category_prevalence"])
    order = np.argsort(ap)

    fig, ax = plt.subplots(figsize=(9.6, 5.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="y", color=SURFACE, linewidth=0)

    y = np.arange(len(order))
    ax.barh(y, ap[order], height=0.62, color="#2a78d6", zorder=3)
    # a surface ring keeps the nested prevalence bar from reading as a stacked segment
    ax.barh(y, prevalence[order], height=0.24, color="#eb6834", zorder=4,
            edgecolor=SURFACE, linewidth=1.6)

    ax.set_yticks(y)
    ax.set_yticklabels([categories[i] for i in order], fontsize=9.5, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("average precision (thick)  ·  prevalence in the corpus (thin)",
                  color=INK_SOFT, fontsize=9.5)
    for pos, i in enumerate(order):
        ax.annotate(f"{ap[i]:.2f}", xy=(ap[i], pos), xytext=(4, 0),
                    textcoords="offset points", va="center", color="#2a78d6",
                    fontsize=8.5, fontweight="bold")

    fig.suptitle(
        f"Per-category difficulty, Heisenberg at M = {at_m}",
        color=INK, fontsize=13.5, x=0.045, ha="left", y=0.985, fontweight="bold",
    )
    fig.text(
        0.045, 0.925,
        "Average precision tracks how nameable a category is in a caption, not how common "
        "it is.\n'appliance' is the rarest of the twelve yet scores 0.70; 'accessory' is "
        "twice as common and scores 0.50.",
        color=INK_SOFT, fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("output/coco_heisenberg/results.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    out = args.output or args.results.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    figure_dissociation(payload, out / "01_dissociation.png")
    figure_contrasts(payload, out / "02_contrasts.png")
    figure_error_law(payload, out / "03_error_law.png")
    figure_error_budget(payload, out / "04_error_budget.png")
    figure_classification_metrics(payload, out / "05_classification_metrics.png")
    figure_metric_disagreement(payload, out / "06_metric_disagreement.png")
    figure_per_category(payload, out / "07_per_category.png")
    print(f"wrote 7 figures to {out}")


if __name__ == "__main__":
    main()
