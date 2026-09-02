"""Figures for the measurement chain-of-thought report.

Each function consumes one JSON payload written by ``run_experiment`` and emits a
PDF into ``output/measurement_cot/figures``. Series colours are taken in fixed
slot order from a validated categorical palette, every multi-series panel is
direct-labelled as well as legended, and no panel uses two y-scales.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUTPUT = Path("output/measurement_cot")
FIGURES = OUTPUT / "figures"

# Validated categorical slots, assigned in fixed order and never cycled.
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
SURFACE = "#fcfcfb"

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "grid.color": "#e8e7e2",
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    }
)

# Display names, ordered from the narrowest step-boundary channel to the widest.
CONDITION_ORDER = [
    "none",
    "pause",
    "sample-M1",
    "argmax",
    "sample-M2",
    "sample-M4",
    "sample-M8",
    "sample-M16",
    "expected-t0.5",
    "expected",
    "expected-t2",
]
CONDITION_LABEL = {
    "none": "no feedback",
    "pause": "pause vector",
    "sample-M1": "sample $M{=}1$",
    "sample-M2": "sample $M{=}2$",
    "sample-M4": "sample $M{=}4$",
    "sample-M8": "sample $M{=}8$",
    "sample-M16": "sample $M{=}16$",
    "argmax": "argmax",
    "expected-t0.5": r"expected $\tau{=}0.5$",
    "expected": "expected",
    "expected-t2": r"expected $\tau{=}2$",
}


def _load(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text())


def _grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def _aggregate(records, key_fields, value="test_accuracy"):
    """Mean and standard deviation over seeds for each key."""

    buckets = defaultdict(list)
    for record in records:
        buckets[tuple(record[f] for f in key_fields)].append(record[value])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in buckets.items()}


def figure_plane(name: str = "plane.json") -> Path:
    """Accuracy across the retain gate and the collapse mode."""

    payload = _load(name)
    records = payload["records"]
    stats = _aggregate(records, ["retain_gate", "condition"])
    retains = sorted({r["retain_gate"] for r in records})
    conditions = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in records)]

    matrix = np.full((len(conditions), len(retains)), np.nan)
    for i, condition in enumerate(conditions):
        for j, retain in enumerate(retains):
            if (retain, condition) in stats:
                matrix[i, j] = stats[(retain, condition)][0]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 3.3), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.42}
    )

    ax = axes[0]
    # Sequential magnitude: one hue, light to dark.
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blues", ["#eef4fc", BLUE, "#12325c"])
    image = ax.imshow(matrix, cmap=cmap, vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(retains)), [f"{r:g}" for r in retains])
    ax.set_yticks(range(len(conditions)), [CONDITION_LABEL[c] for c in conditions])
    ax.set_xlabel(r"retain gate $\alpha$")
    ax.set_title("Accuracy over the collapse dial", loc="left", color=INK)
    for i in range(len(conditions)):
        for j in range(len(retains)):
            if np.isnan(matrix[i, j]):
                continue
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=6.2,
                color="white" if matrix[i, j] > 0.78 else INK,
            )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.02)
    bar.set_label("test accuracy", size=7, color=INK_2)
    bar.ax.tick_params(labelsize=6.5, length=0)
    bar.outline.set_visible(False)

    ax = axes[1]
    # Four near-identical condition curves would waste this panel. What the grid
    # actually shows is how fast the choice of collapse stops mattering, so plot
    # the spread across all conditions directly, on a log axis.
    spreads = []
    for retain in retains:
        means = [stats[(retain, c)][0] for c in conditions if (retain, c) in stats]
        spreads.append(max(means) - min(means))
    ax.plot(retains, spreads, color=BLUE, marker="o", markersize=5)
    for retain, spread in zip(retains, spreads, strict=True):
        ax.annotate(
            f"{spread:.3f}", (retain, spread), textcoords="offset points",
            xytext=(0, 7), fontsize=6.5, color=INK_2, ha="center",
        )
    ax.annotate(
        "", xy=(0.25, spreads[1] * 1.9), xytext=(0.0, spreads[0] * 0.55),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "linewidth": 1.2,
                    "connectionstyle": "arc3,rad=-0.25"},
    )
    ax.text(
        0.30, spreads[0] * 0.30,
        f"retaining a quarter of the state\nshrinks the spread "
        f"{spreads[0] / max(spreads[1], 1e-9):.0f}$\\times$",
        fontsize=6.8, color=ORANGE,
    )
    _grid(ax)
    ax.set_yscale("log")
    ax.set_xlabel(r"retain gate $\alpha$")
    ax.set_ylabel("spread across all 11 collapse modes")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("How fast the choice stops mattering", loc="left", color=INK)
    return _save(fig, "plane")


def figure_channel(name: str = "plane.json") -> Path:
    """Accuracy against the number of Monte-Carlo draws at the bottleneck."""

    payload = _load(name)
    records = [r for r in payload["records"] if r["retain_gate"] == 0.0]
    stats = _aggregate(records, ["condition"])
    samples = [1, 2, 4, 8, 16]
    available = [m for m in samples if (f"sample-M{m}",) in stats]
    means = [stats[(f"sample-M{m}",)][0] for m in available]
    errors = [stats[(f"sample-M{m}",)][1] for m in available]

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.errorbar(
        available, means, yerr=errors, color=BLUE, marker="o", markersize=4,
        capsize=2, elinewidth=0.8, label="sampled measurement",
    )
    if ("expected",) in stats:
        limit = stats[("expected",)][0]
        ax.axhline(limit, color=ORANGE, linewidth=1.4)
        ax.annotate(
            f"continuous limit  {limit:.2f}", (available[0], limit), textcoords="offset points",
            xytext=(2, 4), fontsize=6.8, color=ORANGE,
        )
    if ("argmax",) in stats:
        greedy = stats[("argmax",)][0]
        ax.axhline(greedy, color=AQUA, linewidth=1.0, linestyle="--")
        ax.annotate(
            f"argmax  {greedy:.2f}", (available[0], greedy), textcoords="offset points",
            xytext=(2, -9), fontsize=6.8, color=AQUA,
        )
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    _grid(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(available, [str(m) for m in available])
    ax.set_xlabel("measurement draws $M$ per step")
    ax.set_ylabel("test accuracy")
    ax.set_title(r"Widening the channel at $\alpha{=}0$", loc="left", color=INK)
    ax.legend(loc="lower right")
    return _save(fig, "channel")


def figure_mean_field(name: str = "analysis.json", tag: str = "alpha=0") -> Path:
    """Monte-Carlo convergence of sampled feedback to the exact expectation."""

    payload = _load(name)
    rows = payload[f"monte_carlo[{tag}]"]
    samples = np.array([r["samples"] for r in rows])
    distance = np.array([r["distance"] for r in rows])

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(samples, distance, color=BLUE, marker="o", markersize=4, label="measured distance")
    reference = distance[0] * samples**-0.5
    ax.plot(samples, reference, color=INK_2, linestyle="--", linewidth=1.0, label=r"$M^{-1/2}$")
    ax.annotate(
        r"$M^{-1/2}$", (samples[-1], reference[-1]), textcoords="offset points",
        xytext=(3, -2), fontsize=6.8, color=INK_2,
    )
    _grid(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("measurement draws $M$")
    ax.set_ylabel(r"$\|\hat{f}_M - \mathbb{E}_p[a]\|$")
    ax.set_title("Discrete measurement is a Monte-Carlo\nestimate of continuous feedback",
                 loc="left", color=INK)
    ax.legend(loc="upper right")
    return _save(fig, "mean_field")


def figure_jensen(name: str = "analysis.json", tag: str = "alpha=0") -> Path:
    """The gap between evolving the mean and averaging the evolved states."""

    payload = _load(name)
    gate_rows = payload[f"jensen[{tag}]"]
    entropy_rows = payload[f"jensen_by_entropy[{tag}]"]
    temperature_rows = payload.get(f"jensen_by_temperature[{tag}]", [])

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), gridspec_kw={"wspace": 0.40})

    ax = axes[0]
    beta = np.array([r["feedback_gate"] for r in gate_rows])
    gap = np.array([r["gap"] for r in gate_rows])
    ax.plot(beta, gap, color=BLUE, marker="o", markersize=4, label="measured")
    reference = gap[0] * (beta / beta[0]) ** 2
    ax.plot(beta, reference, color=INK_2, linestyle="--", linewidth=1.0, label=r"$\beta^{2}$")
    _grid(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(beta, [f"{b:g}" for b in beta])
    ax.minorticks_off()
    ax.set_xlabel(r"feedback gate $\beta$")
    ax.set_ylabel(r"$\|\Phi(\bar q)-\mathbb{E}[\Phi(q)]\|$")
    ax.set_title("(a) the $\\beta^{2}$ law", loc="left", color=INK)
    ax.legend(loc="upper left")

    ax = axes[1]
    if temperature_rows:
        spread = np.array([r["spread"] for r in temperature_rows])
        gap_t = np.array([r["gap"] for r in temperature_rows])
        ax.plot(spread, gap_t, color=BLUE, marker="o", markersize=4)
        # Where the trained chain actually sits: the whole cross-section of its own
        # steps occupies a narrow, saturated band at the right-hand edge.
        observed = np.array([r["spread"] for r in entropy_rows])
        ax.axvspan(observed.min(), observed.max(), color="#f0efe9", zorder=0)
        ax.annotate(
            "the chain's own\noperating range",
            (float(observed.mean()), float(gap_t.max())),
            textcoords="offset points", xytext=(-64, -6), fontsize=6.4, color=INK_2,
        )
    _grid(ax)
    ax.set_xlabel(r"candidate spread $\mathrm{tr}\,\mathrm{Cov}_p[a]$")
    ax.set_ylabel(r"$\|\Phi(\bar q)-\mathbb{E}[\Phi(q)]\|$")
    ax.set_title("(b) falls as belief sharpens", loc="left", color=INK)

    ax = axes[2]
    entropy = np.array([r["entropy"] for r in entropy_rows])
    gap_e = np.array([r["gap"] for r in entropy_rows])
    ax.plot(entropy, gap_e, color=ORANGE, marker="o", markersize=4)
    ax.set_ylim(0, max(gap_e) * 1.6)
    _grid(ax)
    ax.set_xlabel("index entropy at the step (nats)")
    ax.set_ylabel(r"$\|\Phi(\bar q)-\mathbb{E}[\Phi(q)]\|$")
    ax.set_title("(c) but flat across its own steps", loc="left", color=INK)
    return _save(fig, "jensen")


def figure_frontier(name: str = "analysis.json", tag: str = "alpha=0") -> Path:
    """Frontier mass held by a continuous and a discrete chain, hop by hop."""

    payload = _load(name)
    soft = payload[f"steps[{tag}]"]
    hard = payload[f"steps-discrete[{tag}]"]
    hops = [r["hop"] for r in soft]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    width = 0.36
    positions = np.arange(len(hops))
    for offset, rows, colour, label in (
        (-width / 2 - 0.01, soft, BLUE, "continuous"),
        (+width / 2 + 0.01, hard, ORANGE, "discrete (argmax)"),
    ):
        values = [r["frontier_mass"] for r in rows]
        ax.bar(positions + offset, values, width, color=colour, label=label, zorder=2)
        for x, v in zip(positions + offset, values, strict=True):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.2, color=INK_2)
    chance = [r["frontier_chance"] for r in soft]
    ax.plot(positions, chance, color=INK_2, linestyle=":", linewidth=1.0, marker="_", markersize=8)
    ax.annotate(
        "chance", (positions[-1], chance[-1]), textcoords="offset points",
        xytext=(4, -2), fontsize=6.5, color=INK_2,
    )
    _grid(ax)
    ax.set_xticks(positions, [str(h) for h in hops])
    ax.set_xlabel("reasoning step")
    ax.set_ylabel("probability mass on the true frontier")
    ax.set_ylim(0, 1.0)
    ax.set_title("The superposition is readable at every step", loc="left", color=INK)
    ax.legend(loc="upper left")

    ax = axes[1]
    # Lift over chance, not e^H: the effective-alternatives measure is dominated by
    # the near-zero tail of a 64-way distribution and says little about whether the
    # right nodes are the ones being held.
    for rows, colour, label in ((soft, BLUE, "continuous"), (hard, ORANGE, "discrete (argmax)")):
        lift = [r["frontier_mass"] / max(r["frontier_chance"], 1e-9) for r in rows]
        ax.plot(
            [r["hop"] for r in rows], lift, color=colour, marker="o", markersize=4, label=label,
        )
        ax.annotate(
            f"{lift[-1]:.1f}$\\times$", ([r["hop"] for r in rows][-1], lift[-1]),
            textcoords="offset points", xytext=(4, -1), fontsize=6.5, color=INK_2,
        )
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(hops[-1], 1.35, "chance", fontsize=6.5, color=INK_2, ha="right")
    twin = ax.twiny()
    twin.set_xticks([])
    twin.set_xlabel(
        "true frontier size:  "
        + "   ".join(f"{r['frontier_size']:.1f}" for r in soft),
        fontsize=6.8, color=INK_2, labelpad=6,
    )
    _grid(ax)
    ax.set_xticks(hops)
    ax.set_ylim(0, None)
    ax.set_xlabel("reasoning step")
    ax.set_ylabel("frontier mass relative to chance")
    ax.set_title("Both decline as the frontier widens", loc="left", color=INK)
    ax.legend(loc="center left")
    return _save(fig, "frontier")


def figure_zeno(name: str = "analysis.json", tag: str = "alpha=0") -> Path:
    """Repeated measurement of one window, with no evolution in between."""

    keys = [k for k in _load(name) if k.startswith(f"zeno[{tag}|")]
    payload = _load(name)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), gridspec_kw={"wspace": 0.40})

    def parse(key: str) -> tuple[float, float]:
        inner = key.split("|")[1].rstrip("]")
        a, b = inner.split(",")
        return float(a.split("=")[1]), float(b.split("=")[1])

    keys = sorted(keys, key=parse)
    betas = sorted({parse(k)[1] for k in keys})
    alphas = sorted({parse(k)[0] for k in keys})
    colours = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]

    panels = [
        ("entropy", "index entropy (nats)", "(a) belief sharpens"),
        ("frontier_mass", "mass on true frontier", "(b) usually towards the truth"),
        ("mode_agreement", "agreement with first outcome", "(c) while revising its first answer"),
    ]
    for ax, (field, ylabel, title) in zip(axes, panels, strict=True):
        for index, beta in enumerate(betas):
            for alpha in alphas:
                key = f"zeno[{tag}|a={alpha:g},b={beta:g}]"
                if key not in payload:
                    continue
                rows = payload[key]
                style = {1.0: "-", 0.9: "--", 0.5: ":"}.get(alpha, "-")
                ax.plot(
                    [r["repeat"] for r in rows], [r[field] for r in rows],
                    color=colours[index % len(colours)], linestyle=style, linewidth=1.4,
                )
        _grid(ax)
        ax.set_xlabel("repeated measurements $R$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=8)

    handles = [Line2D([], [], color=colours[i % len(colours)], label=rf"$\beta={b:g}$")
               for i, b in enumerate(betas)]
    handles += [Line2D([], [], color=INK_2, linestyle={1.0: "-", 0.9: "--", 0.5: ":"}.get(a, "-"),
                       label=rf"$\alpha={a:g}$") for a in alphas]
    axes[0].legend(handles=handles, loc="lower left", ncol=2, fontsize=6.2)
    # The one setting that walks away from the truth deserves to be named rather
    # than left as an unexplained dip.
    axes[1].annotate(
        "weak feedback into\na decaying state\nloses the evidence",
        (5.5, 0.34), textcoords="offset points", xytext=(4, -2),
        fontsize=6.2, color=INK_2, va="top",
    )
    return _save(fig, "zeno")


def figure_depth(name: str = "depth.json") -> Path:
    """How the discrete-continuous gap grows with the search frontier."""

    payload = _load(name)
    records = payload["records"]
    stats = _aggregate(records, ["hops", "condition"])
    hops = sorted({r["hops"] for r in records})
    frontier = {r["hops"]: r["frontier_sizes"][-1] for r in records}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    series = [
        ("expected", BLUE, "continuous"),
        ("argmax", ORANGE, "discrete (argmax)"),
        ("sample-M8", AQUA, "sample $M{=}8$"),
        ("sample-M1", YELLOW, "sample $M{=}1$"),
        ("none", MUTED, "no feedback"),
    ]
    for condition, colour, label in series:
        points = [(h, stats[(h, condition)]) for h in hops if (h, condition) in stats]
        if not points:
            continue
        xs = [p[0] for p in points]
        means = [p[1][0] for p in points]
        errors = [p[1][1] for p in points]
        # Five overlapping series converge at the right edge, so identity is carried
        # by the legend alone rather than by colliding direct labels.
        ax.errorbar(
            xs, means, yerr=errors, color=colour, marker="o", markersize=4, capsize=2,
            elinewidth=0.8, label=label, linestyle="--" if condition == "none" else "-",
        )
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(min(hops), 0.512, "chance", fontsize=6.5, color=INK_2, ha="left")
    _grid(ax)
    ax.set_xticks(hops)
    ax.set_xlabel("search depth (hops)")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(min(hops) - 0.15, max(hops) + 0.15)
    ax.set_ylim(0.38, 1.04)
    ax.set_title(r"Deeper search at $\alpha{=}0$", loc="left", color=INK)
    ax.legend(loc="lower center", ncol=2, fontsize=6.4)

    ax = axes[1]
    gaps, widths = [], []
    for hop in hops:
        if (hop, "expected") in stats and (hop, "argmax") in stats:
            gaps.append(stats[(hop, "expected")][0] - stats[(hop, "argmax")][0])
            widths.append(frontier[hop])
    ax.plot(widths, gaps, color=VIOLET, marker="o", markersize=4)
    for width, gap, hop in zip(widths, gaps, hops, strict=False):
        ax.annotate(f"{hop} hops", (width, gap), textcoords="offset points",
                    xytext=(5, -6), fontsize=6.3, color=INK_2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_xlim(min(widths) - 1.5, max(widths) + 3.5)
    _grid(ax)
    ax.set_xlabel("terminal frontier size")
    ax.set_ylabel("continuous $-$ discrete accuracy")
    ax.set_title("The gap tracks how much must be held open", loc="left", color=INK)
    return _save(fig, "depth")


def figure_capacity(name: str = "capacity.json") -> Path:
    """Whether the alpha=1 null survives narrowing the state that crosses the step."""

    payload = _load(name)
    records = payload["records"]
    stats = _aggregate(records, ["state_dim", "condition"])
    dims = sorted({r["state_dim"] for r in records})

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    for condition, colour, label in (
        ("expected", BLUE, "continuous"),
        ("argmax", ORANGE, "discrete (argmax)"),
        ("none", MUTED, "no feedback"),
    ):
        points = [(d, stats[(d, condition)]) for d in dims if (d, condition) in stats]
        if not points:
            continue
        xs = [p[0] for p in points]
        means = [p[1][0] for p in points]
        errors = [p[1][1] for p in points]
        # The legend carries identity here; direct labels would collide at the
        # right edge where all three conditions converge.
        ax.errorbar(
            xs, means, yerr=errors, color=colour, marker="o", markersize=4, capsize=2,
            elinewidth=0.8, label=label, linestyle="--" if condition == "none" else "-",
        )
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(dims[0], 0.508, "chance", fontsize=6.5, color=INK_2, ha="left")
    _grid(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.set_xlabel("pre-CBS width crossing the step")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(min(dims) * 0.85, max(dims) * 1.2)
    ax.set_title(r"At $\alpha{=}1$, narrowing the state", loc="left", color=INK)
    ax.legend(loc="lower right")

    ax = axes[1]
    gaps, widths = [], []
    for dim in dims:
        if (dim, "expected") in stats and (dim, "argmax") in stats:
            gaps.append(stats[(dim, "expected")][0] - stats[(dim, "argmax")][0])
            widths.append(dim)
    # Below the width at which anything is learned, a zero gap means "no model",
    # not "no effect"; shade that region rather than let it read as a data point.
    floor = [w for w in widths if stats[(w, "expected")][0] < 0.55]
    if floor:
        ax.axvspan(min(widths) * 0.8, max(floor) * 1.4, color="#f0efe9", zorder=0)
        ax.annotate(
            "nothing is learned\nat any sharpness",
            (max(floor), 0.0), textcoords="offset points", xytext=(-2, 24),
            fontsize=6.3, color=INK_2, ha="right",
        )
    ax.plot(widths, gaps, color=VIOLET, marker="o", markersize=4, zorder=3)
    ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=":")
    _grid(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_xlim(min(widths) * 0.8, max(widths) * 1.25)
    ax.set_xlabel("pre-CBS width crossing the step")
    ax.set_ylabel("continuous $-$ discrete accuracy")
    ax.set_title("The gap peaks at intermediate width", loc="left", color=INK)
    return _save(fig, "capacity")


def figure_schedule(name: str = "schedule.json") -> Path:
    """Annealed measurement schedules against the uniform endpoints."""

    payload = _load(name)
    stats = _aggregate(payload["records"], ["schedule"])
    # The run names say which end was annealed; the labels say what that means for
    # *when* the chain commits, which is what the result is actually about.
    label = {
        "anneal-to-discrete": "collapse only at step 3 (widest frontier)",
        "all-discrete": "collapse at every step",
        "alternating": "collapse only at step 2",
        "anneal-to-continuous": "collapse only at step 1 (narrowest frontier)",
        "all-continuous": "never collapse",
    }
    # Bottom to top: worst to best, so the single-collapse schedules read off in the
    # order of the step they collapse at.
    order = [
        "anneal-to-discrete", "all-discrete", "alternating",
        "anneal-to-continuous", "all-continuous",
    ]
    order = [s for s in order if (s,) in stats]
    means = [stats[(s,)][0] for s in order]
    errors = [stats[(s,)][1] for s in order]
    colours = [ORANGE if s == "all-discrete" else BLUE if s == "all-continuous" else AQUA
               for s in order]

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    positions = np.arange(len(order))
    ax.barh(positions, means, 0.6, xerr=errors, color=colours,
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK_2}, zorder=2)
    for y, value in zip(positions, means, strict=True):
        ax.text(value + 0.010, y, f"{value:.3f}", va="center", fontsize=6.8, color=INK_2)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(0.505, len(order) - 0.35, "chance", fontsize=6.5, color=INK_2)
    _grid(ax, axis="x")
    ax.set_yticks(positions, [label.get(s, s) for s in order])
    ax.set_xlabel("test accuracy")
    ax.set_xlim(0.4, max(means) + 0.09)
    ax.set_title("Collapsing late costs more than collapsing early", loc="left", color=INK)
    return _save(fig, "schedule")


def figure_dial() -> Path:
    """The two knobs of the step update, and where known methods sit in the plane."""

    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.20, 1.26)

    # The band where the whole latent state survives the step: nothing has to pass
    # through the index layer, so the sharpness axis stops mattering.
    ax.axhspan(0.82, 1.26, color="#f0efe9", zorder=0)
    ax.text(
        0.02, 1.16, "no bottleneck: the full state crosses the step,\n"
        "so measurement sharpness has no effect",
        fontsize=7, color=INK_2, va="top",
    )

    marks = [
        (0.0, 0.0, ORANGE, "token CoT", r"$w=\delta_k$", "one symbol crosses"),
        (0.34, 0.0, AQUA, "self-consistency", r"$w=\frac{1}{M}\sum_m\delta_{k_m}$",
         "$M$ symbols cross"),
        (1.0, 0.0, BLUE, "continuous thought", r"$w=p$", "a simplex point crosses"),
        (0.5, 1.0, VIOLET, "looped / recurrent depth", r"any $w$", "everything crosses"),
    ]
    for x, y, colour, name, formula, note in marks:
        ax.plot([x], [y], marker="o", markersize=10, color=colour, zorder=4)
        ax.plot([x], [y], marker="o", markersize=14, color=SURFACE, zorder=3)
        offset = 0.075 if y < 0.5 else -0.30
        ax.text(x, y + offset, name, ha="center", fontsize=7.6, color=INK, weight="bold")
        ax.text(x, y + offset + 0.075, formula, ha="center", fontsize=7.6, color=INK)
        ax.text(x, y + offset + 0.145, note, ha="center", fontsize=6.6,
                color=INK_2, style="italic")

    # The interpolating family this report measures.
    ax.plot([0.0, 1.0], [0.0, 0.0], color=INK_2, linewidth=1.0, linestyle="--", zorder=2)
    ax.annotate(
        "", xy=(1.0, -0.10), xytext=(0.0, -0.10),
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.0},
    )
    ax.text(0.5, -0.085, "step-boundary channel widens", ha="center",
            fontsize=6.8, color=INK_2)
    ax.annotate(
        "", xy=(0.62, 0.30), xytext=(0.30, 0.30),
        arrowprops={"arrowstyle": "-|>", "color": YELLOW, "linewidth": 1.4},
    )
    ax.text(0.46, 0.345, "annealed schedule", ha="center", fontsize=7, color=YELLOW,
            weight="bold")

    ax.set_xticks([0.0, 0.5, 1.0], ["sharp\n(one-hot)", "tempered", "degenerate\n(expected)"])
    ax.set_yticks([0.0, 0.5, 1.0], [r"$\alpha=0$", r"$\alpha=0.5$", r"$\alpha=1$"])
    ax.set_xlabel("measurement sharpness")
    ax.set_ylabel(r"retain gate $\alpha$")
    ax.set_title(
        r"$q \;\leftarrow\; \alpha\,q \;+\; \beta\sum_k w_k\,a_k$"
        "   —   one Tensor Brain update, two knobs",
        loc="left", color=INK, fontsize=9,
    )
    ax.tick_params(length=0)
    _grid(ax, axis="both")
    return _save(fig, "dial")


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main() -> None:
    builders = {
        "dial.json": figure_dial,
        "plane.json": lambda: (figure_plane(), figure_channel()),
        "analysis.json": lambda: (
            figure_mean_field(), figure_jensen(), figure_frontier(), figure_zeno()
        ),
        "depth.json": figure_depth,
        "schedule.json": figure_schedule,
        "capacity.json": figure_capacity,
    }
    figure_dial()
    for source, builder in builders.items():
        if source == "dial.json" or not (OUTPUT / source).exists():
            if source != "dial.json":
                print(f"skipping {source}: not found")
            continue
        builder()


if __name__ == "__main__":
    main()
