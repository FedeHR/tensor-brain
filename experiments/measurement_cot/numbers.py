"""Emit the report's headline numbers as LaTeX macros.

The prose should quote the campaign, not a transcription of it. Every number the
report states in text is defined here from the JSON payloads and used by name in
``main.tex``, so a rerun that moves a result moves the sentence with it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OUTPUT = Path("output/measurement_cot")
TARGET = Path("reports/measurement_cot/numbers.tex")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _aggregate(records, key_fields, value="test_accuracy"):
    buckets = defaultdict(list)
    for record in records:
        buckets[tuple(record[f] for f in key_fields)].append(record[value])
    return buckets


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def collect() -> dict[str, str]:
    macros: dict[str, str] = {}

    plane_path = OUTPUT / "plane.json"
    if plane_path.exists():
        payload = json.loads(plane_path.read_text())
        records = payload["records"]
        buckets = _aggregate(records, ["retain_gate", "condition"])
        seeds = len({r["seed"] for r in records})
        macros["NumSeeds"] = str(seeds)
        macros["TerminalShortcut"] = _fmt(payload["task"]["shortcut_terminal_only"], 2)
        macros["NumTrainQueries"] = str(int(payload["task"]["train_queries"]))
        macros["NumTestQueries"] = str(int(payload["task"]["test_queries"]))

        for retain, tag in ((0.0, "Bottleneck"), (0.25, "Quarter"), (1.0, "Open")):
            present = {c: v for (a, c), v in buckets.items() if a == retain}
            if not present:
                continue
            for condition, macro in (
                ("expected", "Continuous"),
                ("argmax", "Argmax"),
                ("sample-M1", "SampleOne"),
                ("sample-M16", "SampleSixteen"),
                ("none", "NoFeedback"),
                ("pause", "Pause"),
            ):
                if condition in present:
                    macros[f"{tag}{macro}"] = _fmt(_mean(present[condition]))
            lows = [min(v) for v in present.values()]
            highs = [max(v) for v in present.values()]
            macros[f"{tag}Min"] = _fmt(min(lows))
            macros[f"{tag}Max"] = _fmt(max(highs))
            macros[f"{tag}Spread"] = _fmt(max(highs) - min(lows))
            means = {c: _mean(v) for c, v in present.items()}
            macros[f"{tag}MeanSpread"] = _fmt(max(means.values()) - min(means.values()))

    analysis_path = OUTPUT / "analysis.json"
    if analysis_path.exists():
        payload = json.loads(analysis_path.read_text())
        rows = payload.get("monte_carlo[alpha=0]")
        if rows:
            first, last = rows[0], rows[-1]
            macros["MCFirstSamples"] = str(int(first["samples"]))
            macros["MCLastSamples"] = str(int(last["samples"]))
            macros["MCFirstDistance"] = _fmt(first["distance"], 2)
            macros["MCLastDistance"] = _fmt(last["distance"], 3)
            ratio = (first["distance"] / last["distance"]) if last["distance"] else 0.0
            predicted = (last["samples"] / first["samples"]) ** 0.5
            macros["MCObservedRatio"] = _fmt(ratio, 1)
            macros["MCPredictedRatio"] = _fmt(predicted, 1)
        gaps = payload.get("jensen[alpha=0]")
        if gaps and len(gaps) >= 2:
            # A quadratic law means the gap ratio between successive gates equals the
            # squared gate ratio; report the worst deviation across the sweep.
            worst = 0.0
            for before, after in zip(gaps[:-1], gaps[1:], strict=False):
                predicted = (after["feedback_gate"] / before["feedback_gate"]) ** 2
                observed = after["gap"] / before["gap"] if before["gap"] else 0.0
                worst = max(worst, abs(observed - predicted) / predicted)
            macros["JensenQuadraticError"] = _fmt(100 * worst, 1)
        by_entropy = payload.get("jensen_by_entropy[alpha=0]")
        if by_entropy and len(by_entropy) >= 2:
            macros["JensenLowEntropy"] = _fmt(by_entropy[0]["entropy"], 2)
            macros["JensenHighEntropy"] = _fmt(by_entropy[-1]["entropy"], 2)
            low, high = by_entropy[0]["gap"], by_entropy[-1]["gap"]
            macros["JensenGapRatio"] = _fmt(high / low, 2) if low else "n/a"
        steps = payload.get("steps[alpha=0]")
        hard = payload.get("steps-discrete[alpha=0]")
        if steps and hard:
            macros["FrontierMassContinuous"] = _fmt(_mean([r["frontier_mass"] for r in steps]), 2)
            macros["FrontierMassDiscrete"] = _fmt(_mean([r["frontier_mass"] for r in hard]), 2)
            macros["FrontierChance"] = _fmt(_mean([r["frontier_chance"] for r in steps]), 3)
        zeno = payload.get("zeno[alpha=0|a=1,b=1]")
        if zeno:
            macros["ZenoEntropyStart"] = _fmt(zeno[0]["entropy"], 2)
            macros["ZenoEntropyEnd"] = _fmt(zeno[-1]["entropy"], 2)
            macros["ZenoFrontierStart"] = _fmt(zeno[0]["frontier_mass"], 3)
            macros["ZenoFrontierEnd"] = _fmt(zeno[-1]["frontier_mass"], 3)
            macros["ZenoRepeats"] = str(int(zeno[-1]["repeat"]))

    capacity_path = OUTPUT / "capacity.json"
    if capacity_path.exists():
        payload = json.loads(capacity_path.read_text())
        records = payload["records"]
        buckets = _aggregate(records, ["state_dim", "condition"])
        dims = sorted({r["state_dim"] for r in records})
        gaps = {
            d: _mean(buckets[(d, "expected")]) - _mean(buckets[(d, "argmax")])
            for d in dims
            if (d, "expected") in buckets and (d, "argmax") in buckets
        }
        if gaps:
            peak = max(gaps, key=lambda d: gaps[d])
            macros["CapacityPeakDim"] = str(peak)
            macros["CapacityPeakGap"] = _fmt(gaps[peak])
            macros["CapacityWideGap"] = _fmt(gaps[max(dims)])
            # The floor where no condition learns anything, so the gap is
            # uninformative rather than absent.
            floor = min(dims)
            floor_best = max(
                _mean(v) for (d, _c), v in buckets.items() if d == floor
            )
            macros["CapacityFloorAccuracy"] = _fmt(floor_best)
            macros["CapacityFloorDim"] = str(floor)

    schedule_path = OUTPUT / "schedule.json"
    if schedule_path.exists():
        payload = json.loads(schedule_path.read_text())
        buckets = _aggregate(payload["records"], ["schedule"])
        for name, macro in (
            ("all-continuous", "SchedNeverCollapse"),
            ("all-discrete", "SchedAlways"),
            ("anneal-to-continuous", "SchedEarly"),
            ("anneal-to-discrete", "SchedLate"),
            ("alternating", "SchedAlternating"),
        ):
            if (name,) in buckets:
                macros[macro] = _fmt(_mean(buckets[(name,)]))

    depth_path = OUTPUT / "depth.json"
    if depth_path.exists():
        payload = json.loads(depth_path.read_text())
        records = payload["records"]
        buckets = _aggregate(records, ["hops", "condition"])
        hops = sorted({r["hops"] for r in records})
        frontier = {r["hops"]: r["frontier_sizes"][-1] for r in records}
        gaps = {}
        for hop in hops:
            if (hop, "expected") in buckets and (hop, "argmax") in buckets:
                gaps[hop] = _mean(buckets[(hop, "expected")]) - _mean(buckets[(hop, "argmax")])
        if gaps:
            shallow, deep = min(gaps), max(gaps)
            macros["DepthShallow"] = str(shallow)
            macros["DepthDeep"] = str(deep)
            continuous = [
                _mean(buckets[(h, "expected")]) for h in hops if (h, "expected") in buckets
            ]
            if continuous:
                macros["DepthContinuousMin"] = _fmt(min(continuous))
                macros["DepthContinuousMax"] = _fmt(max(continuous))
            for condition, macro in (("sample-M8", "DepthSampleEight"), ("argmax", "DepthArgmax")):
                present = [h for h in hops if (h, condition) in buckets]
                if present:
                    macros[f"{macro}Shallow"] = _fmt(_mean(buckets[(min(present), condition)]))
                    macros[f"{macro}Deep"] = _fmt(_mean(buckets[(max(present), condition)]))
            macros["DepthGapShallow"] = _fmt(gaps[shallow], 3)
            macros["DepthGapDeep"] = _fmt(gaps[deep], 3)
            macros["DepthFrontierShallow"] = str(int(frontier[shallow]))
            macros["DepthFrontierDeep"] = str(int(frontier[deep]))

    return macros


def referenced_macros() -> set[str]:
    """Macro names the report actually uses, so gaps can be reported not hidden."""

    source = TARGET.parent / "main.tex"
    if not source.exists():
        return set()
    known = set(collect())
    return {name for name in known if rf"\{name}" in source.read_text()} | _cited(source, known)


# Prefixes owned by this module. Fixed rather than derived from what happens to be
# computable, so a macro whose whole sweep is missing is still detected as missing.
_OWNED_PREFIXES = (
    "Bottleneck", "Quarter", "Open", "MC", "Jensen", "Frontier", "Zeno", "Sched", "Depth",
    "Capacity", "Terminal", "NumSeeds", "NumTrain", "NumTest",
)


def _cited(source: Path, known: set[str]) -> set[str]:
    """Every capitalised macro in the report that this module is expected to define."""

    import re

    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", source.read_text()))
    return {name for name in used if name.startswith(_OWNED_PREFIXES)}


def main() -> None:
    macros = collect()
    missing = sorted(referenced_macros() - set(macros))
    lines = [
        "% Generated by experiments.measurement_cot.numbers -- do not edit by hand.",
        "% Every figure quoted in the prose is defined here from the campaign JSON.",
    ]
    for name, value in sorted(macros.items()):
        lines.append(rf"\providecommand{{\{name}}}{{}}\renewcommand{{\{name}}}{{{value}\xspace}}")
    if missing:
        lines.append("% Not computable from the JSON present; rendered visibly rather")
        lines.append("% than silently, so a partial campaign cannot ship as a finished one.")
        for name in missing:
            lines.append(rf"\providecommand{{\{name}}}{{\textbf{{??}}\xspace}}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n".join(lines) + "\n")
    print(f"wrote {TARGET} with {len(macros)} macros")
    if missing:
        print(f"MISSING (rendered as ??): {', '.join(missing)}")


if __name__ == "__main__":
    main()
