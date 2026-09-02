"""Run the measurement-schedule campaign and write results as JSON.

Sub-commands correspond one-to-one to the report's figures, so each can be rerun
without repeating the others:

``plane``     accuracy over the (retain gate, collapse mode) grid;
``depth``     the discrete-continuous gap as the search frontier grows;
``schedule``  annealed and entropy-adaptive measurement schedules;
``capacity``  whether the alpha=1 null survives a narrower pre-CBS;
``analysis``  frontier mass, Monte-Carlo convergence, Jensen gap, Zeno traces.

Everything is small enough for a laptop CPU; the whole campaign is well under an
hour and needs no accelerator.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.measurement_cot.analysis import (
    jensen_gap,
    jensen_gap_by_entropy,
    jensen_gap_by_temperature,
    monte_carlo_convergence,
    step_report,
    zeno_trajectory,
)
from experiments.measurement_cot.collapse import CollapseSpec
from experiments.measurement_cot.data import build_queries
from experiments.measurement_cot.graph import GraphSpec, LayeredDAG
from experiments.measurement_cot.model import uniform_schedule
from experiments.measurement_cot.train import TrainConfig, train_chain

OUTPUT = Path("output/measurement_cot")

# The conditions that make up the collapse dial. `none` and `pause` are the
# controls that separate "extra evolution steps" from "extra information".
CONDITIONS: dict[str, CollapseSpec] = {
    "none": CollapseSpec(mode="none"),
    "pause": CollapseSpec(mode="pause"),
    "sample-M1": CollapseSpec(mode="sample", samples=1),
    "sample-M2": CollapseSpec(mode="sample", samples=2),
    "sample-M4": CollapseSpec(mode="sample", samples=4),
    "sample-M8": CollapseSpec(mode="sample", samples=8),
    "sample-M16": CollapseSpec(mode="sample", samples=16),
    "argmax": CollapseSpec(mode="argmax"),
    "expected-t0.5": CollapseSpec(mode="expected", temperature=0.5),
    "expected": CollapseSpec(mode="expected"),
    "expected-t2": CollapseSpec(mode="expected", temperature=2.0),
}

MODEL_KWARGS = {"state_dim": 384, "hidden_dim": 1024, "learn_index_bank": False}


def make_task(hops: int = 4, layer_size: int = 64, branching: int = 2, seed: int = 0):
    graph = LayeredDAG(
        GraphSpec(layer_sizes=(400,) + (layer_size,) * hops, branching=branching, seed=seed)
    )
    train, test, stats = build_queries(graph, seed=seed)
    return graph, train, test, stats


def run_one(graph, train, test, spec_or_schedule, retain_gate: float, seed: int, steps: int):
    schedule = (
        spec_or_schedule
        if isinstance(spec_or_schedule, list)
        else uniform_schedule(spec_or_schedule, graph.spec.num_hops)
    )
    config = TrainConfig(
        steps=steps, supervision="frontier", curriculum_stages=3, seed=seed, eval_every=1000
    )
    kwargs = dict(MODEL_KWARGS, retain_gate=retain_gate)
    return train_chain(graph, train, test, schedule, config, model_kwargs=kwargs)


def command_plane(args) -> None:
    """Accuracy across retain gates and collapse modes."""

    graph, train, test, stats = make_task()
    records = []
    total = len(args.retain) * len(args.conditions) * len(args.seeds)
    done = 0
    started = time.time()
    for retain in args.retain:
        for name in args.conditions:
            for seed in args.seeds:
                _, result = run_one(graph, train, test, CONDITIONS[name], retain, seed, args.steps)
                records.append(
                    {
                        "retain_gate": retain,
                        "condition": name,
                        "seed": seed,
                        "train_accuracy": result.train_accuracy,
                        "test_accuracy": result.test_accuracy,
                        "test_accuracy_std": result.test_accuracy_std,
                    }
                )
                done += 1
                print(
                    f"[{done}/{total} {time.time() - started:.0f}s] alpha={retain} "
                    f"{name} seed={seed} test={result.test_accuracy:.3f}",
                    flush=True,
                )
    write(args.out or "plane.json", {"task": stats, "records": records})


def command_depth(args) -> None:
    """How the gap between collapse modes scales with search depth."""

    records = []
    for hops in args.hops:
        graph, train, test, stats = make_task(hops=hops)
        for name in args.conditions:
            for seed in args.seeds:
                _, result = run_one(
                    graph, train, test, CONDITIONS[name], args.retain[0], seed, args.steps
                )
                frontier = [int(len(s)) for s in graph.reachable_layer_sets(0)]
                records.append(
                    {
                        "hops": hops,
                        "condition": name,
                        "seed": seed,
                        "test_accuracy": result.test_accuracy,
                        "frontier_sizes": frontier,
                        "num_queries": stats["num_queries"],
                    }
                )
                print(
                    f"hops={hops} {name} seed={seed} test={result.test_accuracy:.3f}", flush=True
                )
    write(args.out or "depth.json", {"records": records})


def command_schedule(args) -> None:
    """Annealed measurement schedules against the two uniform endpoints."""

    graph, train, test, stats = make_task()
    hops = graph.spec.num_hops
    soft = CollapseSpec(mode="expected")
    hard = CollapseSpec(mode="argmax")
    schedules: dict[str, list[CollapseSpec]] = {
        "all-continuous": [soft] * (hops - 1),
        "all-discrete": [hard] * (hops - 1),
        # Hold the superposition while the frontier is still wide, then commit.
        "anneal-to-discrete": [soft] * (hops - 2) + [hard],
        # The reverse ordering, to check the schedule is not just "one soft step".
        "anneal-to-continuous": [hard] + [soft] * (hops - 2),
        "alternating": [soft if i % 2 == 0 else hard for i in range(hops - 1)],
    }
    records = []
    for name, schedule in schedules.items():
        for seed in args.seeds:
            _, result = run_one(graph, train, test, schedule, args.retain[0], seed, args.steps)
            records.append(
                {
                    "schedule": name,
                    "steps": [s.label() for s in schedule],
                    "seed": seed,
                    "test_accuracy": result.test_accuracy,
                    "test_accuracy_std": result.test_accuracy_std,
                }
            )
            print(f"{name} seed={seed} test={result.test_accuracy:.3f}", flush=True)
    write(args.out or "schedule.json", {"task": stats, "records": records})


def command_capacity(args) -> None:
    """Does the alpha=1 null survive a pre-CBS too narrow to hold the frontier?

    At ``alpha = 1`` a wide state crosses the step boundary intact, so nothing has
    to pass through the index layer and every collapse mode behaves alike. If that
    null is really about *capacity* rather than about the retain gate, narrowing the
    state until it can no longer carry the frontier should bring the collapse modes
    apart again even at ``alpha = 1``.
    """

    graph, train, test, stats = make_task()
    records = []
    for state_dim in args.state_dims:
        for name in args.conditions:
            for seed in args.seeds:
                schedule = uniform_schedule(CONDITIONS[name], graph.spec.num_hops)
                config = TrainConfig(
                    steps=args.steps, supervision="frontier", curriculum_stages=3,
                    seed=seed, eval_every=1000,
                )
                kwargs = dict(
                    MODEL_KWARGS,
                    retain_gate=args.retain[0],
                    state_dim=state_dim,
                    hidden_dim=max(64, 2 * state_dim),
                )
                _, result = train_chain(
                    graph, train, test, schedule, config, model_kwargs=kwargs
                )
                records.append(
                    {
                        "state_dim": state_dim,
                        "retain_gate": args.retain[0],
                        "condition": name,
                        "seed": seed,
                        "test_accuracy": result.test_accuracy,
                    }
                )
                print(
                    f"state_dim={state_dim} {name} seed={seed} "
                    f"test={result.test_accuracy:.3f}",
                    flush=True,
                )
    write(args.out or "capacity.json", {"task": stats, "records": records})


def command_analysis(args) -> None:
    """Read the index layer of a trained chain and measure the commuting failure."""

    graph, train, test, stats = make_task()
    probe = test.index(torch.arange(min(args.probe_size, len(test))))
    payload: dict[str, object] = {"task": stats}

    for retain in args.retain:
        model, result = run_one(
            graph, train, test, CONDITIONS["expected"], retain, args.seeds[0], args.steps
        )
        tag = f"alpha={retain:g}"
        schedule = uniform_schedule(CONDITIONS["expected"], graph.spec.num_hops)
        payload[f"steps[{tag}]"] = [asdict(r) for r in step_report(model, test, schedule)]
        payload[f"monte_carlo[{tag}]"] = monte_carlo_convergence(
            model, probe, hop=2, sample_counts=(1, 2, 4, 8, 16, 32, 64, 128)
        )
        payload[f"jensen[{tag}]"] = jensen_gap(model, probe, hop=2)
        payload[f"jensen_by_entropy[{tag}]"] = jensen_gap_by_entropy(model, probe, hop=2)
        payload[f"jensen_by_temperature[{tag}]"] = jensen_gap_by_temperature(model, probe, hop=2)
        payload[f"accuracy[{tag}]"] = result.test_accuracy
        print(f"analysis alpha={retain} test={result.test_accuracy:.3f}", flush=True)

        for gate_retain in args.zeno_retain:
            for gate_feedback in args.zeno_feedback:
                key = f"zeno[{tag}|a={gate_retain:g},b={gate_feedback:g}]"
                payload[key] = zeno_trajectory(
                    model,
                    probe,
                    hop=2,
                    repeats=args.zeno_repeats,
                    retain_gate=gate_retain,
                    feedback_gate=gate_feedback,
                )

        # A discrete chain, for the frontier-mass comparison at the same alpha.
        model_hard, result_hard = run_one(
            graph, train, test, CONDITIONS["argmax"], retain, args.seeds[0], args.steps
        )
        hard_schedule = uniform_schedule(CONDITIONS["argmax"], graph.spec.num_hops)
        payload[f"steps-discrete[{tag}]"] = [
            asdict(r) for r in step_report(model_hard, test, hard_schedule)
        ]
        payload[f"accuracy-discrete[{tag}]"] = result_hard.test_accuracy

    write(args.out or "analysis.json", payload)


def write(name: str, payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["plane", "depth", "schedule", "analysis", "capacity"]
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--retain", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--hops", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--state-dims", type=int, nargs="+", default=[16, 32, 64, 128, 384])
    parser.add_argument("--probe-size", type=int, default=512)
    parser.add_argument("--zeno-repeats", type=int, default=12)
    parser.add_argument("--zeno-retain", type=float, nargs="+", default=[0.5, 0.9, 1.0])
    parser.add_argument("--zeno-feedback", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    {
        "plane": command_plane,
        "depth": command_depth,
        "schedule": command_schedule,
        "analysis": command_analysis,
        "capacity": command_capacity,
    }[args.command](args)


if __name__ == "__main__":
    main()
