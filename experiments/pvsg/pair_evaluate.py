"""Re-evaluate a finished pair run without retraining it.

Two questions motivate this entry point, and neither needs new training.

Current-arXiv Algorithms 2 and 3 run in sequence — attention as part of input
integration, then generative measurement — whereas original Section 5.3 has attention
*replace* the sampled injection. Every completed run implements the original's reading,
so the QTB schedule has never been reported. It is a readout of the same checkpoint.

``beta`` scales every injected embedding, attention included. Sweeping it here shows
how a checkpoint trained at one gate responds to a different one at inference, which
is a robustness reading rather than the learned-use question; training at each gate is
what answers the latter, and `cluster/pvsg/pair_feedback_gate.sbatch` does that.

The saved vocabulary is reused rather than rebuilt, so candidate groups and column
identities match the checkpoint exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from experiments.pvsg.data import PVSGPairDataset, role_indices
from experiments.pvsg.evaluation import evaluate_pairs
from experiments.pvsg.hierarchy import load_object_hierarchy
from experiments.pvsg.io import read_json, write_json
from experiments.pvsg.pair_experiment import (
    INTEGRAL_FEEDBACK,
    PAIR_PROTOCOLS,
    PairExperimentConfig,
    _enrolled_identities,
    _forward,
    _loader,
    _resolve_evaluation_views,
    _seen_triples,
)
from experiments.pvsg.runtime import build_model, prepare_device
from tb.vocabulary import IndexVocabulary

# Readouts of one checkpoint. "p-sa"/"p-samp" are the original paper's alternatives;
# the sequential variants are the QTB attention-then-measurement order.
READOUTS = (
    ("expected", "p-sa"),
    ("winner", "p-samp"),
    ("sequential-argmax", "sequential-argmax"),
    ("sequential-sample", "sequential-sample"),
)


def _modes(trained: tuple[str, str], readout: str) -> tuple[str, str]:
    """Apply one readout to whichever pathways the run actually trained."""

    return tuple(readout if mode == "p-sa" else mode for mode in trained)


def reevaluate_pair_run(
    run_dir: Path,
    manifest_root: Path,
    feature_root: Path,
    *,
    feedback_gates: tuple[float, ...] = (1.0,),
    device_name: str = "cuda",
    output_name: str = "reevaluation.json",
) -> dict[str, Any]:
    """Report a finished checkpoint under extra readouts and feedback gates."""

    saved = read_json(run_dir / "config.json")
    if saved["condition"] not in INTEGRAL_FEEDBACK:
        raise ValueError(
            f"re-evaluation is defined for Integral conditions, not {saved['condition']}"
        )
    trained_feedback = INTEGRAL_FEEDBACK[saved["condition"]]
    if "p-sa" not in trained_feedback:
        raise ValueError(
            f"{saved['condition']} trains no feedback pathway, so every readout is "
            "identical to the one already reported"
        )
    config = PairExperimentConfig(
        **{
            field: saved[field]
            for field in PairExperimentConfig.__dataclass_fields__
            if field in saved
        }
    )
    device = prepare_device(device_name, config.seed)

    protocol = PAIR_PROTOCOLS[config.protocol]
    train_data = PVSGPairDataset(manifest_root / protocol.train_manifest, feature_root)
    train_records = [
        train_data.records[index]
        for index in role_indices(train_data.records, protocol.train_role)
    ]
    ontology = read_json(manifest_root / "ontology.json")
    hierarchy = load_object_hierarchy(
        ontology["object_categories"],
        ontology["identities"],
        path=manifest_root / "object_hierarchy.json",
    )
    supported = set(ontology["train_supported_predicates"])
    views = _resolve_evaluation_views(protocol, manifest_root, feature_root, supported)
    _identities, _enrollment = _enrolled_identities(
        protocol, manifest_root, feature_root, train_records
    )
    vocabulary = IndexVocabulary.from_dict(read_json(run_dir / "vocabulary.json"))
    seen = _seen_triples(train_records)

    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location=device, weights_only=True
    )
    sample = train_data[0]
    model = build_model(
        int(sample["subject_features"].shape[-1]),
        len(vocabulary.labels),
        model="integral",
        evolution=config.evolution,
        score_mode=config.score_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    evaluation: dict[str, dict[str, dict[str, Any]]] = {}
    for name, view in views.items():
        loader = _loader(view.data, view.indices, config)
        by_readout: dict[str, dict[str, Any]] = {}
        for label, readout in READOUTS:
            modes = _modes(trained_feedback, readout)
            for gate in feedback_gates:
                # The gate reaches attention as well as measurement, so every readout
                # responds to it, including the attention-only one.
                by_readout[f"{label}@beta={gate:g}"] = evaluate_pairs(
                    lambda batch, candidates, modes=modes, gate=gate: _forward(
                        model,
                        batch,
                        candidates,
                        feedback_mode=modes,
                        category_feedback_level=config.category_feedback_level,
                        feedback_gate=gate,
                    ),
                    loader,
                    vocabulary,
                    device=device,
                    hierarchy=hierarchy,
                    seen_triples=seen,
                    identities=view.set.known_entities,
                )
        evaluation[name] = by_readout

    result = {
        "source_run": run_dir.name,
        "checkpoint_step": int(checkpoint["step"]),
        "condition": config.condition,
        "trained_feedback": {
            "identity": trained_feedback[0],
            "category": trained_feedback[1],
        },
        "feedback_gates": list(feedback_gates),
        "gate_applies_to": "attention_and_measurement_injections",
        "evaluation": evaluation,
    }
    write_json(run_dir / output_name, result, sort_keys=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument(
        "--feedback-gates",
        default="1",
        help="comma-separated beta values for the measurement injection",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-name", default="reevaluation.json")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = reevaluate_pair_run(
        arguments.run_dir,
        arguments.manifest_root,
        arguments.feature_root,
        feedback_gates=tuple(
            float(value) for value in arguments.feedback_gates.split(",")
        ),
        device_name=arguments.device,
        output_name=arguments.output_name,
    )
    for evaluation_set, readouts in result["evaluation"].items():
        for label, metrics in readouts.items():
            print(
                f"{evaluation_set:>12} {label:>26} "
                f"KL={metrics['loss/predicate_kl']:.4f} "
                f"R@1={100 * metrics['recall/predicate_assignment_micro@1']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
