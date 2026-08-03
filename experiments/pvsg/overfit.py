"""Overfit one fixed PVSG pair batch before running full experiments."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from experiments.pvsg.data import PVSGPairDataset, collate_pair_batch
from experiments.pvsg.diagnostics import scale_trace_rows
from experiments.pvsg.hierarchy import load_object_hierarchy
from experiments.pvsg.indices import build_section6_vocabulary
from experiments.pvsg.io import (
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.pvsg.models import IntegralTB, PDirect, PerceptionOutputs
from experiments.pvsg.supervision import build_pair_targets, pair_losses, pair_metrics
from tb import IndexVocabulary
from tb.evolution import Evolution, OriginalTBDynamicContext, QTBEvolution, ReLUEvolution

ModelName = Literal["integral", "p-direct"]
EvolutionName = Literal["original", "qtb", "relu"]
SemanticCondition = Literal["identity", "source", "hierarchy"]

FEATURE_KEYS = (
    "scene_features",
    "subject_features",
    "object_features",
    "union_features",
)
SEMANTIC_LEVELS: dict[SemanticCondition, tuple[str, ...]] = {
    "identity": (),
    "source": ("source",),
    "hierarchy": ("fine", "basic", "coarse", "domain"),
}
EVOLUTION_TYPES: dict[EvolutionName, type[Evolution]] = {
    "original": OriginalTBDynamicContext,
    "qtb": QTBEvolution,
    "relu": ReLUEvolution,
}


@dataclass(frozen=True)
class OverfitConfig:
    """All choices that can change one fixed-batch overfit result."""

    run_name: str
    model: ModelName = "integral"
    evolution: EvolutionName = "original"
    semantic_condition: SemanticCondition = "hierarchy"
    num_examples: int = 200
    hidden_dim: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 5_000
    success_loss: float = 1e-2
    log_every: int = 10
    capture_steps: tuple[int, ...] = (0, 1, 10, 100, 500, 2_000)
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.model not in ("integral", "p-direct"):
            raise ValueError(f"unknown model: {self.model}")
        if self.evolution not in EVOLUTION_TYPES:
            raise ValueError(f"unknown evolution: {self.evolution}")
        if self.semantic_condition not in SEMANTIC_LEVELS:
            raise ValueError(f"unknown semantic condition: {self.semantic_condition}")
        if not self.run_name or "/" in self.run_name:
            raise ValueError("run_name must be a nonempty directory name")
        if self.num_examples <= 0 or self.max_steps <= 0:
            raise ValueError("num_examples and max_steps must be positive")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.success_loss <= 0 or self.log_every <= 0:
            raise ValueError("success_loss and log_every must be positive")
        if any(step < 0 for step in self.capture_steps):
            raise ValueError("capture_steps must be nonnegative")


@dataclass(frozen=True)
class PreparedOverfit:
    """The immutable examples, labels, and candidate groups for one run."""

    batch: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    vocabulary: IndexVocabulary
    hierarchy: Mapping[str, Any] | None


def prepare_overfit(
    manifest_root: Path,
    feature_root: Path,
    config: OverfitConfig,
) -> PreparedOverfit:
    """Load a deterministic manifest prefix and construct its finite vocabulary."""

    manifest_path = manifest_root / "heldout_video" / "train_pairs.jsonl"
    dataset = PVSGPairDataset(manifest_path, feature_root)
    if len(dataset) < config.num_examples:
        raise ValueError(
            f"requested {config.num_examples} examples from a manifest with {len(dataset)} rows"
        )
    # Materialization is video-major. A prefix is deterministic and keeps feature I/O
    # contiguous without introducing a special tiny-data sampler.
    records = tuple(dataset.records[: config.num_examples])
    examples = [dataset[index] for index in range(config.num_examples)]
    batch = collate_pair_batch(examples)

    ontology = read_json(manifest_root / "ontology.json")
    hierarchy = None
    if config.semantic_condition == "hierarchy":
        hierarchy = load_object_hierarchy(
            ontology["object_categories"],
            ontology["identities"],
            path=manifest_root / "object_hierarchy.json",
        )
    identity_names = {
        identity
        for record in records
        for identity in (record["subject_identity"], record["object_identity"])
    }
    vocabulary = build_section6_vocabulary(
        ontology,
        identity_names=identity_names,
        category_levels=SEMANTIC_LEVELS[config.semantic_condition],
        hierarchy=hierarchy,
    )
    return PreparedOverfit(batch, records, vocabulary, hierarchy)


def _forward(
    model: PDirect | IntegralTB,
    batch: Mapping[str, Any],
    candidates: Mapping[str, Tensor],
    *,
    feedback_mode: Literal["p-sa", "p-samp"] = "p-sa",
    return_trace: bool = False,
) -> PerceptionOutputs:
    category_candidates = {
        group: indices
        for group, indices in candidates.items()
        if group.startswith("object_category/")
    }
    if isinstance(model, PDirect):
        return model(
            batch["subject_features"],
            batch["object_features"],
            batch["union_features"],
            candidates["identity"],
            candidates["predicate"],
            category_candidates=category_candidates,
            return_trace=return_trace,
        )
    return model(
        batch["scene_features"],
        batch["subject_features"],
        batch["object_features"],
        batch["union_features"],
        candidates["identity"],
        candidates["predicate"],
        category_candidates=category_candidates,
        feedback_mode=feedback_mode,
        return_trace=return_trace,
    )


def _cpu_outputs(outputs: PerceptionOutputs) -> dict[str, Any]:
    return {
        key: (
            {group: logits.detach().cpu() for group, logits in value.items()}
            if isinstance(value, dict)
            else value.detach().cpu()
        )
        for key, value in outputs.items()
        if key != "trace"
    }


def _append_scale_trace(
    path: Path,
    model: PDirect | IntegralTB,
    outputs: PerceptionOutputs,
    candidates: Mapping[str, Tensor],
    config: OverfitConfig,
    step: int,
) -> None:
    common = {
        "run_name": config.run_name,
        "step": step,
        "model": config.model,
        "evolution": config.evolution if config.model == "integral" else None,
        "semantic_condition": config.semantic_condition,
        "feedback_mode": "p-sa" if config.model == "integral" else "none",
        "input_mapping": "sqrt_dim_l2_normalize_dino",
    }
    write_jsonl(
        path,
        [{**common, **row} for row in scale_trace_rows(model, outputs, candidates)],
        append=True,
    )


def run_overfit(
    prepared: PreparedOverfit,
    output_dir: Path,
    config: OverfitConfig,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimize and diagnose one fixed batch, writing all reproducibility artifacts."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
    if len(prepared.records) != config.num_examples:
        raise ValueError("prepared record count must match config.num_examples")
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config.device)
    batch = dict(prepared.batch)
    batch.update({key: prepared.batch[key].to(device) for key in FEATURE_KEYS})
    state_dim = int(batch["subject_features"].shape[-1])
    candidates = {
        group: prepared.vocabulary.indices(group, device=device)
        for group in prepared.vocabulary.groups
    }
    targets = build_pair_targets(
        prepared.batch,
        prepared.vocabulary,
        hierarchy=prepared.hierarchy,
    ).to(device)
    model: PDirect | IntegralTB
    if config.model == "p-direct":
        model = PDirect(state_dim, len(prepared.vocabulary))
    else:
        evolution = EVOLUTION_TYPES[config.evolution](
            state_dim, config.hidden_dim or state_dim
        )
        model = IntegralTB(state_dim, len(prepared.vocabulary), evolution)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            **(dict(provenance) if provenance is not None else {}),
            "resolved_state_dim": state_dim,
            "resolved_hidden_dim": (
                (config.hidden_dim or state_dim) if isinstance(model, IntegralTB) else None
            ),
            "resolved_evolution": (
                config.evolution if isinstance(model, IntegralTB) else None
            ),
            "num_indices": len(prepared.vocabulary),
        },
        sort_keys=True,
    )
    write_json(
        output_dir / "vocabulary.json",
        prepared.vocabulary.to_dict(),
        sort_keys=True,
    )
    write_jsonl(output_dir / "batch.jsonl", prepared.records)

    capture_steps = set(config.capture_steps) | {config.max_steps}
    training_trace_path = output_dir / "training_trace.jsonl"
    scale_trace_path = output_dir / "scale_trace.jsonl"
    completed_steps = 0
    success = False
    final_outputs: PerceptionOutputs | None = None
    final_values: dict[str, float] = {}

    for step in range(config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        capture = step in capture_steps
        outputs = _forward(model, batch, candidates, return_trace=capture)
        losses = pair_losses(outputs, targets)
        if not bool(torch.isfinite(losses.total)):
            raise FloatingPointError(f"non-finite total loss at step {step}")
        losses.total.backward()
        metrics = pair_metrics(outputs, targets)
        values = {"step": step, **losses.scalars(), **metrics}
        success = metrics["accuracy/all_exact"] == 1.0 and (
            losses.total.detach().item() <= config.success_loss
        )
        if step % config.log_every == 0 or capture or success:
            write_jsonl(training_trace_path, [values], append=True)
            print(
                f"step={step} loss={values['loss/total']:.6f} "
                f"all_exact={values['accuracy/all_exact']:.4f}",
                flush=True,
            )
        if capture:
            _append_scale_trace(
                scale_trace_path, model, outputs, candidates, config, step
            )
        if success or step == config.max_steps:
            if not capture:
                optimizer.zero_grad(set_to_none=True)
                outputs = _forward(model, batch, candidates, return_trace=True)
                losses = pair_losses(outputs, targets)
                losses.total.backward()
                _append_scale_trace(
                    scale_trace_path, model, outputs, candidates, config, step
                )
            completed_steps = step
            final_outputs = outputs
            final_values = values
            break
        optimizer.step()
        completed_steps = step + 1

    if final_outputs is None:
        raise RuntimeError("overfit loop ended without final outputs")

    model.eval()
    primary_mode = "p-sa" if isinstance(model, IntegralTB) else "p-direct"
    evaluation: dict[str, dict[str, float]] = {
        primary_mode: {key: value for key, value in final_values.items() if key != "step"}
    }
    prediction_sets: dict[str, Any] = {primary_mode: _cpu_outputs(final_outputs)}
    if isinstance(model, IntegralTB):
        with torch.no_grad():
            sampled = _forward(model, batch, candidates, feedback_mode="p-samp")
        sampled_losses = pair_losses(sampled, targets)
        evaluation["p-samp"] = {
            **sampled_losses.scalars(),
            **pair_metrics(sampled, targets),
        }
        prediction_sets["p-samp"] = _cpu_outputs(sampled)

    torch.save(
        {
            "completed_steps": completed_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
        },
        output_dir / "checkpoint.pt",
    )
    torch.save(
        {
            "targets": asdict(targets.to("cpu")),
            "outputs": prediction_sets,
        },
        output_dir / "predictions.pt",
    )
    result = {
        "success": success,
        "completed_steps": completed_steps,
        "success_criterion": {
            "accuracy/all_exact": 1.0,
            "maximum_loss/total": config.success_loss,
        },
        "evaluation": evaluation,
        "runtime": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
    }
    write_json(output_dir / "result.json", result, sort_keys=True)
    return result


def run_from_snapshot(
    manifest_root: Path,
    feature_root: Path,
    output_root: Path,
    config: OverfitConfig,
) -> dict[str, Any]:
    """Prepare, record, and execute one snapshot-bound overfit experiment."""

    prepared = prepare_overfit(manifest_root, feature_root, config)
    output_dir = output_root / config.run_name
    provenance = {
        "manifest_root": str(manifest_root.resolve()),
        "feature_root": str(feature_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "training_manifest_sha256": sha256_file(
            manifest_root / "heldout_video" / "train_pairs.jsonl"
        ),
        "ontology_sha256": sha256_file(manifest_root / "ontology.json"),
        "object_hierarchy_sha256": (
            sha256_file(manifest_root / "object_hierarchy.json")
            if config.semantic_condition == "hierarchy"
            else None
        ),
        "snapshot_provenance_sha256": sha256_file(manifest_root / "provenance.json"),
        "code_revision": os.environ.get("PVSG_CODE_REVISION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "selection": "first num_examples rows in immutable video-major training manifest",
        "normalization": "sqrt(feature_dim) * L2_normalize(raw_DINO)",
    }
    return run_overfit(prepared, output_dir, config, provenance=provenance)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", choices=("integral", "p-direct"))
    parser.add_argument("--evolution", choices=("original", "qtb", "relu"))
    parser.add_argument(
        "--semantic-condition",
        choices=("identity", "source", "hierarchy"),
    )
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--success-loss", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument(
        "--capture-steps",
        type=int,
        nargs="+",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    values = vars(_parse_args())
    manifest_root = values.pop("manifest_root")
    feature_root = values.pop("feature_root")
    output_root = values.pop("output_root")
    if "capture_steps" in values:
        values["capture_steps"] = tuple(values["capture_steps"])
    result = run_from_snapshot(
        manifest_root,
        feature_root,
        output_root,
        OverfitConfig(**values),
    )
    if not result["success"]:
        raise SystemExit("tiny-data overfit did not satisfy its recorded success criterion")


if __name__ == "__main__":
    main()
