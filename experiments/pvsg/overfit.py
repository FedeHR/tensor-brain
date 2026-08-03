"""Overfit one fixed PVSG pair batch before running full experiments."""

from __future__ import annotations

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
from experiments.pvsg.io import read_json, write_json, write_jsonl
from experiments.pvsg.models import IntegralTB, PDirect, PerceptionOutputs
from experiments.pvsg.runtime import (
    EvolutionName,
    ModelName,
    category_candidates,
    experiment_parser,
    move_features,
    pop_paths,
    save_checkpoint,
    start_training,
)
from experiments.pvsg.supervision import build_pair_targets, pair_losses, pair_metrics
from tb import IndexVocabulary, ScoreMode

SemanticCondition = Literal["identity", "source", "hierarchy"]
FEATURE_KEYS = ("scene_features", "subject_features", "object_features", "union_features")
SEMANTIC_LEVELS = {
    "identity": (),
    "source": ("source",),
    "hierarchy": ("fine", "basic", "coarse", "domain"),
}


@dataclass(frozen=True)
class OverfitConfig:
    run_name: str
    model: ModelName = "integral"
    evolution: EvolutionName = "original"
    semantic_condition: SemanticCondition = "hierarchy"
    score_mode: ScoreMode = "direct"
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
        if self.evolution not in ("original", "qtb", "relu"):
            raise ValueError(f"unknown evolution: {self.evolution}")
        if self.semantic_condition not in SEMANTIC_LEVELS:
            raise ValueError(f"unknown semantic condition: {self.semantic_condition}")
        if not self.run_name or "/" in self.run_name:
            raise ValueError("run_name must be a nonempty directory name")
        if min(self.num_examples, self.max_steps, self.learning_rate, self.log_every) <= 0:
            raise ValueError("example, step, learning-rate, and logging values must be positive")


@dataclass(frozen=True)
class PreparedOverfit:
    batch: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    vocabulary: IndexVocabulary
    hierarchy: Mapping[str, Any] | None


def prepare_overfit(
    manifest_root: Path, feature_root: Path, config: OverfitConfig
) -> PreparedOverfit:
    dataset = PVSGPairDataset(
        manifest_root / "heldout_video" / "train_pairs.jsonl", feature_root
    )
    if len(dataset) < config.num_examples:
        raise ValueError(f"requested {config.num_examples} examples from {len(dataset)} rows")
    records = tuple(dataset.records[: config.num_examples])
    batch = collate_pair_batch([dataset[index] for index in range(config.num_examples)])
    ontology = read_json(manifest_root / "ontology.json")
    hierarchy = (
        load_object_hierarchy(
            ontology["object_categories"],
            ontology["identities"],
            path=manifest_root / "object_hierarchy.json",
        )
        if config.semantic_condition == "hierarchy"
        else None
    )
    identities = {
        identity
        for record in records
        for identity in (record["subject_identity"], record["object_identity"])
    }
    vocabulary = build_section6_vocabulary(
        ontology,
        identity_names=identities,
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
    trace: bool = False,
) -> PerceptionOutputs:
    categories = category_candidates(candidates)
    if isinstance(model, PDirect):
        return model(
            batch["subject_features"],
            batch["object_features"],
            batch["union_features"],
            candidates["identity"],
            candidates["predicate"],
            category_candidates=categories,
            return_trace=trace,
        )
    return model(
        batch["scene_features"],
        batch["subject_features"],
        batch["object_features"],
        batch["union_features"],
        candidates["identity"],
        candidates["predicate"],
        category_candidates=categories,
        feedback_mode=feedback_mode,
        return_trace=trace,
    )


def run_overfit(
    prepared: PreparedOverfit, output_dir: Path, config: OverfitConfig
) -> dict[str, Any]:
    state_dim = int(prepared.batch["subject_features"].shape[-1])
    run = start_training(
        output_dir.parent,
        output_dir.name,
        asdict(config),
        prepared.vocabulary,
        state_dim,
        model=config.model,
        evolution=config.evolution,
        score_mode=config.score_mode,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        hidden_dim=config.hidden_dim,
    )
    model, optimizer, candidates = run.model, run.optimizer, run.candidates
    batch = move_features(prepared.batch, FEATURE_KEYS, run.device)
    targets = build_pair_targets(
        prepared.batch, prepared.vocabulary, hierarchy=prepared.hierarchy
    ).to(run.device)
    output_dir = run.directory
    write_jsonl(output_dir / "batch.jsonl", prepared.records)

    captures = set(config.capture_steps) | {config.max_steps}
    for step in range(config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch, candidates, trace=step in captures)
        losses = pair_losses(outputs, targets)
        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        losses.total.backward()
        metrics = pair_metrics(outputs, targets)
        values = {"step": step, **losses.scalars(), **metrics}
        success = metrics["accuracy/all_exact"] == 1.0 and (
            float(losses.overfit_excess.detach()) <= config.success_loss
        )
        if step % config.log_every == 0 or step in captures or success:
            write_jsonl(output_dir / "training_trace.jsonl", [values], append=True)
            print(f"step={step} loss={values['loss/total']:.6f}", flush=True)
        if step in captures:
            context = {
                "run_name": config.run_name,
                "step": step,
                "model": config.model,
                "evolution": config.evolution if config.model == "integral" else None,
                "score_mode": config.score_mode,
            }
            write_jsonl(
                output_dir / "scale_trace.jsonl",
                [
                    {**context, **row}
                    for row in scale_trace_rows(model, outputs, candidates, batch)
                ],
                append=True,
            )
        if success or step == config.max_steps:
            break
        optimizer.step()

    model.eval()
    evaluation = {
        "p-sa" if isinstance(model, IntegralTB) else "p-direct": {
            **losses.scalars(),
            **metrics,
        }
    }
    if isinstance(model, IntegralTB):
        with torch.no_grad():
            sampled = _forward(model, batch, candidates, feedback_mode="p-samp")
        evaluation["p-samp"] = {
            **pair_losses(sampled, targets).scalars(),
            **pair_metrics(sampled, targets),
        }
    save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, step=step)
    result = {"success": success, "completed_steps": step, "evaluation": evaluation}
    write_json(output_dir / "result.json", result, sort_keys=True)
    return result


def run_from_snapshot(
    manifest_root: Path,
    feature_root: Path,
    output_root: Path,
    config: OverfitConfig,
) -> dict[str, Any]:
    return run_overfit(
        prepare_overfit(manifest_root, feature_root, config),
        output_root / config.run_name,
        config,
    )


def _parse_args() -> tuple[Path, Path, Path, OverfitConfig]:
    parser = experiment_parser(__doc__)
    parser.add_argument("--model", choices=("integral", "p-direct"))
    parser.add_argument("--evolution", choices=("original", "qtb", "relu"))
    parser.add_argument("--semantic-condition", choices=tuple(SEMANTIC_LEVELS))
    parser.add_argument(
        "--score-mode", choices=("direct", "centered", "learned-bias", "softplus-bias")
    )
    for name, type_ in (
        ("num-examples", int),
        ("hidden-dim", int),
        ("learning-rate", float),
        ("weight-decay", float),
        ("max-steps", int),
        ("success-loss", float),
        ("log-every", int),
    ):
        parser.add_argument(f"--{name}", type=type_)
    parser.add_argument("--capture-steps", type=int, nargs="+")
    values = vars(parser.parse_args())
    roots = pop_paths(values)
    if "capture_steps" in values:
        values["capture_steps"] = tuple(values["capture_steps"])
    return *roots, OverfitConfig(**values)


def main() -> None:
    manifest_root, feature_root, output_root, config = _parse_args()
    if not run_from_snapshot(manifest_root, feature_root, output_root, config)["success"]:
        raise SystemExit("tiny-data overfit failed")


if __name__ == "__main__":
    main()
