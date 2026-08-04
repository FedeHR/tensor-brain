"""Train the full PVSG scene-to-object Tensor Brain experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, default_collate

from experiments.pvsg.baselines import FusedLinear, LinearProbe
from experiments.pvsg.data import PVSGObjectDataset, VideoChunkSampler
from experiments.pvsg.diagnostics import object_scale_trace_rows
from experiments.pvsg.evaluation import evaluate_objects
from experiments.pvsg.hierarchy import load_object_hierarchy
from experiments.pvsg.indices import build_section6_vocabulary
from experiments.pvsg.io import read_json, write_json, write_jsonl
from experiments.pvsg.models import FeedbackMode, IntegralTB, ObjectOutputs, PDirect
from experiments.pvsg.runtime import (
    category_candidates,
    experiment_parser,
    move_features,
    pop_paths,
    save_checkpoint,
    start_training,
)
from experiments.pvsg.supervision import build_object_targets, object_losses, object_metrics

SemanticCondition = Literal["source", "hierarchy"]
UnaryCondition = Literal[
    "linear-probe", "fused-linear", "p-direct", "integral-none", "integral-p-sa"
]
CAPTURE_STEPS = {0, 1, 10, 100, 1_000, 5_000, 10_000}
LEVELS = {"source": ("source",), "hierarchy": ("fine", "basic", "coarse", "domain")}


@dataclass(frozen=True)
class ObjectExperimentConfig:
    run_name: str
    evolution: Literal["original", "qtb"]
    score_mode: Literal["centered", "softplus-bias"]
    learning_rate: float
    condition: UnaryCondition = "integral-p-sa"
    semantic_condition: SemanticCondition = "hierarchy"
    batch_size: int = 128
    max_steps: int = 10_000
    validation_every: int = 1_000
    validation_examples: int = 20_000
    chunk_size: int = 1024
    log_every: int = 100
    num_workers: int = 2
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.run_name or "/" in self.run_name:
            raise ValueError("run_name must be a nonempty directory name")
        if self.evolution not in ("original", "qtb"):
            raise ValueError(f"unknown evolution: {self.evolution}")
        if self.score_mode not in ("centered", "softplus-bias"):
            raise ValueError(f"unknown score mode: {self.score_mode}")
        if self.condition not in (
            "linear-probe",
            "fused-linear",
            "p-direct",
            "integral-none",
            "integral-p-sa",
        ):
            raise ValueError(f"unknown unary condition: {self.condition}")
        if min(
            self.learning_rate,
            self.batch_size,
            self.max_steps,
            self.validation_every,
            self.validation_examples,
            self.chunk_size,
            self.log_every,
        ) <= 0:
            raise ValueError("learning-rate, batch, chunk, and step values must be positive")


def _role_indices(records: Sequence[dict[str, Any]], role: str) -> list[int]:
    indices = [i for i, record in enumerate(records) if record["experiment_split"] == role]
    if not indices:
        raise ValueError(f"manifest contains no {role!r} rows")
    return indices


def _loader(
    dataset: PVSGObjectDataset,
    indices: Sequence[int],
    config: ObjectExperimentConfig,
    *,
    train: bool = False,
) -> DataLoader:
    sampler = (
        VideoChunkSampler(
            [dataset.records[index] for index in indices],
            chunk_size=config.chunk_size,
            generator=torch.Generator().manual_seed(config.seed),
        )
        if train
        else None
    )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        persistent_workers=config.num_workers > 0,
    )


def _forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    candidates: Mapping[str, Tensor],
    *,
    feedback_mode: FeedbackMode | None,
    sequential_categories: bool = False,
    trace: bool = False,
) -> ObjectOutputs:
    categories = category_candidates(candidates)
    if isinstance(model, IntegralTB):
        if feedback_mode is None:
            raise ValueError("Integral TB requires an explicit feedback mode")
        return model.forward_object(
            batch["scene_features"],
            batch["object_features"],
            candidates["identity"],
            category_candidates=categories,
            feedback_mode=feedback_mode,
            sequential_categories=sequential_categories,
            return_trace=trace,
        )
    if sequential_categories:
        raise ValueError("sequential category feedback requires Integral TB")
    if isinstance(model, PDirect):
        return model.forward_object(
            batch["object_features"],
            candidates["identity"],
            category_candidates=categories,
            return_trace=trace,
        )
    if trace:
        raise ValueError("scale traces are defined only for Tensor Brain models")
    if isinstance(model, LinearProbe):
        return model.forward_object(
            batch["object_features"],
            candidates["identity"],
            category_candidates=categories,
        )
    if isinstance(model, FusedLinear):
        return model.forward_object(
            batch["scene_features"],
            batch["object_features"],
            candidates["identity"],
            category_candidates=categories,
        )
    raise TypeError(f"unsupported unary model: {type(model).__name__}")


def _condition(config: ObjectExperimentConfig) -> tuple[str, FeedbackMode | None]:
    if config.condition == "integral-none":
        return "integral", "none"
    if config.condition == "integral-p-sa":
        return "integral", "p-sa"
    return config.condition, None


def run_object_experiment(
    manifest_root: Path,
    feature_root: Path,
    output_root: Path,
    config: ObjectExperimentConfig,
) -> dict[str, Any]:
    train_data = PVSGObjectDataset(
        manifest_root / "blocked" / "train_objects.jsonl", feature_root
    )
    blocked_data = PVSGObjectDataset(
        manifest_root / "blocked" / "evaluation_objects.jsonl", feature_root
    )
    development_data = PVSGObjectDataset(
        manifest_root / "heldout_video" / "development_objects.jsonl", feature_root
    )
    train_indices = _role_indices(train_data.records, "train")
    blocked_indices = _role_indices(blocked_data.records, "train")
    development_indices = _role_indices(development_data.records, "development")

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
    identities = {train_data.records[index]["identity"] for index in train_indices}
    vocabulary = build_section6_vocabulary(
        ontology,
        identity_names=identities,
        category_levels=LEVELS[config.semantic_condition],
        hierarchy=hierarchy,
    )
    if not {
        blocked_data.records[index]["identity"] for index in blocked_indices
    } <= identities:
        raise ValueError("blocked evaluation contains an unseen identity")

    generator = torch.Generator().manual_seed(config.seed)
    validation_indices = development_indices
    if len(validation_indices) > config.validation_examples:
        positions = torch.randperm(len(validation_indices), generator=generator)[
            : config.validation_examples
        ]
        validation_indices = sorted(development_indices[position] for position in positions)
    diagnostic_indices = [
        train_indices[position]
        for position in torch.randperm(len(train_indices), generator=generator)[: config.batch_size]
    ]
    diagnostic_cpu = default_collate([train_data[index] for index in diagnostic_indices])
    train_loader = _loader(train_data, train_indices, config, train=True)
    validation_loader = _loader(development_data, validation_indices, config)

    state_dim = int(diagnostic_cpu["object_features"].shape[-1])
    model_name, training_feedback = _condition(config)
    run = start_training(
        output_root,
        config.run_name,
        {
            **asdict(config),
            "train_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "blocked_examples": len(blocked_indices),
            "diagnostic_indices": diagnostic_indices,
        },
        vocabulary,
        state_dim,
        model=model_name,
        evolution=config.evolution,
        score_mode=config.score_mode,
        learning_rate=config.learning_rate,
    )
    model, optimizer, candidates = run.model, run.optimizer, run.candidates
    device, output_dir = run.device, run.directory
    diagnostic_batch = move_features(
        diagnostic_cpu, ("scene_features", "object_features"), device
    )
    trace_path = output_dir / "scale_trace.jsonl"

    def diagnose(step: int, checkpoint: str | None = None) -> None:
        if not isinstance(model, (IntegralTB, PDirect)):
            return
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(
            model,
            diagnostic_batch,
            candidates,
            feedback_mode=training_feedback,
            trace=True,
        )
        object_losses(
            outputs,
            build_object_targets(
                diagnostic_batch, vocabulary, hierarchy=hierarchy
            ).to(device),
        ).total.backward()
        context = {
            "run_name": config.run_name,
            "evolution": config.evolution,
            "score_mode": config.score_mode,
            "condition": config.condition,
            "step": step,
        }
        if checkpoint:
            context["checkpoint"] = checkpoint
        write_jsonl(
            trace_path,
            [
                {**context, **row}
                for row in object_scale_trace_rows(
                    model, outputs, candidates, diagnostic_batch
                )
            ],
            append=True,
        )
        optimizer.zero_grad(set_to_none=True)

    diagnose(0)

    def evaluate(
        loader: DataLoader,
        mode: FeedbackMode | None = training_feedback,
        *,
        identities: bool = False,
        sequential_categories: bool = False,
    ) -> dict[str, float | int]:
        model.eval()
        return evaluate_objects(
            lambda batch, evaluation_candidates: _forward(
                model,
                batch,
                evaluation_candidates,
                feedback_mode=mode,
                sequential_categories=sequential_categories,
            ),
            loader,
            vocabulary,
            device=device,
            hierarchy=hierarchy,
            identities=identities,
        )

    best_metrics: dict[str, float | int] | None = None
    best_loss = float("inf")
    best_step = 0

    step = 0
    while step < config.max_steps:
        for cpu_batch in train_loader:
            step += 1
            batch = move_features(cpu_batch, ("scene_features", "object_features"), device)
            targets = build_object_targets(cpu_batch, vocabulary, hierarchy=hierarchy).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward(
                model, batch, candidates, feedback_mode=training_feedback
            )
            losses = object_losses(outputs, targets)
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            losses.total.backward()
            optimizer.step()

            if step == 1 or step % config.log_every == 0:
                row = {"step": step, **losses.scalars(), **object_metrics(outputs, targets)}
                write_jsonl(output_dir / "training_trace.jsonl", [row], append=True)
                print(f"step={step} loss={row['loss/total']:.6f}", flush=True)
            if step in CAPTURE_STEPS or step == config.max_steps:
                diagnose(step)
            if step % config.validation_every == 0 or step == config.max_steps:
                metrics = evaluate(validation_loader)
                write_jsonl(
                    output_dir / "validation_trace.jsonl",
                    [{"step": step, **metrics}],
                    append=True,
                )
                if metrics["loss/category_total"] < best_loss:
                    best_step, best_metrics = step, metrics
                    best_loss = float(metrics["loss/category_total"])
                    save_checkpoint(
                        output_dir / "checkpoint.pt",
                        model,
                        optimizer,
                        step=step,
                        validation_loss=best_loss,
                    )
            if step == config.max_steps:
                break

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    diagnose(best_step, "best")
    assert best_metrics is not None
    evaluation = {}
    if config.condition == "integral-p-sa":
        evaluation_modes = (
            ("p-sa", "p-sa", False),
            ("p-samp", "p-samp", False),
            ("p-sa-sequential", "p-sa", True),
            ("p-samp-sequential", "p-samp", True),
        )
    elif config.condition == "integral-none":
        evaluation_modes = (("none", "none", False),)
    else:
        evaluation_modes = (("default", None, False),)
    for name, data, indices, identities in (
        ("development", development_data, development_indices, False),
        ("blocked", blocked_data, blocked_indices, True),
    ):
        loader = _loader(data, indices, config)
        evaluation[name] = {
            label: evaluate(
                loader,
                mode,
                identities=identities,
                sequential_categories=sequential,
            )
            for label, mode, sequential in evaluation_modes
        }
    result = {"best_step": best_step, "selection": best_metrics, "evaluation": evaluation}
    write_json(output_dir / "result.json", result, sort_keys=True)
    return result


def _parse_args() -> tuple[Path, Path, Path, ObjectExperimentConfig]:
    parser = experiment_parser(__doc__)
    parser.add_argument("--evolution", choices=("original", "qtb"), required=True)
    parser.add_argument(
        "--score-mode", choices=("centered", "softplus-bias"), required=True
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument(
        "--condition",
        choices=(
            "linear-probe",
            "fused-linear",
            "p-direct",
            "integral-none",
            "integral-p-sa",
        ),
    )
    parser.add_argument("--semantic-condition", choices=("source", "hierarchy"))
    for name in (
        "batch-size",
        "max-steps",
        "validation-every",
        "validation-examples",
        "chunk-size",
        "log-every",
        "num-workers",
    ):
        parser.add_argument(f"--{name}", type=int)
    values = vars(parser.parse_args())
    roots = pop_paths(values)
    return *roots, ObjectExperimentConfig(**values)


def main() -> None:
    manifest_root, feature_root, output_root, config = _parse_args()
    run_object_experiment(manifest_root, feature_root, output_root, config)


if __name__ == "__main__":
    main()
