"""Train the PVSG subject-object-predicate recognition comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments.pvsg.baselines import (
    FlatFusion,
    FusedLinear,
    LinearProbe,
    PredicatePriors,
)
from experiments.pvsg.data import (
    PVSGPairDataset,
    collate_pair_batch,
    experiment_loader,
    role_indices,
)
from experiments.pvsg.diagnostics import scale_trace_rows
from experiments.pvsg.evaluation import evaluate_pairs, predicate_metrics
from experiments.pvsg.indices import build_section6_vocabulary, predicate_label
from experiments.pvsg.io import read_json, write_json, write_jsonl
from experiments.pvsg.models import FeedbackMode, IntegralTB, PDirect, PerceptionOutputs
from experiments.pvsg.runtime import (
    category_candidates,
    experiment_parser,
    move_features,
    pop_paths,
    runtime_metadata,
    save_checkpoint,
    start_training,
)
from experiments.pvsg.supervision import (
    build_pair_targets,
    build_predicate_targets,
    pair_losses,
    pair_metrics,
)

PairCondition = Literal[
    "priors",
    "linear-probe",
    "fused-linear",
    "flat-fusion",
    "p-direct",
    "integral-none",
    "integral-p-sa",
]
FEATURE_KEYS = (
    "scene_features",
    "subject_features",
    "object_features",
    "union_features",
)
CAPTURE_STEPS = {0, 1, 10, 100, 1_000, 5_000, 10_000}


@dataclass(frozen=True)
class PairExperimentConfig:
    run_name: str
    condition: PairCondition
    evolution: Literal["original", "qtb"] = "qtb"
    score_mode: Literal["centered", "softplus-bias"] = "softplus-bias"
    learning_rate: float = 1e-3
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
        if self.condition not in (
            "priors",
            "linear-probe",
            "fused-linear",
            "flat-fusion",
            "p-direct",
            "integral-none",
            "integral-p-sa",
        ):
            raise ValueError(f"unknown pair condition: {self.condition}")
        if self.evolution not in ("original", "qtb"):
            raise ValueError(f"unknown evolution: {self.evolution}")
        if self.score_mode not in ("centered", "softplus-bias"):
            raise ValueError(f"unknown score mode: {self.score_mode}")
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


def _loader(
    dataset: PVSGPairDataset,
    indices: Sequence[int],
    config: PairExperimentConfig,
    *,
    train: bool = False,
) -> DataLoader:
    return experiment_loader(
        dataset,
        indices,
        batch_size=config.batch_size,
        chunk_size=config.chunk_size,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        train=train,
    )


def _condition(config: PairExperimentConfig) -> tuple[str, FeedbackMode | None]:
    if config.condition == "integral-none":
        return "integral", "none"
    if config.condition == "integral-p-sa":
        return "integral", "p-sa"
    return config.condition, None


def _forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    candidates: Mapping[str, Tensor],
    *,
    feedback_mode: FeedbackMode | None,
    trace: bool = False,
) -> PerceptionOutputs:
    categories = category_candidates(candidates)
    if isinstance(model, IntegralTB):
        if feedback_mode is None:
            raise ValueError("Integral TB requires an explicit feedback mode")
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
    if trace:
        raise ValueError("scale traces are defined only for Tensor Brain models")
    if isinstance(model, LinearProbe):
        return model(
            batch["subject_features"],
            batch["object_features"],
            batch["union_features"],
            candidates["identity"],
            candidates["predicate"],
            category_candidates=categories,
        )
    if isinstance(model, (FusedLinear, FlatFusion)):
        return model(
            batch["scene_features"],
            batch["subject_features"],
            batch["object_features"],
            batch["union_features"],
            candidates["identity"],
            candidates["predicate"],
            category_candidates=categories,
        )
    raise TypeError(f"unsupported pair model: {type(model).__name__}")


def _seen_triples(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (
            str(record["subject_category"]),
            predicate_label(predicate),
            str(record["object_category"]),
        )
        for record in records
        for predicate in record["predicates"]
    }


def _run_priors(
    output_root: Path,
    config: PairExperimentConfig,
    train_records: Sequence[dict[str, Any]],
    development_records: Sequence[dict[str, Any]],
    vocabulary,
) -> dict[str, Any]:
    output_dir = output_root / config.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    predicate_names = tuple(
        label.removeprefix("predicate:")
        for label in vocabulary.group_labels("predicate")
    )
    priors = PredicatePriors.fit_records(train_records, predicate_names)
    targets = build_predicate_targets(
        {"predicates": [record["predicates"] for record in development_records]},
        vocabulary,
        allow_unknown=True,
    )
    subject_categories = tuple(
        str(record["subject_category"]) for record in development_records
    )
    object_categories = tuple(
        str(record["object_category"]) for record in development_records
    )
    videos = tuple(
        (str(record["source"]), str(record["video_id"]))
        for record in development_records
    )
    frequency_logits = priors.frequency_logits.expand(len(development_records), -1)
    pair_logits = priors.logits(subject_categories, object_categories)
    seen = _seen_triples(train_records)
    result = {
        "evaluation": {
            "frequency": predicate_metrics(
                frequency_logits,
                targets,
                vocabulary.group_labels("predicate"),
                subject_categories,
                object_categories,
                videos,
                seen_triples=seen,
            ),
            "category-pair": predicate_metrics(
                pair_logits,
                targets,
                vocabulary.group_labels("predicate"),
                subject_categories,
                object_categories,
                videos,
                seen_triples=seen,
            ),
        }
    }
    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "train_examples": len(train_records),
            "development_examples": len(development_records),
            "runtime": runtime_metadata(torch.device("cpu")),
        },
        sort_keys=True,
    )
    write_json(output_dir / "vocabulary.json", vocabulary.to_dict(), sort_keys=True)
    write_json(output_dir / "result.json", result, sort_keys=True)
    return result


def run_pair_experiment(
    manifest_root: Path,
    feature_root: Path,
    output_root: Path,
    config: PairExperimentConfig,
) -> dict[str, Any]:
    train_data = PVSGPairDataset(
        manifest_root / "heldout_video" / "train_pairs.jsonl", feature_root
    )
    development_data = PVSGPairDataset(
        manifest_root / "heldout_video" / "development_pairs.jsonl", feature_root
    )
    train_indices = role_indices(train_data.records, "train")
    ontology = read_json(manifest_root / "ontology.json")
    supported = set(ontology["train_supported_predicates"])
    development_indices = [
        index
        for index in role_indices(development_data.records, "development")
        if supported.intersection(development_data.records[index]["predicates"])
    ]
    if not development_indices:
        raise ValueError("development contains no train-supported predicate targets")
    train_records = [train_data.records[index] for index in train_indices]
    development_records = [
        development_data.records[index] for index in development_indices
    ]
    identities = {
        identity
        for record in train_records
        for identity in (record["subject_identity"], record["object_identity"])
    }
    vocabulary = build_section6_vocabulary(ontology, identity_names=identities)
    if config.condition == "priors":
        return _run_priors(
            output_root, config, train_records, development_records, vocabulary
        )

    generator = torch.Generator().manual_seed(config.seed)
    validation_indices = development_indices
    if len(validation_indices) > config.validation_examples:
        positions = torch.randperm(len(validation_indices), generator=generator)[
            : config.validation_examples
        ]
        validation_indices = sorted(
            development_indices[position] for position in positions
        )
    diagnostic_indices = [
        train_indices[position]
        for position in torch.randperm(len(train_indices), generator=generator)[
            : config.batch_size
        ]
    ]
    diagnostic_cpu = collate_pair_batch(
        [train_data[index] for index in diagnostic_indices]
    )
    train_loader = _loader(train_data, train_indices, config, train=True)
    validation_loader = _loader(development_data, validation_indices, config)
    seen = _seen_triples(train_records)

    state_dim = int(diagnostic_cpu["subject_features"].shape[-1])
    model_name, training_feedback = _condition(config)
    run = start_training(
        output_root,
        config.run_name,
        {
            **asdict(config),
            "train_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "development_examples": len(development_indices),
            "diagnostic_indices": diagnostic_indices,
        },
        vocabulary,
        state_dim,
        model=model_name,
        evolution=config.evolution,
        score_mode=config.score_mode,
        learning_rate=config.learning_rate,
        num_sources=4,
    )
    model, optimizer, candidates = run.model, run.optimizer, run.candidates
    device, output_dir = run.device, run.directory
    diagnostic_batch = move_features(diagnostic_cpu, FEATURE_KEYS, device)

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
        pair_losses(
            outputs, build_pair_targets(diagnostic_cpu, vocabulary).to(device)
        ).total.backward()
        context = {
            "run_name": config.run_name,
            "condition": config.condition,
            "step": step,
        }
        if checkpoint:
            context["checkpoint"] = checkpoint
        write_jsonl(
            output_dir / "scale_trace.jsonl",
            [
                {**context, **row}
                for row in scale_trace_rows(
                    model, outputs, candidates, diagnostic_batch
                )
            ],
            append=True,
        )
        optimizer.zero_grad(set_to_none=True)

    def evaluate(
        loader: DataLoader, mode: FeedbackMode | None = training_feedback
    ) -> dict[str, float | int]:
        model.eval()
        return evaluate_pairs(
            lambda batch, evaluation_candidates: _forward(
                model,
                batch,
                evaluation_candidates,
                feedback_mode=mode,
            ),
            loader,
            vocabulary,
            device=device,
            seen_triples=seen,
        )

    diagnose(0)
    best_metrics = None
    best_loss = float("inf")
    best_step = 0
    step = 0
    while step < config.max_steps:
        for cpu_batch in train_loader:
            step += 1
            batch = move_features(cpu_batch, FEATURE_KEYS, device)
            targets = build_pair_targets(cpu_batch, vocabulary).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward(
                model, batch, candidates, feedback_mode=training_feedback
            )
            losses = pair_losses(outputs, targets)
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            losses.total.backward()
            optimizer.step()

            if step == 1 or step % config.log_every == 0:
                row = {"step": step, **losses.scalars(), **pair_metrics(outputs, targets)}
                write_jsonl(
                    output_dir / "training_trace.jsonl", [row], append=True
                )
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
                if metrics["loss/predicate_kl"] < best_loss:
                    best_step, best_metrics = step, metrics
                    best_loss = float(metrics["loss/predicate_kl"])
                    save_checkpoint(
                        output_dir / "checkpoint.pt",
                        model,
                        optimizer,
                        step=step,
                        validation_loss=best_loss,
                    )
            if step == config.max_steps:
                break

    checkpoint = torch.load(
        output_dir / "checkpoint.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    diagnose(best_step, "best")
    assert best_metrics is not None
    if config.condition == "integral-p-sa":
        evaluation_modes = (("p-sa", "p-sa"), ("p-samp", "p-samp"))
    elif config.condition == "integral-none":
        evaluation_modes = (("none", "none"),)
    else:
        evaluation_modes = (("default", None),)
    development_loader = _loader(
        development_data, development_indices, config
    )
    result = {
        "best_step": best_step,
        "selection": best_metrics,
        "evaluation": {
            label: evaluate(development_loader, mode)
            for label, mode in evaluation_modes
        },
    }
    write_json(output_dir / "result.json", result, sort_keys=True)
    return result


def _parse_args() -> tuple[Path, Path, Path, PairExperimentConfig]:
    parser = experiment_parser(__doc__)
    parser.add_argument(
        "--condition",
        choices=(
            "priors",
            "linear-probe",
            "fused-linear",
            "flat-fusion",
            "p-direct",
            "integral-none",
            "integral-p-sa",
        ),
        required=True,
    )
    parser.add_argument("--evolution", choices=("original", "qtb"))
    parser.add_argument("--score-mode", choices=("centered", "softplus-bias"))
    for name, type_ in (
        ("learning-rate", float),
        ("batch-size", int),
        ("max-steps", int),
        ("validation-every", int),
        ("validation-examples", int),
        ("chunk-size", int),
        ("log-every", int),
        ("num-workers", int),
    ):
        parser.add_argument(f"--{name}", type=type_)
    values = vars(parser.parse_args())
    roots = pop_paths(values)
    return *roots, PairExperimentConfig(**values)


def main() -> None:
    manifest_root, feature_root, output_root, config = _parse_args()
    run_pair_experiment(manifest_root, feature_root, output_root, config)


if __name__ == "__main__":
    main()
