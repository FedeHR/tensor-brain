"""Train the PVSG subject-object-predicate recognition comparison.

``--protocol`` selects which entities are seen at evaluation. ``heldout_video`` reserves
whole videos, and since PVSG identities are video-scoped its evaluation entities are
novel by construction, matching the paper's VRD-E regime. ``blocked`` trains on each
training video's observation window and additionally reports the same checkpoint on
later frames of those videos, where the identity bank holds the evaluated entity; that
is the VRD-EX regime, and ``docs/pair_known_entity_protocol.md`` records why the blocked
construction is the stronger of the two.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.pvsg.baselines import (
    ComplementarityOutputs,
    FlatFusion,
    FusedLinear,
    LinearProbe,
    PredicateComplementarity,
    PredicatePriors,
)
from experiments.pvsg.data import (
    PVSGObjectDataset,
    PVSGPairDataset,
    collate_pair_batch,
    experiment_loader,
    role_indices,
)
from experiments.pvsg.diagnostics import scale_trace_rows
from experiments.pvsg.evaluation import (
    batch_delays,
    delay_metrics,
    evaluate_pairs,
    predicate_metrics,
    record_delays,
)
from experiments.pvsg.hierarchy import load_object_hierarchy
from experiments.pvsg.indices import build_section6_vocabulary, predicate_label
from experiments.pvsg.io import read_json, write_json, write_jsonl
from experiments.pvsg.models import FeedbackMode, IntegralTB, PDirect, PerceptionOutputs
from experiments.pvsg.runtime import (
    category_candidates,
    experiment_parser,
    move_features,
    pop_paths,
    prepare_device,
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
    "category-only",
    "union-only",
    "union-category-oracle",
    "union-category-predicted",
    "linear-probe",
    "fused-linear",
    "flat-fusion",
    "p-direct",
    "integral-none",
    "integral-p-sa",
    "integral-cat-sa",
    "integral-id-cat-sa",
]
COMPLEMENTARITY_CONDITIONS = (
    "union-only",
    "union-category-oracle",
    "union-category-predicted",
)
# (identity feedback, category feedback) trained for each Integral condition.
INTEGRAL_FEEDBACK: dict[str, tuple[FeedbackMode, FeedbackMode]] = {
    "integral-none": ("none", "none"),
    "integral-p-sa": ("p-sa", "none"),
    "integral-cat-sa": ("none", "p-sa"),
    "integral-id-cat-sa": ("p-sa", "p-sa"),
}
FEATURE_KEYS = (
    "scene_features",
    "subject_features",
    "object_features",
    "union_features",
)
CAPTURE_STEPS = {0, 1, 10, 100, 1_000, 5_000, 10_000}
PAIR_CATEGORY_LEVELS = ("source", "fine", "basic", "coarse", "domain")


@dataclass(frozen=True)
class EvaluationSet:
    """One reported evaluation view and the VRD protocol it stands in for."""

    name: str
    manifest: str
    role: str
    known_entities: bool
    vrd_analogue: str
    description: str


@dataclass(frozen=True)
class PairProtocol:
    """Which manifests supply training, entity enrollment, selection, and reporting.

    ``enrollment_manifest`` names the object manifest whose observations populate the
    identity group. ``None`` keeps the original rule of enrolling exactly the entities
    that appear in a training pair.
    """

    name: str
    train_manifest: str
    train_role: str
    enrollment_manifest: str | None
    selection_set: str
    evaluation_sets: tuple[EvaluationSet, ...]

    def __post_init__(self) -> None:
        names = [view.name for view in self.evaluation_sets]
        if len(set(names)) != len(names):
            raise ValueError("evaluation set names must be unique")
        if self.selection_set not in names:
            raise ValueError("selection_set must name one of the evaluation sets")


# Novel entities at evaluation: whole videos are reserved, and PVSG identities are
# video-scoped, so no identity candidate can be correct. This is the regime the
# corrected pair runs already measured.
NOVEL_ENTITY_SET = EvaluationSet(
    name="development",
    manifest="heldout_video/development_pairs.jsonl",
    role="development",
    known_entities=False,
    vrd_analogue="VRD-E",
    description=(
        "pairs from videos never trained on; entity instances are novel by "
        "construction, so the identity bank cannot hold the evaluated entity"
    ),
)
# Known entities at evaluation, and a stronger construction than the paper's: VRD-EX
# distorted copies of the training images (tb_original p.23), so "known entity" partly
# meant "nearly the same pixels". Blocked evaluates genuinely later frames of the same
# video behind a 10% temporal embargo, which is real re-identification.
KNOWN_ENTITY_SET = EvaluationSet(
    name="blocked",
    manifest="blocked/evaluation_pairs.jsonl",
    role="train",
    known_entities=True,
    vrd_analogue="VRD-EX",
    description=(
        "pairs from the last 45% of the training videos, after a 10% embargo; both "
        "participants were observed before the observation boundary, so the identity "
        "bank holds the evaluated entity"
    ),
)
PAIR_PROTOCOLS: dict[str, PairProtocol] = {
    "heldout_video": PairProtocol(
        name="heldout_video",
        train_manifest="heldout_video/train_pairs.jsonl",
        train_role="train",
        enrollment_manifest=None,
        selection_set="development",
        evaluation_sets=(NOVEL_ENTITY_SET,),
    ),
    "blocked": PairProtocol(
        name="blocked",
        train_manifest="blocked/train_pairs.jsonl",
        train_role="train",
        # Pair records alone do not cover every re-observed entity: a blocked evaluation
        # pair only requires both participants to have been *visible* before the
        # boundary. Enrolling from the observation-window object manifest is what makes
        # the known-entity claim hold for the whole evaluation set rather than a
        # silently filtered subset.
        enrollment_manifest="blocked/train_objects.jsonl",
        # Selection stays on the novel-entity set, so the known-entity result is
        # reported rather than selected for.
        selection_set="development",
        evaluation_sets=(NOVEL_ENTITY_SET, KNOWN_ENTITY_SET),
    ),
}


@dataclass(frozen=True)
class EvaluationView:
    """One evaluation set resolved against the materialized manifests."""

    set: EvaluationSet
    data: PVSGPairDataset
    indices: list[int]

    @property
    def records(self) -> list[dict[str, Any]]:
        return [self.data.records[index] for index in self.indices]


@dataclass(frozen=True)
class PairExperimentConfig:
    run_name: str
    condition: PairCondition
    protocol: Literal["heldout_video", "blocked"] = "heldout_video"
    evolution: Literal["original", "qtb"] = "qtb"
    score_mode: Literal["direct", "centered", "softplus-bias"] = "softplus-bias"
    category_feedback_level: Literal[
        "source", "fine", "basic", "coarse", "domain"
    ] = "source"
    # Algorithm 3's beta, applied to every injected embedding. QTB bounds it by one;
    # values above one deliberately leave that range to test whether the measured
    # feedback null follows from the injection being small next to the visual drive.
    feedback_gate: float = 1.0
    # Learn beta instead of fixing it, so the model reports the magnitude it prefers
    # rather than one chosen by a sweep. Softplus-parameterized, initialized at one.
    learn_feedback_gate: bool = False
    feedback_gate_learning_rate: float = 1e-2
    learning_rate: float = 1e-4
    input_mapping_learning_rate: float = 1e-5
    zero_initialize_union: bool = False
    batch_size: int = 128
    max_steps: int = 10_000
    validation_every: int = 250
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
            "category-only",
            *COMPLEMENTARITY_CONDITIONS,
            "linear-probe",
            "fused-linear",
            "flat-fusion",
            "p-direct",
            *INTEGRAL_FEEDBACK,
        ):
            raise ValueError(f"unknown pair condition: {self.condition}")
        if self.protocol not in PAIR_PROTOCOLS:
            raise ValueError(f"unknown pair protocol: {self.protocol}")
        if self.evolution not in ("original", "qtb"):
            raise ValueError(f"unknown evolution: {self.evolution}")
        if self.score_mode not in ("direct", "centered", "softplus-bias"):
            raise ValueError(f"unknown score mode: {self.score_mode}")
        if self.feedback_gate < 0.0 or self.feedback_gate_learning_rate <= 0.0:
            raise ValueError(
                "feedback-gate must be non-negative and its learning rate positive"
            )
        if self.category_feedback_level not in PAIR_CATEGORY_LEVELS:
            raise ValueError(
                f"unknown category feedback level: {self.category_feedback_level}"
            )
        if self.zero_initialize_union and self.condition not in COMPLEMENTARITY_CONDITIONS:
            raise ValueError("zero-initialize-union is defined only for complementarity models")
        if (
            min(
                self.learning_rate,
                self.input_mapping_learning_rate,
                self.batch_size,
                self.max_steps,
                self.validation_every,
                self.validation_examples,
                self.chunk_size,
                self.log_every,
            )
            <= 0
        ):
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


FeedbackModes = tuple[FeedbackMode, FeedbackMode]


def _condition(config: PairExperimentConfig) -> tuple[str, FeedbackModes | None]:
    if config.condition in INTEGRAL_FEEDBACK:
        return "integral", INTEGRAL_FEEDBACK[config.condition]
    return config.condition, None


def _sequential(modes: FeedbackModes, selection: str) -> FeedbackModes:
    """Read the checkpoint back under QTB's attention-then-measurement order."""

    sequential = f"sequential-{selection}"
    return tuple(sequential if mode == "p-sa" else mode for mode in modes)


def _sampled(modes: FeedbackModes) -> FeedbackModes:
    """Read the same checkpoint back with winner-take-all on every active pathway."""

    identity, category = modes
    return (
        "p-samp" if identity == "p-sa" else identity,
        "p-samp" if category == "p-sa" else category,
    )


def _forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    candidates: Mapping[str, Tensor],
    *,
    feedback_mode: FeedbackModes | None,
    category_feedback_level: str = "source",
    feedback_gate: float = 1.0,
    trace: bool = False,
) -> PerceptionOutputs:
    categories = category_candidates(candidates)
    if isinstance(model, IntegralTB):
        if feedback_mode is None:
            raise ValueError("Integral TB requires an explicit feedback mode")
        identity_mode, category_mode = feedback_mode
        return model(
            batch["scene_features"],
            batch["subject_features"],
            batch["object_features"],
            batch["union_features"],
            candidates["identity"],
            candidates["predicate"],
            category_candidates=categories,
            feedback_mode=identity_mode,
            category_feedback_candidates=categories[
                f"object_category/{category_feedback_level}"
            ],
            category_feedback_mode=category_mode,
            feedback_gate=feedback_gate,
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


def _prior_metrics(
    priors: PredicatePriors,
    records: Sequence[Mapping[str, Any]],
    vocabulary,
    *,
    seen_triples: set[tuple[str, str, str]],
) -> dict[str, dict[str, float | int]]:
    """Score the frequency and category-pair priors on one evaluation set."""

    targets = build_predicate_targets(
        {"predicates": [record["predicates"] for record in records]},
        vocabulary,
        allow_unknown=True,
    )
    subject_categories = tuple(str(record["subject_category"]) for record in records)
    object_categories = tuple(str(record["object_category"]) for record in records)
    videos = tuple((str(record["source"]), str(record["video_id"])) for record in records)
    delays = record_delays(records)
    predicate_names = vocabulary.group_labels("predicate")
    metrics = {}
    for label, logits in (
        ("frequency", priors.frequency_logits.expand(len(records), -1)),
        ("category-pair", priors.logits(subject_categories, object_categories)),
    ):
        arguments = (
            logits,
            targets,
            predicate_names,
            subject_categories,
            object_categories,
            videos,
        )
        metrics[label] = {
            **predicate_metrics(*arguments, seen_triples=seen_triples),
            **delay_metrics(delays, *arguments, seen_triples=seen_triples),
        }
    return metrics


def _run_priors(
    output_root: Path,
    config: PairExperimentConfig,
    train_records: Sequence[dict[str, Any]],
    views: Mapping[str, EvaluationView],
    vocabulary,
    protocol_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_root / config.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    predicate_names = tuple(
        label.removeprefix("predicate:") for label in vocabulary.group_labels("predicate")
    )
    priors = PredicatePriors.fit_records(train_records, predicate_names)
    seen = _seen_triples(train_records)
    evaluation = {}
    for name, view in views.items():
        by_prior = _prior_metrics(priors, view.records, vocabulary, seen_triples=seen)
        evaluation[name] = (
            {"default": by_prior["category-pair"]}
            if config.condition == "category-only"
            else by_prior
        )
    result = {"evaluation": evaluation}
    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "train_examples": len(train_records),
            "protocol_layout": protocol_metadata,
            "prior_smoothing": 1.0,
            "runtime": runtime_metadata(torch.device("cpu")),
        },
        sort_keys=True,
    )
    write_json(output_dir / "vocabulary.json", vocabulary.to_dict(), sort_keys=True)
    write_json(output_dir / "result.json", result, sort_keys=True)
    return result


def _source_category_names(ontology: Mapping[str, Any]) -> tuple[str, ...]:
    object_categories = ontology["object_categories"]
    return tuple(object_categories["thing"]) + tuple(object_categories["stuff"])


def _category_targets(
    batch: Mapping[str, Any],
    positions: Mapping[str, int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    try:
        subject = torch.tensor(
            [positions[name] for name in batch["subject_category"]],
            device=device,
        )
        object_ = torch.tensor(
            [positions[name] for name in batch["object_category"]],
            device=device,
        )
    except KeyError as error:
        raise ValueError(f"pair uses unknown object category: {error.args[0]}") from error
    return subject, object_


def _predicate_loss_terms(
    logits: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    distribution = targets / targets.sum(dim=-1, keepdim=True)
    log_probabilities = logits.log_softmax(dim=-1)
    cross_entropy = -(distribution * log_probabilities).sum(dim=-1).mean()
    target_entropy = -torch.xlogy(distribution, distribution).sum(dim=-1).mean()
    kl = F.kl_div(log_probabilities, distribution, reduction="batchmean")
    return cross_entropy, target_entropy, kl


def _complementarity_forward(
    model: PredicateComplementarity,
    batch: Mapping[str, Any],
    category_targets: tuple[Tensor, Tensor],
    *,
    oracle_categories: bool = False,
) -> ComplementarityOutputs:
    return model(
        batch["subject_features"],
        batch["object_features"],
        batch["union_features"],
        subject_categories=category_targets[0],
        object_categories=category_targets[1],
        oracle_categories=oracle_categories,
    )


@torch.inference_mode()
def _evaluate_complementarity(
    model: PredicateComplementarity,
    batches: DataLoader,
    vocabulary,
    category_positions: Mapping[str, int],
    *,
    device: torch.device,
    seen_triples: set[tuple[str, str, str]],
    oracle_categories: bool = False,
) -> dict[str, float | int]:
    logits = []
    targets = []
    subject_categories = []
    object_categories = []
    videos = []
    delays = []
    subject_correct = 0
    object_correct = 0
    pair_correct = 0
    category_examples = 0
    model.eval()
    for cpu_batch in batches:
        batch = move_features(cpu_batch, FEATURE_KEYS, device)
        category_targets = _category_targets(cpu_batch, category_positions, device)
        outputs = _complementarity_forward(
            model,
            batch,
            category_targets,
            oracle_categories=oracle_categories,
        )
        logits.append(outputs.predicate_logits.cpu())
        targets.append(build_predicate_targets(cpu_batch, vocabulary, allow_unknown=True))
        subject_categories.extend(cpu_batch["subject_category"])
        object_categories.extend(cpu_batch["object_category"])
        videos.extend(zip(cpu_batch["source"], cpu_batch["video_id"], strict=True))
        batch_delay = batch_delays(cpu_batch)
        if batch_delay is not None:
            delays.append(batch_delay)
        if outputs.subject_category_logits is not None:
            assert outputs.object_category_logits is not None
            subject_matches = outputs.subject_category_logits.argmax(dim=-1) == category_targets[0]
            object_matches = outputs.object_category_logits.argmax(dim=-1) == category_targets[1]
            subject_correct += int(subject_matches.sum())
            object_correct += int(object_matches.sum())
            pair_correct += int((subject_matches & object_matches).sum())
            category_examples += len(subject_matches)
    if not logits:
        raise ValueError("evaluation requires at least one pair")
    arguments = (
        torch.cat(logits),
        torch.cat(targets),
        vocabulary.group_labels("predicate"),
        subject_categories,
        object_categories,
        videos,
    )
    metrics = predicate_metrics(*arguments, seen_triples=seen_triples)
    metrics.update(
        delay_metrics(
            torch.cat(delays) if delays else None,
            *arguments,
            seen_triples=seen_triples,
        )
    )
    if category_examples:
        metrics.update(
            {
                "accuracy/subject_category": subject_correct / category_examples,
                "accuracy/object_category": object_correct / category_examples,
                "accuracy/entity_category": (
                    (subject_correct + object_correct) / (2 * category_examples)
                ),
                "accuracy/category_pair_exact": pair_correct / category_examples,
            }
        )
    return metrics


def _run_complementarity_learned(
    output_root: Path,
    config: PairExperimentConfig,
    train_records: Sequence[dict[str, Any]],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    evaluation_loaders: Mapping[str, DataLoader],
    vocabulary,
    category_names: Sequence[str],
    state_dim: int,
    seen_triples: set[tuple[str, str, str]],
    protocol_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_root / config.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
    device = prepare_device(config.device, config.seed)
    predicate_names = tuple(
        label.removeprefix("predicate:") for label in vocabulary.group_labels("predicate")
    )
    priors = PredicatePriors.fit_records(train_records, predicate_names)
    pair_logits = priors.dense_logits(category_names)
    model = PredicateComplementarity(
        state_dim,
        pair_logits,
        condition=config.condition,
    ).to(device)
    if config.zero_initialize_union:
        nn.init.zeros_(model.union_readout.weight)
        nn.init.zeros_(model.union_readout.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    category_positions = {name: position for position, name in enumerate(category_names)}
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "train_examples": len(train_records),
            "validation_examples": len(validation_loader.dataset),
            "protocol_layout": protocol_metadata,
            "state_dim": state_dim,
            "category_names": list(category_names),
            "predicate_gradient_into_category_classifier": False,
            "fusion": (
                "union logits + learned scalar * log category-conditioned predicate probability"
            ),
            "prior_smoothing": 1.0,
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "runtime": runtime_metadata(device),
        },
        sort_keys=True,
    )
    write_json(output_dir / "vocabulary.json", vocabulary.to_dict(), sort_keys=True)

    def evaluate(
        loader: DataLoader,
        *,
        oracle_categories: bool = False,
    ) -> dict[str, float | int]:
        return _evaluate_complementarity(
            model,
            loader,
            vocabulary,
            category_positions,
            device=device,
            seen_triples=seen_triples,
            oracle_categories=oracle_categories,
        )

    best_metrics = evaluate(validation_loader)
    best_loss = float(best_metrics["loss/predicate_kl"])
    best_step = 0
    write_jsonl(
        output_dir / "validation_trace.jsonl",
        [{"step": 0, **best_metrics}],
        append=True,
    )
    save_checkpoint(
        output_dir / "checkpoint.pt",
        model,
        optimizer,
        step=0,
        validation_loss=best_loss,
    )
    step = 0
    while step < config.max_steps:
        for cpu_batch in train_loader:
            step += 1
            batch = move_features(cpu_batch, FEATURE_KEYS, device)
            category_targets = _category_targets(cpu_batch, category_positions, device)
            predicate_targets = build_predicate_targets(cpu_batch, vocabulary).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = _complementarity_forward(model, batch, category_targets)
            predicate_ce, predicate_entropy, predicate_kl = _predicate_loss_terms(
                outputs.predicate_logits, predicate_targets
            )
            total = predicate_ce
            subject_category_ce = total.new_zeros(())
            object_category_ce = total.new_zeros(())
            if outputs.subject_category_logits is not None:
                assert outputs.object_category_logits is not None
                subject_category_ce = F.cross_entropy(
                    outputs.subject_category_logits, category_targets[0]
                )
                object_category_ce = F.cross_entropy(
                    outputs.object_category_logits, category_targets[1]
                )
                total = total + subject_category_ce + object_category_ce
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            total.backward()
            optimizer.step()

            if step == 1 or step % config.log_every == 0:
                row = {
                    "step": step,
                    "loss/total": float(total.detach()),
                    "loss/predicate_cross_entropy": float(predicate_ce.detach()),
                    "loss/predicate_target_entropy": float(predicate_entropy.detach()),
                    "loss/predicate_kl": float(predicate_kl.detach()),
                }
                if model.category_logit_scale is not None:
                    row["parameter/category_logit_scale"] = float(
                        model.category_logit_scale.detach()
                    )
                if outputs.subject_category_logits is not None:
                    row.update(
                        {
                            "loss/subject_category": float(subject_category_ce.detach()),
                            "loss/object_category": float(object_category_ce.detach()),
                            "accuracy/subject_category": float(
                                (
                                    outputs.subject_category_logits.argmax(dim=-1)
                                    == category_targets[0]
                                )
                                .float()
                                .mean()
                            ),
                            "accuracy/object_category": float(
                                (
                                    outputs.object_category_logits.argmax(dim=-1)
                                    == category_targets[1]
                                )
                                .float()
                                .mean()
                            ),
                        }
                    )
                write_jsonl(output_dir / "training_trace.jsonl", [row], append=True)
                print(f"step={step} loss={row['loss/total']:.6f}", flush=True)
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

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    assert best_metrics is not None
    evaluation = {}
    for name, loader in evaluation_loaders.items():
        evaluation[name] = {"default": evaluate(loader)}
        if config.condition == "union-category-predicted":
            evaluation[name]["oracle-category-intervention"] = evaluate(
                loader, oracle_categories=True
            )
    result = {
        "best_step": best_step,
        "selection": best_metrics,
        "evaluation": evaluation,
    }
    if model.category_logit_scale is not None:
        result["category_logit_scale"] = float(model.category_logit_scale.detach())
    write_json(output_dir / "result.json", result, sort_keys=True)
    return result


def _resolve_evaluation_views(
    protocol: PairProtocol,
    manifest_root: Path,
    feature_root: Path,
    supported_predicates: set[str],
) -> dict[str, EvaluationView]:
    """Load every reported evaluation set, keeping only scoreable predicate rows."""

    views = {}
    for evaluation_set in protocol.evaluation_sets:
        data = PVSGPairDataset(manifest_root / evaluation_set.manifest, feature_root)
        indices = [
            index
            for index in role_indices(data.records, evaluation_set.role)
            if supported_predicates.intersection(data.records[index]["predicates"])
        ]
        if not indices:
            raise ValueError(
                f"{evaluation_set.name} contains no train-supported predicate targets"
            )
        views[evaluation_set.name] = EvaluationView(evaluation_set, data, indices)
    return views


def _pair_participants(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        identity
        for record in records
        for identity in (record["subject_identity"], record["object_identity"])
    }


def _enrolled_identities(
    protocol: PairProtocol,
    manifest_root: Path,
    feature_root: Path,
    train_records: Sequence[Mapping[str, Any]],
) -> tuple[set[str], str]:
    """Return the entities that own an index column, and where they were enrolled."""

    pair_participants = _pair_participants(train_records)
    if protocol.enrollment_manifest is None:
        return pair_participants, "training_pair_participants"
    observations = PVSGObjectDataset(
        manifest_root / protocol.enrollment_manifest, feature_root
    )
    enrolled = {
        observations.records[index]["identity"]
        for index in role_indices(observations.records, protocol.train_role)
    }
    unenrolled = pair_participants - enrolled
    if unenrolled:
        raise ValueError(
            "training pairs reference entities missing from the enrollment manifest: "
            f"{sorted(unenrolled)[:5]!r}"
        )
    return enrolled, protocol.enrollment_manifest


def _protocol_metadata(
    protocol: PairProtocol,
    views: Mapping[str, EvaluationView],
    *,
    train_examples: int,
    enrollment: str,
    identity_columns: int,
    supervised_identity_columns: int,
) -> dict[str, Any]:
    """Record the protocol layout and its VRD correspondence alongside the run."""

    return {
        "name": protocol.name,
        "train_manifest": protocol.train_manifest,
        "train_role": protocol.train_role,
        "train_examples": train_examples,
        "identity_enrollment": {
            "source": enrollment,
            "columns": identity_columns,
            "columns_supervised_by_training_pairs": supervised_identity_columns,
        },
        "selection_set": protocol.selection_set,
        "evaluation_sets": {
            name: {
                "manifest": view.set.manifest,
                "role": view.set.role,
                "examples": len(view.indices),
                "known_entities": view.set.known_entities,
                "vrd_analogue": view.set.vrd_analogue,
                "description": view.set.description,
            }
            for name, view in views.items()
        },
    }


def run_pair_experiment(
    manifest_root: Path,
    feature_root: Path,
    output_root: Path,
    config: PairExperimentConfig,
) -> dict[str, Any]:
    protocol = PAIR_PROTOCOLS[config.protocol]
    train_data = PVSGPairDataset(
        manifest_root / protocol.train_manifest, feature_root
    )
    train_indices = role_indices(train_data.records, protocol.train_role)
    train_records = [train_data.records[index] for index in train_indices]
    ontology = read_json(manifest_root / "ontology.json")
    hierarchy = load_object_hierarchy(
        ontology["object_categories"],
        ontology["identities"],
        path=manifest_root / "object_hierarchy.json",
    )
    supported = set(ontology["train_supported_predicates"])
    views = _resolve_evaluation_views(protocol, manifest_root, feature_root, supported)
    identities, enrollment = _enrolled_identities(
        protocol, manifest_root, feature_root, train_records
    )
    vocabulary = build_section6_vocabulary(
        ontology,
        identity_names=identities,
        category_levels=PAIR_CATEGORY_LEVELS,
        hierarchy=hierarchy,
    )
    for view in views.values():
        if not view.set.known_entities:
            continue
        unknown = _pair_participants(view.records) - identities
        if unknown:
            raise ValueError(
                f"{view.set.name} claims known entities but {len(unknown)} of them own "
                f"no index column, for example {sorted(unknown)[:5]!r}"
            )
    supervised_identities = _pair_participants(train_records)
    # Enrolled but never pair-supervised columns stay near initialization and compete
    # in the identity softmax; the diagnostics report how much mass they attract.
    unsupervised_identity_columns = torch.tensor(
        [label not in supervised_identities for label in vocabulary.group_labels("identity")],
        dtype=torch.bool,
    )
    protocol_metadata = _protocol_metadata(
        protocol,
        views,
        train_examples=len(train_indices),
        enrollment=enrollment,
        identity_columns=len(identities),
        supervised_identity_columns=len(supervised_identities),
    )
    if config.condition in ("priors", "category-only"):
        return _run_priors(
            output_root, config, train_records, views, vocabulary, protocol_metadata
        )

    generator = torch.Generator().manual_seed(config.seed)
    selection_view = views[protocol.selection_set]
    validation_indices = selection_view.indices
    if len(validation_indices) > config.validation_examples:
        positions = torch.randperm(len(validation_indices), generator=generator)[
            : config.validation_examples
        ]
        validation_indices = sorted(
            selection_view.indices[position] for position in positions
        )
    diagnostic_indices = [
        train_indices[position]
        for position in torch.randperm(len(train_indices), generator=generator)[: config.batch_size]
    ]
    diagnostic_cpu = collate_pair_batch([train_data[index] for index in diagnostic_indices])
    train_loader = _loader(train_data, train_indices, config, train=True)
    validation_loader = _loader(selection_view.data, validation_indices, config)
    evaluation_loaders = {
        name: _loader(view.data, view.indices, config) for name, view in views.items()
    }
    seen = _seen_triples(train_records)

    state_dim = int(diagnostic_cpu["subject_features"].shape[-1])
    if config.condition in COMPLEMENTARITY_CONDITIONS:
        return _run_complementarity_learned(
            output_root,
            config,
            train_records,
            train_loader,
            validation_loader,
            evaluation_loaders,
            vocabulary,
            _source_category_names(ontology),
            state_dim,
            seen,
            protocol_metadata,
        )
    model_name, training_feedback = _condition(config)
    training_config = {
        **asdict(config),
        "objective": {
            "predicate_weight": 1.0,
            "identity_weight": 1.0,
            "category_block": "mean_over_active_subject_object_groups",
        },
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "protocol_layout": protocol_metadata,
        "diagnostic_indices": diagnostic_indices,
    }
    if training_feedback is not None:
        identity_mode, category_mode = training_feedback
        training_config["feedback"] = {
            "identity": {
                "mode": identity_mode,
                "candidates": "enrolled_identity_columns",
            },
            "category": {
                "mode": category_mode,
                "candidates": (
                    f"object_category/{config.category_feedback_level}"
                    if category_mode != "none"
                    else None
                ),
                "schedule": (
                    "injected after the unary readout and before evolution; each group "
                    "is scored before its own feedback so no readout confirms itself"
                ),
                "reference": (
                    "original Algorithm 1 samples c* at line 22 without injecting it; "
                    "q_S_ddot = q_S + a_{c*} is defined on p.18 and used for chaining "
                    "in Section 5.2, so perception-path category feedback is an "
                    "experimental extension"
                ),
            },
        }
    if model_name in ("integral", "p-direct"):
        training_config["input_mapping"] = {
            "name": "shared_linear_g",
            "shared_windows": ["scene", "subject", "object", "predicate"],
            "input_space": "rms_normalized_dinov3",
            "output_space": "pre_cbs_q",
            "initialization": "identity_weight_zero_bias",
            "decoder_contract": "g_plus maps CBS coordinates back to DINO feature space",
        }
    run = start_training(
        output_root,
        config.run_name,
        training_config,
        vocabulary,
        state_dim,
        model=model_name,
        evolution=config.evolution,
        score_mode=config.score_mode,
        learning_rate=config.learning_rate,
        input_mapping_learning_rate=config.input_mapping_learning_rate,
        feedback_gate_learning_rate=(
            config.feedback_gate_learning_rate if config.learn_feedback_gate else None
        ),
        learn_feedback_gate=config.learn_feedback_gate,
        feature_dim=state_dim,
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
            category_feedback_level=config.category_feedback_level,
            feedback_gate=None if config.learn_feedback_gate else config.feedback_gate,
            trace=True,
        )
        pair_losses(
            outputs,
            build_pair_targets(
                diagnostic_cpu, vocabulary, hierarchy=hierarchy
            ).to(device),
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
                    model,
                    outputs,
                    candidates,
                    diagnostic_batch,
                    unsupervised_identity_columns=unsupervised_identity_columns,
                )
            ],
            append=True,
        )
        optimizer.zero_grad(set_to_none=True)

    def evaluate(
        loader: DataLoader,
        mode: FeedbackModes | None = training_feedback,
        *,
        identities: bool = False,
    ) -> dict[str, float | int]:
        model.eval()
        return evaluate_pairs(
            lambda batch, evaluation_candidates: _forward(
                model,
                batch,
                evaluation_candidates,
                feedback_mode=mode,
                category_feedback_level=config.category_feedback_level,
                feedback_gate=None if config.learn_feedback_gate else config.feedback_gate,
            ),
            loader,
            vocabulary,
            device=device,
            hierarchy=hierarchy,
            seen_triples=seen,
            identities=identities,
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
            targets = build_pair_targets(
                cpu_batch, vocabulary, hierarchy=hierarchy
            ).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward(
                model,
                batch,
                candidates,
                feedback_mode=training_feedback,
                category_feedback_level=config.category_feedback_level,
                feedback_gate=None if config.learn_feedback_gate else config.feedback_gate,
            )
            losses = pair_losses(outputs, targets)
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            losses.total.backward()
            optimizer.step()

            if step == 1 or step % config.log_every == 0:
                row = {"step": step, **losses.scalars(), **pair_metrics(outputs, targets)}
                if config.learn_feedback_gate:
                    row["feedback_gate"] = float(model.resolve_feedback_gate())
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

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    diagnose(best_step, "best")
    assert best_metrics is not None
    if training_feedback is None:
        evaluation_modes = (("default", None),)
    elif "p-sa" in training_feedback:
        # Read the same checkpoint back with expected and with winner feedback.
        expected_label = config.condition.removeprefix("integral-")
        evaluation_modes = (
            (expected_label, training_feedback),
            (f"{expected_label.removesuffix('sa')}samp", _sampled(training_feedback)),
            # Current-arXiv Algorithms 2 and 3 run in sequence rather than as
            # alternatives; these read the same checkpoint under that schedule.
            (f"{expected_label}-sequential", _sequential(training_feedback, "argmax")),
            (
                f"{expected_label}-sequential-sample",
                _sequential(training_feedback, "sample"),
            ),
        )
    else:
        evaluation_modes = (("none", training_feedback),)
    resolved_gate = (
        float(model.resolve_feedback_gate())
        if config.learn_feedback_gate
        else config.feedback_gate
    )
    result = {
        "best_step": best_step,
        "selection": best_metrics,
        "feedback_gate": resolved_gate,
        "evaluation": {
            name: {
                label: evaluate(
                    evaluation_loaders[name],
                    mode,
                    identities=view.set.known_entities,
                )
                for label, mode in evaluation_modes
            }
            for name, view in views.items()
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
            "category-only",
            *COMPLEMENTARITY_CONDITIONS,
            "linear-probe",
            "fused-linear",
            "flat-fusion",
            "p-direct",
            *INTEGRAL_FEEDBACK,
        ),
        required=True,
    )
    parser.add_argument("--protocol", choices=tuple(PAIR_PROTOCOLS))
    parser.add_argument("--evolution", choices=("original", "qtb"))
    parser.add_argument(
        "--score-mode", choices=("direct", "centered", "softplus-bias")
    )
    parser.add_argument("--category-feedback-level", choices=PAIR_CATEGORY_LEVELS)
    parser.add_argument("--feedback-gate", type=float)
    parser.add_argument("--learn-feedback-gate", action="store_true")
    parser.add_argument("--feedback-gate-learning-rate", type=float)
    parser.add_argument("--zero-initialize-union", action="store_true")
    for name, type_ in (
        ("learning-rate", float),
        ("input-mapping-learning-rate", float),
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
