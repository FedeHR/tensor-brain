"""Small shared mechanics for PVSG training scripts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from experiments.pvsg.baselines import FlatFusion, FusedLinear, LinearProbe
from experiments.pvsg.io import write_json
from experiments.pvsg.models import IntegralTB, PDirect
from tb import IndexVocabulary, ScoreMode
from tb.evolution import OriginalTBDynamicContext, QTBEvolution, ReLUEvolution

ModelName = Literal[
    "integral", "p-direct", "linear-probe", "fused-linear", "flat-fusion"
]
EvolutionName = Literal["original", "qtb", "relu"]


def experiment_parser(description: str) -> argparse.ArgumentParser:
    """Create the common CLI used by concrete PVSG experiments."""

    parser = argparse.ArgumentParser(
        description=description, argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    return parser


def pop_paths(values: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Remove and return dataset and output paths from parsed arguments."""

    return (
        values.pop("manifest_root"),
        values.pop("feature_root"),
        values.pop("output_root"),
    )


def prepare_device(name: str, seed: int) -> torch.device:
    """Seed PyTorch and resolve one requested device."""

    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    return torch.device(name)


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    """Record the numerical runtime, including heterogeneous cluster GPUs."""

    metadata: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "device_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "device_memory_bytes": properties.total_memory,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return metadata


def move_features(
    batch: Mapping[str, Any], keys: Sequence[str], device: torch.device
) -> dict[str, Any]:
    """Move only dense feature tensors; symbolic targets remain on the CPU."""

    result = dict(batch)
    result.update({key: batch[key].to(device, non_blocking=True) for key in keys})
    return result


def candidate_tensors(
    vocabulary: IndexVocabulary, device: torch.device
) -> dict[str, Tensor]:
    return {
        group: vocabulary.indices(group, device=device)
        for group in vocabulary.groups
    }


def category_candidates(candidates: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        group: indices
        for group, indices in candidates.items()
        if group.startswith("object_category/")
    }


def build_model(
    state_dim: int,
    num_indices: int,
    *,
    model: ModelName,
    evolution: EvolutionName,
    score_mode: ScoreMode,
    feature_dim: int | None = None,
    hidden_dim: int | None = None,
    num_sources: int = 2,
) -> nn.Module:
    """Construct a named model without hiding its forward schedule."""

    if model == "p-direct":
        return PDirect(
            state_dim, num_indices, feature_dim=feature_dim, score_mode=score_mode
        )
    if model == "linear-probe":
        return LinearProbe(state_dim, num_indices)
    if model == "fused-linear":
        return FusedLinear(state_dim, num_indices, num_sources=num_sources)
    if model == "flat-fusion":
        return FlatFusion(state_dim, num_indices, hidden_dim or state_dim)
    evolution_type = {
        "original": OriginalTBDynamicContext,
        "qtb": QTBEvolution,
        "relu": ReLUEvolution,
    }[evolution]
    return IntegralTB(
        state_dim,
        num_indices,
        evolution_type(state_dim, hidden_dim or state_dim),
        feature_dim=feature_dim,
        score_mode=score_mode,
    )


@dataclass(frozen=True)
class TrainingRun:
    """Resolved state shared by ordinary PVSG optimization loops."""

    directory: Path
    device: torch.device
    model: nn.Module
    optimizer: torch.optim.Optimizer
    candidates: dict[str, Tensor]


def start_training(
    output_root: Path,
    run_name: str,
    config: Mapping[str, Any],
    vocabulary: IndexVocabulary,
    state_dim: int,
    *,
    model: ModelName,
    evolution: EvolutionName,
    score_mode: ScoreMode,
    learning_rate: float,
    input_mapping_learning_rate: float | None = None,
    weight_decay: float = 0.0,
    feature_dim: int | None = None,
    hidden_dim: int | None = None,
    num_sources: int = 2,
) -> TrainingRun:
    """Resolve the repetitive state surrounding a concrete training loop."""

    output_dir = output_root / run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
    device = prepare_device(str(config["device"]), int(config["seed"]))
    network = build_model(
        state_dim,
        len(vocabulary),
        model=model,
        evolution=evolution,
        score_mode=score_mode,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_sources=num_sources,
    ).to(device)
    input_mapping = getattr(network, "g", None)
    if input_mapping is None or input_mapping_learning_rate is None:
        parameter_groups: list[dict[str, Any]] = [
            {"params": list(network.parameters()), "lr": learning_rate}
        ]
    else:
        mapping_parameters = list(input_mapping.parameters())
        mapping_parameter_ids = {id(parameter) for parameter in mapping_parameters}
        parameter_groups = [
            {
                "params": [
                    parameter
                    for parameter in network.parameters()
                    if id(parameter) not in mapping_parameter_ids
                ],
                "lr": learning_rate,
            },
            {"params": mapping_parameters, "lr": input_mapping_learning_rate},
        ]
    optimizer = torch.optim.Adam(parameter_groups, weight_decay=weight_decay)
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "config.json",
        {
            **config,
            "state_dim": state_dim,
            "feature_dim": feature_dim or state_dim,
            "num_indices": len(vocabulary),
            "num_parameters": sum(parameter.numel() for parameter in network.parameters()),
            "optimizer": {
                "name": "Adam",
                "main_learning_rate": learning_rate,
                "input_mapping_learning_rate": (
                    input_mapping_learning_rate if input_mapping is not None else None
                ),
                "weight_decay": weight_decay,
            },
            "runtime": runtime_metadata(device),
        },
        sort_keys=True,
    )
    write_json(output_dir / "vocabulary.json", vocabulary.to_dict(), sort_keys=True)
    return TrainingRun(
        output_dir,
        device,
        network,
        optimizer,
        candidate_tensors(vocabulary, device),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    **values: float,
) -> None:
    torch.save(
        {
            "step": step,
            **values,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
