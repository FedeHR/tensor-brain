"""A fixed layered DAG that plays the role of Tensor Brain semantic memory.

The task is a Tensor Brain native analogue of ProsQA (Hao et al., 2024): a start
concept is written into the workspace together with two candidate terminal
concepts, and the agent must decide which terminal concept is reachable. The
graph itself is *fixed* across the whole dataset, so it is knowledge that has to
live in the evolution operator and the index bank, not in the input. Held-out
(start, terminal) pairs are what forbid a lookup table and force the chain to
actually traverse the intermediate layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Int
from torch import Tensor


@dataclass(frozen=True)
class GraphSpec:
    """Shape of the layered DAG."""

    layer_sizes: tuple[int, ...]
    branching: int
    seed: int = 0

    @property
    def num_hops(self) -> int:
        return len(self.layer_sizes) - 1

    @property
    def num_nodes(self) -> int:
        return sum(self.layer_sizes)


class LayeredDAG:
    """Layered DAG with per-layer global index ranges and reachability tables.

    Global index ``k`` identifies a node; ``layer_of[k]`` is its layer. Nodes in
    layer ``l`` are exactly the measurement candidate set at hop ``l``, which is
    the ordinary Tensor Brain notion of a concept group restricting a
    measurement.
    """

    def __init__(self, spec: GraphSpec) -> None:
        self.spec = spec
        generator = torch.Generator().manual_seed(spec.seed)

        offsets: list[int] = []
        running = 0
        for size in spec.layer_sizes:
            offsets.append(running)
            running += size
        self.layer_offsets = tuple(offsets)
        self.layer_indices = tuple(
            torch.arange(offset, offset + size)
            for offset, size in zip(offsets, spec.layer_sizes, strict=True)
        )

        layer_of = torch.empty(spec.num_nodes, dtype=torch.long)
        for layer, indices in enumerate(self.layer_indices):
            layer_of[indices] = layer
        self.layer_of = layer_of

        # children[l][i] holds the `branching` positions inside layer l+1 that
        # node i of layer l points to. Sampling without replacement keeps the
        # effective branching factor equal to `branching`.
        self.children: list[Int[Tensor, "nodes children"]] = []
        for layer in range(spec.num_hops):
            source_size = spec.layer_sizes[layer]
            target_size = spec.layer_sizes[layer + 1]
            if spec.branching > target_size:
                raise ValueError("branching cannot exceed the size of the next layer")
            picks = torch.stack(
                [
                    torch.randperm(target_size, generator=generator)[: spec.branching]
                    for _ in range(source_size)
                ]
            )
            self.children.append(picks.sort(dim=-1).values)

        self.reachable = self._reachability()
        self.frontier_masks = self._frontier_masks()

    def _frontier_masks(self) -> list[Bool[Tensor, "starts nodes"]]:
        """``frontier_masks[l][s]`` marks the layer-``l`` nodes reachable from start ``s``.

        This is the breadth-first frontier an exhaustive search would hold open at
        hop ``l``. It is the well-posed target for an intermediate step: several
        children of the current frontier are equally valid continuations, so a
        one-hot path target would be forcing an arbitrary tie-break, whereas the
        frontier is the whole truth about where the search can be at that hop.
        """

        masks = [torch.eye(self.spec.layer_sizes[0], dtype=torch.bool)]
        for layer in range(self.spec.num_hops):
            child = self.children[layer]
            previous = masks[-1]
            nxt = torch.zeros(previous.shape[0], self.spec.layer_sizes[layer + 1], dtype=torch.bool)
            for position in range(previous.shape[1]):
                active = previous[:, position].nonzero(as_tuple=True)[0]
                if active.numel() == 0:
                    continue
                nxt[active[:, None], child[position][None, :]] = True
            masks.append(nxt)
        return masks

    def _reachability(self) -> Bool[Tensor, "starts terminals"]:
        """Boolean table: can layer-0 node ``s`` reach terminal-layer node ``t``."""

        frontier = torch.eye(self.spec.layer_sizes[0], dtype=torch.bool)
        for layer in range(self.spec.num_hops):
            child = self.children[layer]
            next_size = self.spec.layer_sizes[layer + 1]
            nxt = torch.zeros(frontier.shape[0], next_size, dtype=torch.bool)
            for position in range(frontier.shape[1]):
                active = frontier[:, position].nonzero(as_tuple=True)[0]
                if active.numel() == 0:
                    continue
                nxt[active[:, None], child[position][None, :]] = True
            frontier = nxt
        return frontier

    def reachable_layer_sets(self, start_position: int) -> list[Int[Tensor, " nodes"]]:
        """Per-layer local positions reachable from a layer-0 node.

        This is the ground-truth breadth-first frontier that an exhaustive search
        would have to represent, and is what the frontier-mass analysis compares
        the index distribution against.
        """

        frontier = torch.zeros(self.spec.layer_sizes[0], dtype=torch.bool)
        frontier[start_position] = True
        sets = [frontier.nonzero(as_tuple=True)[0]]
        for layer in range(self.spec.num_hops):
            child = self.children[layer]
            nxt = torch.zeros(self.spec.layer_sizes[layer + 1], dtype=torch.bool)
            for position in frontier.nonzero(as_tuple=True)[0].tolist():
                nxt[child[position]] = True
            frontier = nxt
            sets.append(frontier.nonzero(as_tuple=True)[0])
        return sets

    def gold_path(self, start_position: int, terminal_position: int) -> list[int]:
        """One path of local positions, used for step-level supervision.

        Depth-first search returning the first path found. Every start-to-terminal
        path in a layered DAG has the same length, so any of them is a valid
        chain-of-thought target.
        """

        def descend(layer: int, position: int) -> list[int] | None:
            if layer == self.spec.num_hops:
                return [position] if position == terminal_position else None
            for nxt in self.children[layer][position].tolist():
                tail = descend(layer + 1, nxt)
                if tail is not None:
                    return [position, *tail]
            return None

        path = descend(0, start_position)
        if path is None:
            raise ValueError("terminal is not reachable from start")
        return path

    def to_global(self, layer: int, positions: Int[Tensor, "*batch"]) -> Int[Tensor, "*batch"]:
        return positions + self.layer_offsets[layer]
