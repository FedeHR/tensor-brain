"""Queries over the fixed layered DAG, with the marginal shortcut removed.

A query is ``(start, terminal_a, terminal_b, answer)``: exactly one of the two
terminals is reachable from the start. Two things keep the task honest.

Quota-balanced negatives
    Matching the negative to the positive on in-reachability count is *not*
    enough: a terminal that many starts reach is, for that same reason, rarely
    available as somebody else's unreachable option, so it ends up in the correct
    slot more often than chance. Negatives are therefore assigned under a quota,
    so that each terminal occupies the reachable and the unreachable slot equally
    often across the dataset. A rule that looks only at terminal identity is then
    at chance by construction, and a model has to traverse the graph.

Held-out pairs
    The split is over reachable ``(start, terminal)`` pairs. Every start and every
    terminal is seen during training, but their composition is not, so the answer
    has to be recomputed through the intermediate layers rather than looked up.

``shortcut_baselines`` reports what a terminal-only and a start-only rule fitted
on the training split achieve on the test split. Both should sit at chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Int
from torch import Tensor

from experiments.measurement_cot.graph import LayeredDAG


@dataclass
class QuerySet:
    """A batch of reachability queries in global Tensor Brain index space."""

    start: Int[Tensor, " queries"]
    terminal_a: Int[Tensor, " queries"]
    terminal_b: Int[Tensor, " queries"]
    answer: Int[Tensor, " queries"]
    start_position: Int[Tensor, " queries"]
    gold_path: Int[Tensor, "queries hops"]

    def __len__(self) -> int:
        return int(self.start.shape[0])

    def index(self, selector: Tensor) -> QuerySet:
        return QuerySet(
            start=self.start[selector],
            terminal_a=self.terminal_a[selector],
            terminal_b=self.terminal_b[selector],
            answer=self.answer[selector],
            start_position=self.start_position[selector],
            gold_path=self.gold_path[selector],
        )

    def to(self, device: torch.device) -> QuerySet:
        return QuerySet(
            start=self.start.to(device),
            terminal_a=self.terminal_a.to(device),
            terminal_b=self.terminal_b.to(device),
            answer=self.answer.to(device),
            start_position=self.start_position.to(device),
            gold_path=self.gold_path.to(device),
        )


def build_queries(
    graph: LayeredDAG,
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
    negatives_per_positive: int = 2,
) -> tuple[QuerySet, QuerySet, dict[str, float]]:
    """Enumerate reachable pairs, attach quota-balanced negatives, split by pair."""

    generator = torch.Generator().manual_seed(seed)
    terminal_layer = graph.spec.num_hops

    positive_rows: list[tuple[int, int]] = []
    for start_position in range(graph.spec.layer_sizes[0]):
        reachable = graph.reachable[start_position].nonzero(as_tuple=True)[0].tolist()
        for positive in reachable:
            positive_rows.extend([(start_position, positive)] * negatives_per_positive)
    if not positive_rows:
        raise ValueError("graph produced no queries; check layer sizes and branching")

    # Quota: terminal t must serve as the unreachable option exactly as often as it
    # serves as the reachable one. Filling the scarcest quota first is what keeps
    # the assignment feasible when most terminals are reachable from most starts.
    num_terminals = graph.spec.layer_sizes[terminal_layer]
    quota = torch.zeros(num_terminals)
    for _, positive in positive_rows:
        quota[positive] += 1.0

    order = torch.randperm(len(positive_rows), generator=generator).tolist()
    starts: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    paths: list[list[int]] = []
    path_cache: dict[tuple[int, int], list[int]] = {}

    for row in order:
        start_position, positive = positive_rows[row]
        unreachable = ~graph.reachable[start_position]
        if not bool(unreachable.any()):
            continue
        # Break quota ties randomly so the assignment does not track terminal index.
        # Reachable terminals are masked with -inf rather than a finite sentinel:
        # quotas go negative once a terminal has served often enough, and a finite
        # mask could then be outscored, silently selecting a *reachable* negative.
        scores = torch.where(unreachable, quota, torch.full_like(quota, float("-inf")))
        scores = scores + 1e-3 * torch.rand(num_terminals, generator=generator)
        negative = int(scores.argmax())
        quota[negative] -= 1.0

        key = (start_position, positive)
        if key not in path_cache:
            path_cache[key] = graph.gold_path(start_position, positive)
        starts.append(start_position)
        positives.append(positive)
        negatives.append(negative)
        paths.append(path_cache[key])

    start_position = torch.tensor(starts)
    positive = torch.tensor(positives)
    negative = torch.tensor(negatives)
    gold_path = torch.tensor(paths)

    put_positive_first = torch.rand(len(starts), generator=generator) < 0.5
    terminal_a = torch.where(put_positive_first, positive, negative)
    terminal_b = torch.where(put_positive_first, negative, positive)
    answer = torch.where(put_positive_first, 0, 1)

    queries = QuerySet(
        start=graph.to_global(0, start_position),
        terminal_a=graph.to_global(terminal_layer, terminal_a),
        terminal_b=graph.to_global(terminal_layer, terminal_b),
        answer=answer,
        start_position=start_position,
        gold_path=gold_path,
    )

    # Split over distinct reachable (start, terminal) pairs, not over rows, so a
    # test pair never appears in training under a different negative.
    pair_id = start_position * graph.spec.layer_sizes[terminal_layer] + positive
    unique_pairs = pair_id.unique()
    shuffled = unique_pairs[torch.randperm(unique_pairs.numel(), generator=generator)]
    num_train = int(round(train_fraction * shuffled.numel()))
    train_pairs = set(shuffled[:num_train].tolist())
    is_train = torch.tensor([int(p) in train_pairs for p in pair_id.tolist()])

    train = queries.index(is_train.nonzero(as_tuple=True)[0])
    test = queries.index((~is_train).nonzero(as_tuple=True)[0])

    stats = {
        "num_queries": float(len(starts)),
        "num_pairs": float(unique_pairs.numel()),
        "train_queries": float(len(train)),
        "test_queries": float(len(test)),
        "reachable_fraction": float(graph.reachable.float().mean()),
        **shortcut_baselines(train, test, graph),
    }
    return train, test, stats


def shortcut_baselines(
    train: QuerySet, test: QuerySet, graph: LayeredDAG
) -> dict[str, float]:
    """Held-out accuracy of rules that ignore the composition of start and terminal."""

    num_nodes = graph.spec.num_nodes

    def terminal_win_rate(queries: QuerySet) -> Tensor:
        wins = torch.zeros(num_nodes)
        plays = torch.zeros(num_nodes)
        correct = torch.where(queries.answer == 0, queries.terminal_a, queries.terminal_b)
        other = torch.where(queries.answer == 0, queries.terminal_b, queries.terminal_a)
        wins.scatter_add_(0, correct, torch.ones_like(correct, dtype=torch.float))
        plays.scatter_add_(0, correct, torch.ones_like(correct, dtype=torch.float))
        plays.scatter_add_(0, other, torch.ones_like(other, dtype=torch.float))
        return wins / plays.clamp_min(1.0)

    rate = terminal_win_rate(train)
    predicted_first = rate[test.terminal_a] >= rate[test.terminal_b]
    terminal_only = float((predicted_first == (test.answer == 0)).float().mean())

    # A rule that ignores both operands can only bet on one slot; report what that
    # ceiling is, to confirm the answer slot itself carries no signal.
    predict_slot_zero = bool((train.answer == 0).float().mean() >= 0.5)
    slot_only = float((test.answer == (0 if predict_slot_zero else 1)).float().mean())
    return {
        "shortcut_terminal_only": terminal_only,
        "shortcut_slot_only": slot_only,
    }
