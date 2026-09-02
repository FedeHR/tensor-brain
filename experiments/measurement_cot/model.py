r"""A Tensor Brain reasoning chain whose only free choice is how it measures.

One query is processed as a sequence of concept windows:

1. the start concept and both candidate terminals are written into the pre-CBS
   with :meth:`tb.TensorBrain.integrate_input`, which is the paper's
   :math:`q \leftarrow q + \mu\,g(\nu)` with the index embeddings themselves as
   the input drive;
2. for hop :math:`h = 1 \dots H` the evolution operator advances the state, and
   for :math:`h < H` a collapse step writes index feedback back into it;
3. the answer is read out as the terminal-layer index scores restricted to the
   two candidates.

The candidate set at hop ``h`` is layer ``h``, which is the ordinary Tensor Brain
notion of a concept group restricting a measurement, and is identical across all
conditions. Nothing else varies between conditions except the
:class:`~experiments.measurement_cot.collapse.CollapseSpec` at each hop, so any
difference in behaviour is attributable to the measurement and not to capacity,
supervision signal, or number of evolution steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from experiments.measurement_cot.collapse import CollapseSpec, collapse_weights, index_entropy
from experiments.measurement_cot.data import QuerySet
from experiments.measurement_cot.graph import LayeredDAG
from tb import TensorBrain
from tb.evolution import Evolution, QTBEvolution, ReLUEvolution


@dataclass
class ChainTrace:
    """Everything an analysis needs from one forward pass."""

    logits: Float[Tensor, "queries 2"]
    step_probabilities: list[Float[Tensor, "queries candidates"]] = field(default_factory=list)
    step_entropies: list[Float[Tensor, " queries"]] = field(default_factory=list)
    pre_feedback_state: list[Float[Tensor, "queries state"]] = field(default_factory=list)
    post_feedback_state: list[Float[Tensor, "queries state"]] = field(default_factory=list)


def frontier_distribution(
    graph: LayeredDAG, queries: QuerySet, hop: int, device: torch.device
) -> Float[Tensor, "queries candidates"]:
    """Uniform distribution over the layer-``hop`` nodes reachable from each start."""

    mask = graph.frontier_masks[hop].to(device)[queries.start_position]
    return mask.float() / mask.sum(dim=-1, keepdim=True).clamp_min(1)


def build_evolution(kind: str, state_dim: int, hidden_dim: int) -> Evolution:
    if kind == "qtb":
        return QTBEvolution(state_dim, hidden_dim)
    if kind == "relu":
        return ReLUEvolution(state_dim, hidden_dim)
    raise ValueError(f"unknown evolution kind: {kind}")


class MeasurementChain(nn.Module):
    """Tensor Brain reachability reasoner with a configurable measurement schedule."""

    def __init__(
        self,
        graph: LayeredDAG,
        *,
        state_dim: int = 128,
        hidden_dim: int = 256,
        evolution: str = "qtb",
        score_mode: str = "centered",
        retain_gate: float = 1.0,
        feedback_gate: float = 1.0,
        feedback_norm: str = "unit",
        learn_index_bank: bool = False,
        write_targets: bool = True,
        learn_input_gates: bool = True,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.num_hops = graph.spec.num_hops
        self.retain_gate = retain_gate
        # Index embeddings have unit norm spread over `state_dim` components, so
        # writing one raw into the pre-CBS moves each component by about
        # state_dim^-1/2 and leaves sigmoid in its linear region around 0.5. Every
        # write is therefore scaled to a unit per-component pre-CBS magnitude,
        # which is the same requirement the PVSG experiment meets by RMS
        # normalizing its visual features.
        self.write_scale = float(state_dim**0.5)
        self.feedback_gate = feedback_gate
        if feedback_norm not in ("raw", "unit"):
            raise ValueError("feedback_norm must be 'raw' or 'unit'")
        self.feedback_norm = feedback_norm
        self.tb = TensorBrain(
            state_dim=state_dim,
            num_indices=graph.spec.num_nodes,
            evolution=build_evolution(evolution, state_dim, hidden_dim),
            score_mode=score_mode,
        )
        # The index bank is a fixed random code by default. A trainable bank of
        # this size can simply store (start, terminal) associations and answer
        # without ever traversing the graph, which is exactly the confound that
        # makes every collapse condition look identical. Freezing it puts all task
        # knowledge in the evolution operator and leaves superposition readout as
        # the only route to the answer.
        self.learn_index_bank = learn_index_bank
        self.tb.A.requires_grad_(learn_index_bank)
        self.write_targets = write_targets
        # Two scalar input gates, one per role in the query. These are the paper's
        # mu; keeping them scalar means the query is written into the workspace
        # with index embeddings and nothing else.
        self.start_gate = nn.Parameter(torch.ones(()), requires_grad=learn_input_gates)
        self.target_gate = nn.Parameter(torch.ones(()), requires_grad=learn_input_gates)
        # The pause-token control needs a feedback direction that carries no
        # information about the candidate distribution.
        self.pause_vector = nn.Parameter(torch.zeros(state_dim))
        nn.init.normal_(self.pause_vector, std=state_dim**-0.5)
        self.register_buffer(
            "layer_candidates",
            torch.stack([indices for indices in graph.layer_indices[1:]])
            if len({len(i) for i in graph.layer_indices[1:]}) == 1
            else torch.empty(0),
            persistent=False,
        )

    def candidates_at(self, hop: int) -> Int[Tensor, " candidates"]:
        return self.graph.layer_indices[hop].to(self.tb.A.device)

    def write_query(self, queries: QuerySet) -> Float[Tensor, "queries state"]:
        """Write start and both candidate terminals into a zero pre-CBS."""

        embeddings = self.tb.A.T
        q = embeddings.new_zeros(len(queries), self.tb.state_dim)
        q = self.tb.integrate_input(
            q, embeddings[queries.start], input_gate=self.start_gate * self.write_scale
        )
        if self.write_targets:
            # Both terminals enter symmetrically, so the question is "which of
            # these two", and neither slot is distinguishable from the write
            # alone. Turning this off leaves a pure forward search whose answer is
            # only read out at the end.
            both_targets = embeddings[queries.terminal_a] + embeddings[queries.terminal_b]
            q = self.tb.integrate_input(
                q, both_targets, input_gate=self.target_gate * self.write_scale
            )
        return q

    def answer_logits(
        self, q: Float[Tensor, "queries state"], queries: QuerySet
    ) -> Float[Tensor, "queries 2"]:
        """Score the two candidate terminals with the shared index bank."""

        gamma = torch.sigmoid(q)
        if self.tb.score_mode == "centered":
            gamma = gamma - 0.5
        embeddings = self.tb.A.T
        bias = self.tb.index_bias()
        logit_a = (gamma * embeddings[queries.terminal_a]).sum(-1) + bias[queries.terminal_a]
        logit_b = (gamma * embeddings[queries.terminal_b]).sum(-1) + bias[queries.terminal_b]
        return torch.stack([logit_a, logit_b], dim=-1)

    def forward(
        self,
        queries: QuerySet,
        schedule: list[CollapseSpec],
        *,
        record: bool = False,
        generator: torch.Generator | None = None,
    ) -> ChainTrace:
        """Run the chain under a per-hop measurement schedule."""

        if len(schedule) != self.num_hops - 1:
            raise ValueError(
                f"schedule must give one spec per intermediate hop ({self.num_hops - 1}), "
                f"got {len(schedule)}"
            )

        q = self.write_query(queries)
        context: Tensor | None = None
        trace = ChainTrace(logits=q.new_zeros(len(queries), 2))

        for hop in range(1, self.num_hops + 1):
            q, context = self.tb.evolve(q, context)
            if hop == self.num_hops:
                break
            spec = schedule[hop - 1]
            candidates = self.candidates_at(hop)
            scores = self.tb.index_scores(q, candidates)
            teacher = None
            if spec.mode == "teacher":
                teacher = frontier_distribution(self.graph, queries, hop, scores.device)
            weights, probabilities = collapse_weights(
                scores, spec, teacher_distribution=teacher, generator=generator
            )
            if record:
                # Recorded live, not detached: the step supervision in
                # `train.py` differentiates through these states.
                trace.step_probabilities.append(probabilities)
                trace.step_entropies.append(index_entropy(probabilities))
                trace.pre_feedback_state.append(q)

            if spec.mode == "pause":
                feedback = self.pause_vector.expand_as(q)
            elif spec.mode == "none":
                feedback = torch.zeros_like(q)
            else:
                feedback = weights @ self.tb.A[:, candidates].T
                if self.feedback_norm == "unit":
                    # A flat weight vector averages many embeddings towards zero,
                    # so soft feedback is intrinsically smaller in norm than a
                    # one-hot collapse. Rescaling to a common norm separates the
                    # effect of measurement sharpness from the effect of how hard
                    # the step pushes on the state; `raw` keeps the confound and
                    # is reported alongside.
                    feedback = feedback / feedback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                feedback = feedback * self.write_scale
            q = self.retain_gate * q + self.feedback_gate * feedback
            if record:
                trace.post_feedback_state.append(q)

        trace.logits = self.answer_logits(q, queries)
        return trace


def uniform_schedule(spec: CollapseSpec, num_hops: int) -> list[CollapseSpec]:
    """The same collapse at every intermediate hop."""

    return [spec] * (num_hops - 1)
