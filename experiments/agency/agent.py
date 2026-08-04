"""The Tensor Brain agent: one concept-window cycle per environment step.

The schedule below is the whole scientific content of this module and is written
out in paper order rather than hidden behind a runner:

.. code-block:: text

    q, context = evolve(q, context)              # Algorithm 1
    q = integrate_input(q, view_drive)           # Algorithm 2 / Eq. 46 (perception)
    q = integrate_input(q, reward_drive)         # Eq. 46 (reward module)
    q = q + a_cue_color + a_cue_shape            # Eq. 47 (top-down instruction)
    q, k_color = measure(q, percept_color)       # Algorithm 3 (perceptual naming)
    q, k_shape = measure(q, percept_shape)       # Algorithm 3 (perceptual naming)
    v = index_scores(q, reward)                  # internal reward function
    q, k_action = measure(q, action)             # Algorithm 3 -- the *action* index

The action measurement is the same operation as the perceptual ones with a
different candidate group. That identity is the concrete form of the QTB claim
that "actions are generated as any other indices".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from experiments.agency.gridworld import GridConfig
from experiments.agency.vocabulary import NOTHING, build_vocabulary
from tb import (
    IndexVocabulary,
    OriginalTBDynamicContext,
    QTBEvolution,
    ReLUEvolution,
    ScoreMode,
    TensorBrain,
)
from tb.vocabulary import get_candidate_positions

EvolutionName = Literal["original", "qtb", "relu", "none"]
CueMode = Literal["persistent", "initial", "none"]
CriticName = Literal["reward-index", "linear", "none"]


@dataclass(frozen=True)
class AgentConfig:
    """Every controlled variable of the agent, in one place."""

    state_dim: int = 64
    hidden_dim: int = 64
    evolution: EvolutionName = "original"
    score_mode: ScoreMode = "direct"
    # Measurement gates. (1, 1) is HB-POVM, (0, 1) is neural PVM, (1, 0) removes
    # index feedback and leaves a generative RNN.
    action_retain_gate: float = 1.0
    action_feedback_gate: float = 1.0
    percept_retain_gate: float = 1.0
    percept_feedback_gate: float = 1.0
    measure_percepts: bool = True
    cue_mode: CueMode = "persistent"
    view_gate: float = 1.0
    reward_gate: float = 1.0
    critic: CriticName = "reward-index"
    # "planned" is an evaluation-time readout: it imagines each action's
    # consequence through the evolution operator and picks the one the internal
    # reward function likes best. It is not differentiable and is never used for
    # training, exactly as P-Samp is an evaluation readout of a P-SA checkpoint
    # in the PVSG experiments.
    action_selection: Literal["sample", "argmax", "planned"] = "sample"
    # Number of concept windows executed per environment step. Depth in the
    # Tensor Brain comes from the evolution operator: within one window the map
    # from `q` to index scores is a single sigmoid layer, so a one-window agent
    # is a perceptron over `sigma(q)`. QTB Section 13.5.2 describes exactly this
    # remedy -- iterated evolution with indices activated in between.
    deliberation_windows: int = 1
    deliberation_mode: Literal["none", "attend", "measure"] = "attend"
    # Control for the paper's shared bidirectional matrix: when true, top-down
    # feedback uses a second, independently trained matrix.
    decouple_feedback: bool = False


@dataclass
class WindowTrace:
    """Everything one concept-window cycle produced, for loss and diagnostics."""

    q: Float[Tensor, "envs state"]
    context: Float[Tensor, "envs context"] | None
    action_index: Int[Tensor, " envs"]
    action_position: Int[Tensor, " envs"]
    action_probabilities: Float[Tensor, "envs actions"]
    value: Float[Tensor, " envs"]
    percept_color_probabilities: Float[Tensor, "envs colors"] | None
    percept_shape_probabilities: Float[Tensor, "envs shapes"] | None
    percept_color_index: Int[Tensor, " envs"] | None
    percept_shape_index: Int[Tensor, " envs"] | None


def build_evolution(
    name: EvolutionName, state_dim: int, hidden_dim: int
) -> OriginalTBDynamicContext | QTBEvolution | ReLUEvolution | None:
    """Construct one named evolution backend, or ``None`` for the ablation."""

    if name == "original":
        return OriginalTBDynamicContext(state_dim, hidden_dim)
    if name == "qtb":
        return QTBEvolution(state_dim, hidden_dim)
    if name == "relu":
        return ReLUEvolution(state_dim, hidden_dim)
    if name == "none":
        return None
    raise ValueError(f"unknown evolution backend: {name}")


class DecoupledTensorBrain(TensorBrain):
    """A Tensor Brain whose top-down feedback uses a *second* matrix.

    This is the control for the paper's central architectural claim. In the
    Tensor Brain the same column ``A[:, k]`` is the bottom-up readout weight and
    the top-down embedding injected on measurement; ``docs/fidelity.md`` records
    that the direct feedback path must stay ``A``. Nothing in a policy-gradient
    objective forces that sharing, so the honest question is what it buys.

    Here scoring still uses ``A`` while measurement injects ``A_feedback[:, k]``,
    an independently initialized and independently trained matrix of the same
    shape. Everything else -- gates, score mode, evolution, schedule, parameter
    count up to the extra matrix -- is unchanged.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.A_feedback = nn.Parameter(torch.empty_like(self.A))
        nn.init.normal_(self.A_feedback, mean=0.0, std=self.state_dim**-0.5)

    def measure(self, q, candidates=None, *, outcome=None, selection="sample",
                retain_gate=1.0, feedback_gate=1.0):
        # Reproduces TensorBrain.measure with one substitution, kept explicit
        # rather than factored so the single difference stays visible.
        indices = self._resolve_candidate_indices(candidates, q.device)
        probabilities = torch.softmax(self.index_scores(q, indices), dim=-1)
        if selection == "teacher":
            outcome_index = torch.as_tensor(outcome, dtype=torch.long, device=q.device)
            if outcome_index.ndim == 0:
                outcome_index = outcome_index.expand(q.shape[:-1])
        elif selection == "sample":
            outcome_index = indices[torch.distributions.Categorical(probabilities).sample()]
        elif selection == "argmax":
            outcome_index = indices[probabilities.argmax(dim=-1)]
        else:
            raise ValueError(f"unsupported selection mode: {selection}")
        outcome_embedding = self.A_feedback.T[outcome_index]
        return retain_gate * q + feedback_gate * outcome_embedding, outcome_index, probabilities

    def attend(self, q, candidates=None):
        indices = self._resolve_candidate_indices(candidates, q.device)
        probabilities = torch.softmax(self.index_scores(q, indices), dim=-1)
        return q + probabilities @ self.A_feedback[:, indices].T, probabilities


class GridAgent(nn.Module):
    """A Tensor Brain wired to a gridworld through action indices.

    The Tensor Brain core is untouched. This class owns only the parts the
    papers place *outside* it: the perceptual mapping ``g(nu)``, the reward
    module's input drive, and the environment coupling of the action index.
    """

    def __init__(self, grid: GridConfig, config: AgentConfig) -> None:
        super().__init__()
        self.grid = grid
        self.config = config
        self.vocabulary: IndexVocabulary = build_vocabulary(grid)
        brain_type = DecoupledTensorBrain if config.decouple_feedback else TensorBrain
        self.brain = brain_type(
            config.state_dim,
            len(self.vocabulary),
            build_evolution(config.evolution, config.state_dim, config.hidden_dim),
            score_mode=config.score_mode,
        )
        # g(nu) for the visual module: the dimensional part of input integration
        # lives in the experiment, not in the Tensor Brain.
        self.view_projection = nn.Linear(grid.observation_dim, config.state_dim)
        # The reward module writes a single scaled direction into the CBS.
        self.reward_projection = nn.Linear(1, config.state_dim, bias=False)
        self.value_head = (
            nn.Linear(config.state_dim, 1) if config.critic == "linear" else None
        )
        self.register_buffer(
            "action_indices", self.vocabulary.indices("action"), persistent=False
        )
        self.register_buffer(
            "percept_color_indices", self.vocabulary.indices("percept_color"), persistent=False
        )
        self.register_buffer(
            "percept_shape_indices", self.vocabulary.indices("percept_shape"), persistent=False
        )
        self.register_buffer(
            "color_indices", self.vocabulary.indices("color"), persistent=False
        )
        self.register_buffer(
            "shape_indices", self.vocabulary.indices("shape"), persistent=False
        )
        self.register_buffer(
            "reward_indices", self.vocabulary.indices("reward"), persistent=False
        )
        self.nothing_index = self.vocabulary.index(NOTHING)

    # -------------------------------------------------------------- utilities

    def initial_state(
        self, num_envs: int, device: torch.device
    ) -> tuple[Float[Tensor, "envs state"], None]:
        return torch.zeros(num_envs, self.config.state_dim, device=device), None

    def percept_targets(
        self,
        visible_slot: Int[Tensor, " envs"],
        object_color: Int[Tensor, "envs objects"],
        object_shape: Int[Tensor, "envs objects"],
    ) -> tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]]:
        """Ground-truth perceptual labels for the attended region of interest."""

        visible = visible_slot >= 0
        safe_slot = visible_slot.clamp_min(0)[:, None]
        color = self.color_indices[object_color.gather(1, safe_slot).squeeze(1)]
        shape = self.shape_indices[object_shape.gather(1, safe_slot).squeeze(1)]
        nothing = torch.full_like(color, self.nothing_index)
        return torch.where(visible, color, nothing), torch.where(visible, shape, nothing)

    def value_of(self, q: Float[Tensor, "envs state"]) -> Float[Tensor, " envs"]:
        r"""Internal reward function.

        ``reward-index`` reads the score of the ``reward_positive`` index,
        :math:`v(q) = a_{0,r} + a_r^\top \sigma(q)`. This is the cheapest
        Tensor-Brain-native realization of QTB Section 13.5.2's "internal reward
        function": one extra column of the same matrix ``A``.
        """

        if self.config.critic == "reward-index":
            return self.brain.index_scores(q, self.reward_indices).squeeze(-1)
        if self.config.critic == "linear":
            assert self.value_head is not None
            return self.value_head(q).squeeze(-1)
        return torch.zeros(q.shape[:-1], device=q.device)

    def _measure_percept(
        self,
        q: Float[Tensor, "envs state"],
        group_indices: Int[Tensor, " indices"],
        teacher: tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]] | None,
        slot: int,
    ) -> tuple[Float[Tensor, "envs state"], Int[Tensor, " envs"], Float[Tensor, "envs indices"]]:
        """One perceptual measurement, teacher-forced when labels are supplied."""

        return self.brain.measure(
            q,
            group_indices,
            outcome=teacher[slot] if teacher is not None else None,
            selection="teacher" if teacher is not None else "sample",
            retain_gate=self.config.percept_retain_gate,
            feedback_gate=self.config.percept_feedback_gate,
        )

    def imagined_action_values(
        self,
        q: Float[Tensor, "envs state"],
        context: Float[Tensor, "envs context"] | None,
    ) -> Float[Tensor, "envs actions"]:
        r"""Score each action by imagining its consequence.

        QTB Section 13.5.2 describes near-term planning as iterated application
        of the evolution operator with indices activated in between, and the
        resulting states evaluated "through their interaction with internal
        reward functions". This is the one-step form of exactly that:

        .. math::
            \hat v(a) = v\big(\text{evolve}(q + a_a,\ h)\big).

        The imagined evolution's context is discarded, so imagining commits the
        agent to nothing. No new parameters are involved: the action embedding,
        the evolution operator and the reward index are all already trained.
        """

        if self.brain.evolution is None:
            raise RuntimeError("planning requires an evolution operator")
        scores = []
        for action_index in self.action_indices:
            imagined, _ = self.brain.evolve(q + self.brain.A.T[action_index], context)
            scores.append(self.value_of(imagined))
        return torch.stack(scores, dim=-1)

    def plan_action(
        self,
        q: Float[Tensor, "envs state"],
        context: Float[Tensor, "envs context"] | None,
    ) -> Int[Tensor, " envs"]:
        """Return the action index whose imagined outcome scores highest."""

        return self.action_indices[self.imagined_action_values(q, context).argmax(dim=-1)]

    # ------------------------------------------------------------ the schedule

    def window_cycle(
        self,
        q: Float[Tensor, "envs state"],
        context: Float[Tensor, "envs context"] | None,
        observation: Float[Tensor, "envs observation"],
        previous_reward: Float[Tensor, " envs"],
        cue_color: Int[Tensor, " envs"],
        cue_shape: Int[Tensor, " envs"],
        *,
        is_first_step: Bool[Tensor, " envs"] | None = None,
        action_teacher: Int[Tensor, " envs"] | None = None,
        percept_teacher: tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]] | None = None,
    ) -> WindowTrace:
        """Run one environment step's worth of concept windows for the batch.

        With ``deliberation_windows == 1`` this is the reactive schedule written
        in the module docstring. With more windows, the leading windows are
        internal deliberation: evolution runs again, perception and the
        instruction stay gated in, and an index is activated in between, which
        is QTB Section 13.5.2's chain-of-thought reading of near-term planning.
        """

        config = self.config
        color_probabilities: Tensor | None = None
        shape_probabilities: Tensor | None = None
        color_index: Tensor | None = None
        shape_index: Tensor | None = None

        for window in range(config.deliberation_windows):
            is_action_window = window == config.deliberation_windows - 1

            # --- window boundary: Algorithm 1 ----------------------------
            if self.brain.evolution is not None:
                q, context = self.brain.evolve(q, context)

            # --- Algorithm 2 / Equation 46: gated inputs from modules ----
            q = self.brain.integrate_input(
                q, self.view_projection(observation), input_gate=config.view_gate
            )
            q = self.brain.integrate_input(
                q,
                self.reward_projection(previous_reward[:, None]),
                input_gate=config.reward_gate,
            )

            # --- Equation 47: top-down instruction indices ---------------
            if config.cue_mode != "none":
                cue_drive = self.brain.A.T[cue_color] + self.brain.A.T[cue_shape]
                if config.cue_mode == "initial":
                    assert is_first_step is not None
                    cue_drive = cue_drive * is_first_step[:, None].float()
                q = q + cue_drive

            if window == 0 and config.measure_percepts:
                # --- Algorithm 3, perceptual naming of the attended region
                q, color_index, color_probabilities = self._measure_percept(
                    q, self.percept_color_indices, percept_teacher, 0
                )
                q, shape_index, shape_probabilities = self._measure_percept(
                    q, self.percept_shape_indices, percept_teacher, 1
                )

            if not is_action_window:
                # --- deliberation: activate an index over the whole layer
                if config.deliberation_mode == "attend":
                    q, _ = self.brain.attend(q)
                elif config.deliberation_mode == "measure":
                    q, _, _ = self.brain.measure(q, selection="sample")

        # --- internal reward function, read before acting -----------------
        value = self.value_of(q)

        # --- Algorithm 3, the action index: this one changes the world ----
        outcome = action_teacher
        if action_teacher is not None:
            selection = "teacher"
        elif config.action_selection == "planned":
            selection, outcome = "teacher", self.plan_action(q, context)
        else:
            selection = config.action_selection
        q, action_index, action_probabilities = self.brain.measure(
            q,
            self.action_indices,
            outcome=outcome,
            selection=selection,
            retain_gate=config.action_retain_gate,
            feedback_gate=config.action_feedback_gate,
        )
        action_position = get_candidate_positions(self.action_indices, action_index)
        return WindowTrace(
            q=q,
            context=context,
            action_index=action_index,
            action_position=action_position,
            action_probabilities=action_probabilities,
            value=value,
            percept_color_probabilities=color_probabilities,
            percept_shape_probabilities=shape_probabilities,
            percept_color_index=color_index,
            percept_shape_index=shape_index,
        )

    def reset_finished(
        self,
        q: Float[Tensor, "envs state"],
        context: Float[Tensor, "envs context"] | None,
        done: Bool[Tensor, " envs"],
    ) -> tuple[Float[Tensor, "envs state"], Float[Tensor, "envs context"] | None]:
        """Clear the cognitive state of environments whose episode just ended.

        Carrying a CBS across an episode boundary would leak the previous
        layout into the next one, so a finished episode starts from ``q = 0``
        exactly as the first step of a run does.
        """

        keep = (~done)[:, None].float()
        q = q * keep
        if context is not None:
            context = context * keep
        return q, context
