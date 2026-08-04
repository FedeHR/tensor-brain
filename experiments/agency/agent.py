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
class IndexLayout:
    """Which vocabulary groups play which role in the concept-window schedule.

    The schedule itself is task independent: two cue factors are injected
    top-down, the same two factors are named bottom-up over candidate groups
    that additionally contain an "unobserved" label, one group holds the action
    indices, and one holds the reward index. Only the *names* differ between the
    gridworld (colour/shape) and MiniGrid (colour/object).
    """

    percept_factors: tuple[str, str]
    action: str = "action"
    reward: str = "reward"


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
    # QTB Equation 46 gates every input source separately, `g(nu) = sum_k mu_k
    # g(nu_k)`, and the paper leaves the gates to the experiment. They matter:
    # the top-down instruction enters as a column of `A`, whose norm is fixed by
    # initialization, while the perceptual drive's scale is whatever the encoder
    # produces. On MiniGrid the encoder drive is 6x the cue at initialization and
    # 28x after training, so the instruction is drowned unless it is gated up or
    # the drive is normalized.
    cue_gate: float = 1.0
    learn_cue_gate: bool = False
    normalize_drive: bool = False
    # `alpha` is the weight on the log-prior in the measurement update
    # `q <- alpha q + beta a_k`. Making it learnable lets an experiment ask what
    # prior weight an environment's volatility actually calls for.
    learn_action_retain_gate: bool = False


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


class TensorBrainAgent(nn.Module):
    """A Tensor Brain wired to an environment through action indices.

    The Tensor Brain core is untouched. This class owns only the parts the
    papers place *outside* it: the perceptual mapping ``g(nu)`` supplied as an
    ``encoder`` module, the reward module's input drive, and the environment
    coupling of the action index. Subclasses build the vocabulary and encoder
    for a concrete environment; the concept-window schedule lives here once.
    """

    def __init__(
        self,
        vocabulary: IndexVocabulary,
        encoder: nn.Module,
        layout: IndexLayout,
        config: AgentConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.layout = layout
        self.vocabulary = vocabulary
        brain_type = DecoupledTensorBrain if config.decouple_feedback else TensorBrain
        self.brain = brain_type(
            config.state_dim,
            len(vocabulary),
            build_evolution(config.evolution, config.state_dim, config.hidden_dim),
            score_mode=config.score_mode,
        )
        # g(nu): the dimensional part of input integration lives in the
        # experiment, not in the Tensor Brain.
        self.encoder = encoder
        # The reward module writes a single scaled direction into the CBS.
        self.reward_projection = nn.Linear(1, config.state_dim, bias=False)
        self.value_head = (
            nn.Linear(config.state_dim, 1) if config.critic == "linear" else None
        )
        self.cue_gate = (
            nn.Parameter(torch.tensor(float(config.cue_gate)))
            if config.learn_cue_gate
            else None
        )
        self.action_retain_gate = (
            nn.Parameter(torch.tensor(float(config.action_retain_gate)))
            if config.learn_action_retain_gate
            else None
        )
        # Buffers use internal names so that `action_indices` can stay a
        # readable public property without shadowing its own storage.
        self.register_buffer(
            "action_bank", vocabulary.indices(layout.action), persistent=False
        )
        self.register_buffer(
            "reward_bank", vocabulary.indices(layout.reward), persistent=False
        )
        for slot, group in enumerate(layout.percept_factors):
            self.register_buffer(
                f"percept_bank_{slot}", vocabulary.indices(group), persistent=False
            )

    @property
    def action_indices(self) -> Int[Tensor, " indices"]:
        return self.action_bank

    @property
    def reward_indices(self) -> Int[Tensor, " indices"]:
        return self.reward_bank

    def percept_bank(self, slot: int) -> Int[Tensor, " indices"]:
        return getattr(self, f"percept_bank_{slot}")

    # -------------------------------------------------------------- utilities

    def initial_state(
        self, num_envs: int, device: torch.device
    ) -> tuple[Float[Tensor, "envs state"], None]:
        return torch.zeros(num_envs, self.config.state_dim, device=device), None

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
            drive = self.encoder(observation)
            if config.normalize_drive:
                # The repository's PVSG convention: `sqrt(D) * L2-normalize(x)`,
                # so every nonzero input has component RMS one. Here it puts the
                # perceptual drive on the same scale as an index embedding.
                drive = drive * (config.state_dim**0.5) / drive.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
            q = self.brain.integrate_input(q, drive, input_gate=config.view_gate)
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
                gate = self.cue_gate if self.cue_gate is not None else config.cue_gate
                q = q + gate * cue_drive

            if window == 0 and config.measure_percepts:
                # --- Algorithm 3, perceptual naming of the attended region
                q, color_index, color_probabilities = self._measure_percept(
                    q, self.percept_bank(0), percept_teacher, 0
                )
                q, shape_index, shape_probabilities = self._measure_percept(
                    q, self.percept_bank(1), percept_teacher, 1
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
            retain_gate=(
                self.action_retain_gate
                if self.action_retain_gate is not None
                else config.action_retain_gate
            ),
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


def _remap_legacy_encoder_keys(state_dict, prefix, *_arguments) -> None:
    """Accept checkpoints that still call the encoder ``view_projection``."""

    for suffix in ("weight", "bias"):
        legacy = f"{prefix}view_projection.{suffix}"
        if legacy in state_dict:
            state_dict[f"{prefix}encoder.{suffix}"] = state_dict.pop(legacy)


class GridAgent(TensorBrainAgent):
    """The Tensor Brain agent for the symbolic-foraging gridworld."""

    def __init__(self, grid: GridConfig, config: AgentConfig) -> None:
        vocabulary = build_vocabulary(grid)
        super().__init__(
            vocabulary,
            nn.Linear(grid.observation_dim, config.state_dim),
            IndexLayout(percept_factors=("percept_color", "percept_shape")),
            config,
        )
        self.grid = grid
        # Checkpoints written before `view_projection` was renamed to `encoder`
        # must keep loading, because the gridworld study's figures are
        # regenerated from them.
        self._register_load_state_dict_pre_hook(_remap_legacy_encoder_keys)
        self.register_buffer(
            "color_indices", vocabulary.indices("color"), persistent=False
        )
        self.register_buffer(
            "shape_indices", vocabulary.indices("shape"), persistent=False
        )
        self.nothing_index = vocabulary.index(NOTHING)

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
