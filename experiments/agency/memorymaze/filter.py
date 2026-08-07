r"""Belief filters over a recorded Memory Maze stream.

A *filter* is not a policy. It never chooses an action and never sees a reward.
It replays a recorded trajectory, receiving at each step the action that was
actually taken and -- with probability :math:`1-\rho` -- the image that was
actually seen, and maintains a state that should encode what has been observed
so far. That is the object the Tensor Brain's measurement update claims to be:

.. math::
    q \leftarrow \alpha q + \beta a_k

is a recursive belief revision in log-odds coordinates, with :math:`\alpha` the
weight on the log-prior and :math:`a_k` an evidence term.

The schedule, per step, written out rather than hidden behind a runner:

.. code-block:: text

    q = evolve(q)                             # prediction step
    q = q + mu * enc(o_t)      if observed    # bottom-up evidence
    q = feedback(q, latent)                   # index feedback, variant under test
    q = alpha * q + a_{u_t}    if action      # control input (the Kalman `Bu`)

The evolution backend is the **QTB** one, :math:`q' = W\sigma(v_0 + V\sigma(q))`,
and that choice matters for whether any of this is a filter at all. QTB
evolution has no persistent context, so the whole state is ``q``: the map above
is exactly a predict-then-update recursion, with the evolution as the prediction
step and the writes as the update step. The original TB recurrence instead
carries a separate dynamic context ``h`` and regenerates ``q`` from it each
window, which makes ``q`` closer to a readout of the memory than to the belief
being filtered. Both are available; the study uses QTB.

Four choices for the feedback step, which is the thing under test:

``none``
    Skip it. Isolates what the index layer contributes at all.
``raw``
    ``measure``: draw :math:`k \sim p` and add :math:`a_k`. The paper's update.
``corrected``
    Add :math:`a_k - \sum_j p_j a_j`. The same draw with its mean removed, so
    repeated measurement cannot push ``q`` along :math:`\mathbb{E}_p[a]` until
    the CBS saturates. See ``TensorBrain.measure``.
``soft``
    ``attend``: add :math:`\sum_j p_j a_j` with no collapse. This is the control
    that asks whether the *discreteness* of the measurement earns anything, or
    whether ordinary attention would do.

The index feedback runs on masked steps too. It is driven by ``q`` alone and
needs no observation, so withholding it on masked steps would confound "no
observation" with "no internal update" -- and the internal-update-without-input
case is exactly the repeated-evolution regime the QTB describes.

The action write is teacher-forced to the recorded action, which is what makes
it an *intervention* rather than an inference: the outcome is known, not
generated. That is the sense in which it is a control input.

The index bank the observation feedback measures over is **unnamed**. Nothing
supervises it, so what it carves up is the model's own vocabulary, and whether
that vocabulary tracks anything real is measured afterwards by mutual
information against ground truth. Supervising it with target colours would put
the probe's answer into the training loss and make the whole study circular.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from experiments.agency.memorymaze.agent import PixelEncoder
from experiments.agency.memorymaze.env import ACTION_NAMES, IMAGE_SIDE
from tb import (
    IndexVocabulary,
    OriginalTBDynamicContext,
    QTBEvolution,
    ReLUEvolution,
    ScoreMode,
    TensorBrain,
)

FeedbackMode = Literal["none", "raw", "corrected", "soft"]
EvolutionName = Literal["original", "qtb", "relu", "none"]


def build_filter_vocabulary(num_latent: int) -> IndexVocabulary:
    """Two groups: an unnamed observation bank, and the six actions.

    The policy studies' colour and reward indices are absent on purpose. A
    filter has no instruction to receive and no reward to represent, and the
    perceptual labels those groups carried are derived from ground truth --
    which may appear in a probe, never in the filter's own vocabulary.
    """

    return IndexVocabulary.from_groups(
        {
            "latent": tuple(f"latent_{i:03d}" for i in range(num_latent)),
            "action": ACTION_NAMES,
        }
    )


@dataclass(frozen=True)
class FilterConfig:
    """Every controlled variable of a filter, in one place."""

    state_dim: int = 128
    hidden_dim: int = 128
    channels: int = 32
    # The observation index bank. 64 unnamed indices is roughly 6 bits per
    # measurement; the three-colour group the policy study used would have been
    # 2 bits, which would bottleneck the write for a reason that has nothing to
    # do with the architecture.
    num_latent_indices: int = 64
    # QTB rather than the original TB recurrence: it has no persistent context,
    # so `q` is the entire state and the schedule is a genuine predict-update
    # filter. See the module docstring.
    evolution: EvolutionName = "qtb"
    score_mode: ScoreMode = "direct"
    feedback: FeedbackMode = "raw"
    # `alpha`: the weight on the log-prior in the action write.
    action_retain_gate: float = 1.0
    action_write: bool = True
    view_gate: float = 1.0
    # Probability that a step's observation is withheld.
    mask_probability: float = 0.0
    cell: Literal["tb", "gru", "lstm"] = "tb"


def build_evolution(name: EvolutionName, state_dim: int, hidden_dim: int):
    if name == "none":
        return None
    if name == "original":
        return OriginalTBDynamicContext(state_dim, hidden_dim)
    if name == "qtb":
        return QTBEvolution(state_dim, hidden_dim)
    if name == "relu":
        return ReLUEvolution(state_dim, hidden_dim)
    raise ValueError(f"unknown evolution backend: {name}")


@dataclass
class StepTrace:
    """What one filter step produced, for the loss and for the diagnostics."""

    q: Float[Tensor, "batch state"]
    context: Float[Tensor, "batch context"] | None
    # The state a downstream reader sees: `sigma(q)` for a Tensor Brain, the
    # hidden state for a recurrent control. Probes and the decoder both read
    # this, so they read the same thing the architecture actually exposes.
    readout: Float[Tensor, "batch state"]
    # `None` when no index feedback ran.
    index_probabilities: Float[Tensor, "batch indices"] | None = None
    index_outcome: Int[Tensor, " batch"] | None = None


class TensorBrainFilter(nn.Module):
    """The Tensor Brain as a recursive belief filter."""

    def __init__(self, config: FilterConfig) -> None:
        super().__init__()
        self.config = config
        self.vocabulary = build_filter_vocabulary(config.num_latent_indices)
        self.encoder = PixelEncoder(config.state_dim, channels=config.channels)
        self.brain = TensorBrain(
            config.state_dim,
            len(self.vocabulary),
            build_evolution(config.evolution, config.state_dim, config.hidden_dim),
            score_mode=config.score_mode,
        )
        self.register_buffer("latent_bank", self.vocabulary.indices("latent"), persistent=False)
        self.register_buffer("action_bank", self.vocabulary.indices("action"), persistent=False)

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    def initial_state(
        self, batch: int, device: torch.device
    ) -> tuple[Float[Tensor, "batch state"], None]:
        return torch.zeros(batch, self.config.state_dim, device=device), None

    def step(
        self,
        q: Float[Tensor, "batch state"],
        context: Float[Tensor, "batch context"] | None,
        image: Float[Tensor, "batch observation"],
        action: Int[Tensor, " batch"],
        observed: Bool[Tensor, " batch"],
    ) -> StepTrace:
        config = self.config
        if self.brain.evolution is not None:
            q, context = self.brain.evolve(q, context)

        drive = self.encoder(image) * observed[:, None].to(q.dtype)
        q = self.brain.integrate_input(q, drive, input_gate=config.view_gate)

        probabilities: Tensor | None = None
        outcome: Tensor | None = None
        if config.feedback == "soft":
            q, probabilities = self.brain.attend(q, self.latent_bank)
        elif config.feedback in ("raw", "corrected"):
            q, outcome, probabilities = self.brain.measure(
                q,
                self.latent_bank,
                selection="sample",
                drift_correction=config.feedback == "corrected",
            )
        elif config.feedback != "none":
            raise ValueError(f"unknown feedback mode: {config.feedback}")

        if config.action_write:
            # Teacher-forced to the recorded action: the outcome is known, so
            # this is an intervention rather than a generated measurement.
            q, _, _ = self.brain.measure(
                q,
                self.action_bank,
                outcome=self.action_bank[action],
                selection="teacher",
                retain_gate=config.action_retain_gate,
            )
        return StepTrace(
            q=q,
            context=context,
            readout=torch.sigmoid(q),
            index_probabilities=probabilities,
            index_outcome=outcome,
        )


class RecurrentFilter(nn.Module):
    """GRU or LSTM control with the same encoder and the same information.

    The action reaches the control as a one-hot, and the mask reaches it as an
    explicit flag, so it is told exactly what the Tensor Brain is told. Omitting
    either would make it a weaker baseline rather than a comparable one.
    """

    def __init__(self, config: FilterConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = PixelEncoder(config.state_dim, channels=config.channels)
        self.action_projection = nn.Linear(len(ACTION_NAMES) + 1, config.state_dim)
        cell = nn.GRUCell if config.cell == "gru" else nn.LSTMCell
        self.recurrent = cell(config.state_dim, config.state_dim)

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @property
    def state_width(self) -> int:
        return self.config.state_dim * (2 if self.config.cell == "lstm" else 1)

    def initial_state(
        self, batch: int, device: torch.device
    ) -> tuple[Float[Tensor, "batch state"], None]:
        return torch.zeros(batch, self.state_width, device=device), None

    def step(
        self,
        q: Float[Tensor, "batch state"],
        context: Float[Tensor, "batch context"] | None,
        image: Float[Tensor, "batch observation"],
        action: Int[Tensor, " batch"],
        observed: Bool[Tensor, " batch"],
    ) -> StepTrace:
        del context
        flag = observed[:, None].to(q.dtype)
        onehot = torch.zeros(q.shape[0], len(ACTION_NAMES), device=q.device, dtype=q.dtype)
        onehot.scatter_(1, action[:, None], 1.0)
        drive = self.encoder(image) * flag + self.action_projection(
            torch.cat([onehot, flag], dim=1)
        )
        if self.config.cell == "lstm":
            hidden, memory = q.split(self.config.state_dim, dim=-1)
            hidden, memory = self.recurrent(
                torch.relu(drive), (hidden.contiguous(), memory.contiguous())
            )
            return StepTrace(
                q=torch.cat([hidden, memory], dim=-1), context=None, readout=hidden
            )
        hidden = self.recurrent(torch.relu(drive), q)
        return StepTrace(q=hidden, context=None, readout=hidden)


class FrameDecoder(nn.Module):
    """Reconstruct the current 64x64 frame from the filter's readout state.

    This is the whole Phase-1 objective. It contains no ground truth -- only
    pixels -- which is what keeps the Phase-2 probe honest: a probe for target
    positions is not reading back something the loss put there.

    On a masked step there is no image input, so reconstructing the frame is an
    act of imagination from the belief, and that is where the retention pressure
    comes from. At ``mask_probability=0`` the objective is a plain autoencoder
    and nothing needs to be remembered, which makes that setting the control
    rather than a wasted cell.
    """

    def __init__(self, state_dim: int, *, channels: int = 32) -> None:
        super().__init__()
        self.project = nn.Linear(state_dim, channels * 2 * 4 * 4)
        self.channels = channels
        self.body = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(channels * 2, channels * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(
        self, readout: Float[Tensor, "batch state"]
    ) -> Float[Tensor, "batch observation"]:
        seed = self.project(readout).reshape(-1, self.channels * 2, 4, 4)
        images = self.body(seed)
        return images.permute(0, 2, 3, 1).reshape(readout.shape[0], -1)


def build_filter(config: FilterConfig) -> TensorBrainFilter | RecurrentFilter:
    return TensorBrainFilter(config) if config.cell == "tb" else RecurrentFilter(config)


def observation_mask(
    batch: int, probability: float, generator: torch.Generator, device: torch.device
) -> Bool[Tensor, " batch"]:
    """``True`` where the observation is delivered."""

    if probability <= 0.0:
        return torch.ones(batch, dtype=torch.bool, device=device)
    draw = torch.rand(batch, generator=generator, device=generator.device)
    return (draw >= probability).to(device)


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Total and post-perception parameter counts.

    The encoder is identical across architectures, so the post-perception count
    is where any capacity difference actually sits, and it is the number the
    comparison should be judged on.
    """

    total = sum(p.numel() for p in module.parameters())
    encoder = sum(p.numel() for p in module.encoder.parameters())
    return {"total": total, "encoder": encoder, "post_perception": total - encoder}


assert IMAGE_SIDE == 64, "the decoder's four stride-2 stages assume a 64-pixel frame"
