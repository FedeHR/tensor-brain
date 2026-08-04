r"""A foraging task with unreliable perception and a volatile target.

This is the environment for the two experiments that test the Tensor Brain's
*mechanism* rather than its benchmark score. QTB writes the pre-CBS as
:math:`q = \operatorname{logit}(\gamma)`, so ``q`` is a vector of log-odds over
factorized Bernoulli latents, and Section 10 notes that the HB-POVM update is a
PVM update plus a skip connection, "providing an interpretation of skip
connections as logit priors". The measurement update therefore decomposes as

.. math::
    q \leftarrow \underbrace{\alpha q}_{\text{log-prior}}
                 + \underbrace{\beta a_k}_{\text{log-likelihood}},

which is a Bayes filter step for factorized Bernoulli latents with ``alpha`` as
the weight on the prior.

Nothing in the earlier studies exercised that. Their observations were exact, so
a single glance was sufficient and there was no evidence to accumulate; the
Bayes-filter form had nothing to do and the agent reduced to a slower recurrent
policy. Two knobs fix that:

``observation_noise`` (E1)
    Each *reported* attribute of a visible object is resampled uniformly with
    probability ``epsilon``, independently per attribute, per object and per
    step. Repeated glances are then independent draws, so identifying the cued
    object requires accumulating log-likelihood ratios -- exactly the form of
    ``q <- q + a_k``.

``hazard_rate`` (E2)
    With probability ``h`` per step the target switches to a different object
    (and the cue changes with it). Evidence accumulated about which object was
    the target goes stale at rate ``h``, so the optimal weight on the prior
    falls as ``h`` rises. ``alpha = 1`` (HB-POVM) should be best when ``h = 0``
    and ``alpha = 0`` (neural PVM) should improve as ``h`` grows.

The exact Bayes posterior over which object is the target is computable here in
closed form, which makes calibration a measurable quantity rather than a story.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from experiments.agency.gridworld import GridConfig, StepResult, SymbolicForaging


@dataclass(frozen=True)
class NoisyConfig:
    """The two knobs added on top of an ordinary :class:`GridConfig`."""

    observation_noise: float = 0.0
    hazard_rate: float = 0.0


NOISELESS = NoisyConfig()


class NoisyForaging(SymbolicForaging):
    """Symbolic foraging with a corrupted sensor and an optionally moving target."""

    def __init__(
        self,
        config: GridConfig,
        num_envs: int,
        *,
        noisy: NoisyConfig = NOISELESS,
        **arguments,
    ) -> None:
        self.noisy = noisy
        self._pinned: tuple[Tensor, Tensor] | None = None
        super().__init__(config, num_envs, **arguments)
        self._reset_evidence()

    # ------------------------------------------------------------- evidence

    def _reset_evidence(self) -> None:
        """Counts of reported attribute values, per object. Sufficient statistics.

        Because corruption is resampled independently each step, the exact
        posterior depends on the observation history only through these counts.
        """

        shape = (self.num_envs, self.config.num_objects)
        self.color_counts = torch.zeros(*shape, self.config.num_colors, device=self.device)
        self.shape_counts = torch.zeros(*shape, self.config.num_shapes, device=self.device)

    def reset(self) -> None:
        super().reset()
        self._reset_evidence()

    def observed_attributes(
        self,
    ) -> tuple[Int[Tensor, "envs objects"], Int[Tensor, "envs objects"]]:
        """Corrupt each reported attribute with probability ``observation_noise``.

        A corrupted reading is drawn uniformly over the attribute's values, so
        the reading is still informative -- it agrees with the truth with
        probability ``1 - eps + eps / K`` -- but no single glance is decisive.
        """

        state = self.state
        if self._pinned is not None:
            # A reading has already been drawn for this step; the renderer must
            # see the same one rather than resampling the corruption.
            return self._pinned
        epsilon = self.noisy.observation_noise
        if epsilon <= 0.0:
            return state.object_color, state.object_shape
        shape = state.object_color.shape
        corrupt_color = (
            torch.rand(shape, generator=self.generator, device=self.device) < epsilon
        )
        corrupt_shape = (
            torch.rand(shape, generator=self.generator, device=self.device) < epsilon
        )
        random_color = torch.randint(
            self.config.num_colors, shape, generator=self.generator, device=self.device
        )
        random_shape = torch.randint(
            self.config.num_shapes, shape, generator=self.generator, device=self.device
        )
        return (
            torch.where(corrupt_color, random_color, state.object_color),
            torch.where(corrupt_shape, random_shape, state.object_shape),
        )

    def observation(self) -> Float[Tensor, "envs observation"]:
        """Render the (possibly corrupted) view and record the evidence it carried."""

        reported_color, reported_shape = self.observed_attributes()
        # Only objects actually inside the view window inform the posterior.
        row_gap = (self.state.object_row - self.state.agent_row[:, None]).abs()
        col_gap = (self.state.object_col - self.state.agent_col[:, None]).abs()
        visible = torch.maximum(row_gap, col_gap) <= self.config.view_radius
        self.color_counts += (
            torch.nn.functional.one_hot(reported_color, self.config.num_colors)
            * visible[..., None]
        )
        self.shape_counts += (
            torch.nn.functional.one_hot(reported_shape, self.config.num_shapes)
            * visible[..., None]
        )
        # Pin the drawn reading so the parent renderer uses it verbatim.
        self._pinned = (reported_color, reported_shape)
        try:
            return super().observation()
        finally:
            self._pinned = None

    # ------------------------------------------------------------- posterior

    def exact_posterior(self) -> Float[Tensor, "envs objects"]:
        r"""Exact Bayes posterior over which object is the cued one.

        The evidence for object ``j`` is a log-likelihood *ratio*, comparing
        "``j`` carries the cued attributes" against "``j`` carries some other
        value", accumulated over readings:

        .. math::
            \ell_j = \sum_v n_{jv}\big[\log P(v \mid \text{cue})
                                      - \log P(v \mid \lnot\text{cue})\big],

        with :math:`P(v\mid u) = 1-\epsilon+\epsilon/K` when ``v == u`` and
        :math:`\epsilon/K` otherwise, and the alternative marginalized over the
        other :math:`K-1` values. A ratio is essential rather than cosmetic: with
        a bare likelihood an object that was *never observed* scores zero, which
        beats any accumulated negative log-likelihood, so the argmax would prefer
        objects it has never seen. Under the ratio, no evidence is neutral.

        This is exactly the log-odds accumulation the Tensor Brain's measurement
        update is claimed to implement, which is what makes it the right
        reference for the agent's own belief.
        """

        epsilon = min(max(self.noisy.observation_noise, 1e-6), 1.0 - 1e-6)
        cue_color, cue_shape = self.state.cue_color(), self.state.cue_shape()
        scores = torch.zeros(self.num_envs, self.config.num_objects, device=self.device)
        for counts, cue, values in (
            (self.color_counts, cue_color, self.config.num_colors),
            (self.shape_counts, cue_shape, self.config.num_shapes),
        ):
            miss = epsilon / values
            hit = (1.0 - epsilon) + miss
            # P(reading | the object is *not* the cued value), marginalizing the
            # true value uniformly over the other `values - 1` possibilities.
            other = max(values - 1, 1)
            null_hit = miss
            null_miss = (hit + (other - 1) * miss) / other
            positive = torch.full((self.num_envs, values), miss, device=self.device)
            positive.scatter_(1, cue[:, None], hit)
            negative = torch.full((self.num_envs, values), null_miss, device=self.device)
            negative.scatter_(1, cue[:, None], null_hit)
            ratio = positive.log() - negative.log()
            scores = scores + (counts * ratio[:, None, :]).sum(-1)
        return torch.softmax(scores, dim=-1)

    def target_posterior(self) -> Float[Tensor, " envs"]:
        """Posterior mass on the object the agent is standing on, or zero."""

        on = (self.state.object_row == self.state.agent_row[:, None]) & (
            self.state.object_col == self.state.agent_col[:, None]
        )
        return (self.exact_posterior() * on.float()).sum(-1)

    # ------------------------------------------------------------ transition

    def step(self, action: Int[Tensor, " envs"]) -> StepResult:
        """Step the world, then apply the volatility hazard to the target."""

        result = super().step(action)
        if self.noisy.hazard_rate > 0.0 and self.config.num_objects > 1:
            switching = (
                torch.rand(self.num_envs, generator=self.generator, device=self.device)
                < self.noisy.hazard_rate
            )
            offset = 1 + torch.randint(
                self.config.num_objects - 1,
                (self.num_envs,),
                generator=self.generator,
                device=self.device,
            )
            moved = (self.state.target_slot + offset) % self.config.num_objects
            self.state.target_slot = torch.where(switching, moved, self.state.target_slot)
            # Evidence about *which* object was the target is now stale for the
            # environments that switched; the readings themselves remain valid.
            self.color_counts[switching] = 0.0
            self.shape_counts[switching] = 0.0
        return result
