"""The condition grid for the Memory Maze filter study.

Nine conditions, each answering one question. There is no cross-product here on
purpose: an ablation axis that no one will read is a run that could have been a
seed instead.

============================  ==================================================
condition                     the question it answers
============================  ==================================================
``tb-none``                   what does the index layer contribute at all?
``tb-raw``                    the paper's update, alpha = 1
``tb-raw-alpha0``             does the earlier alpha = 0 result reverse once
                              retention is actually required?
``tb-corrected``              does the zero-mean write fix CBS saturation?
``tb-softplus-a0``            does QTB's log-normalizer bias do it instead?
``tb-soft``                   does the *discreteness* of the measurement earn
                              anything over ordinary attention?
``tb-raw-noact``              is the action write a control input?
``tb-accumulator``            is the pure log-odds filter -- no learned
                              prediction step at all -- already enough?
``gru-control``               a capacity-comparable modern baseline
============================  ==================================================

``tb-raw-alpha0`` deserves a note, because it is the condition most likely to be
misread. Three earlier studies in this line found ``alpha = 0`` beating the
paper's ``alpha = 1``. All three ran on tasks where nothing had to be retained
across steps, so discarding the prior cost nothing and avoided saturation. If
that result was a property of those tasks rather than of the architecture, it
should invert here, where retention is the entire point. Either outcome is
informative, which is why the condition is in the grid rather than assumed.

The masking levels are the study's independent variable: ``rho`` is the
probability that a step's observation is withheld. ``rho = 0`` is the control --
with full observation nothing needs to be remembered, so the architectures
should tie, and a difference there would indicate a confound rather than memory.
"""

from __future__ import annotations

from dataclasses import replace

from experiments.agency.memorymaze.filter import FilterConfig

# The reference filter. `state_dim=128` matches the policy study, which keeps
# the parameter counts in the same place and the two stages comparable.
REFERENCE = FilterConfig(
    state_dim=128,
    hidden_dim=128,
    num_latent_indices=64,
    evolution="qtb",
    score_mode="direct",
    feedback="raw",
    action_retain_gate=1.0,
    action_write=True,
    cell="tb",
)

CONDITIONS: dict[str, FilterConfig] = {
    "tb-none": replace(REFERENCE, feedback="none"),
    "tb-raw": REFERENCE,
    "tb-raw-alpha0": replace(REFERENCE, action_retain_gate=0.0),
    "tb-corrected": replace(REFERENCE, feedback="corrected"),
    "tb-softplus-a0": replace(REFERENCE, score_mode="softplus-bias"),
    "tb-soft": replace(REFERENCE, feedback="soft"),
    "tb-raw-noact": replace(REFERENCE, action_write=False),
    # No learned prediction step: `q` is propagated by the identity, so the
    # schedule reduces to `q <- alpha q + drive + a_k + a_u`, which is the
    # log-odds Bayes filter in its bare form.
    "tb-accumulator": replace(REFERENCE, evolution="none"),
    "gru-control": replace(REFERENCE, cell="gru"),
}

# The masking sweep. Three levels rather than five: with one seed per cell the
# budget buys a trend, and a trend needs three points, not five.
MASK_PROBABILITIES: tuple[float, ...] = (0.0, 0.5, 0.9)


def condition_config(name: str, mask_probability: float) -> FilterConfig:
    """The config for one cell of the grid."""

    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; expected one of {sorted(CONDITIONS)}")
    return replace(CONDITIONS[name], mask_probability=mask_probability)


def grid() -> list[tuple[str, float]]:
    """Every cell, in the order the Slurm array indexes them."""

    return [
        (name, probability)
        for name in CONDITIONS
        for probability in MASK_PROBABILITIES
    ]
