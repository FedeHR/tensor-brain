"""The named conditions of the agency ablation grid.

One entry per controlled variable. Every condition differs from ``tb-full`` in
exactly one respect unless its name says otherwise, so a difference in the
reported metric is attributable.
"""

from __future__ import annotations

from dataclasses import replace

from experiments.agency.agent import AgentConfig
from experiments.agency.gridworld import GridConfig

# The task used by every reported run. Partially observed (a corner cell sees
# nine of twenty-five cells), conjunction-cued, with an explicit collect action.
TASK = GridConfig(
    size=5,
    num_objects=3,
    num_colors=3,
    num_shapes=3,
    view_radius=2,
    max_steps=25,
)

REFERENCE = AgentConfig(
    state_dim=64,
    hidden_dim=64,
    evolution="original",
    score_mode="direct",
    deliberation_windows=1,
)

CONDITIONS: dict[str, AgentConfig | None] = {
    # --- C1: does the action-index measurement work as a policy at all? ---
    "tb-full": REFERENCE,
    # `None` selects the non-Tensor-Brain control policy.
    "gru-control": None,
    "lstm-control": None,
    # Control for the shared bidirectional matrix: scoring keeps `A`, feedback
    # gets its own independently trained matrix of the same shape.
    "decoupled-feedback": replace(REFERENCE, decouple_feedback=True),
    # --- C2: is the action index's feedback into the CBS functional? ---
    "no-action-feedback": replace(REFERENCE, action_feedback_gate=0.0),
    "pvm-action": replace(REFERENCE, action_retain_gate=0.0),
    # --- C3: is the symbolic perceptual bottleneck a cost or a benefit? ---
    "no-percept-measure": replace(REFERENCE, measure_percepts=False),
    "no-percept-feedback": replace(REFERENCE, percept_feedback_gate=0.0),
    # --- C4: do the shared index columns actually carry the instruction? ---
    "no-cue": replace(REFERENCE, cue_mode="none"),
    "cue-initial": replace(REFERENCE, cue_mode="initial"),
    # --- memory: what does the evolution operator contribute? ---
    "no-evolution": replace(REFERENCE, evolution="none"),
    "evolution-qtb": replace(REFERENCE, evolution="qtb"),
    "evolution-relu": replace(REFERENCE, evolution="relu"),
    # --- scoring conditions from the existing fidelity ledger ---
    "score-softplus-bias": replace(REFERENCE, score_mode="softplus-bias"),
    "score-centered": replace(REFERENCE, score_mode="centered"),
    # --- C5: can one column of A serve as the internal reward function? ---
    "linear-critic": replace(REFERENCE, critic="linear"),
    "no-critic": replace(REFERENCE, critic="none"),
    # --- C6: iterated evolution with indices activated in between ---
    "deliberate-2-attend": replace(REFERENCE, deliberation_windows=2, deliberation_mode="attend"),
    "deliberate-3-attend": replace(REFERENCE, deliberation_windows=3, deliberation_mode="attend"),
    "deliberate-2-measure": replace(REFERENCE, deliberation_windows=2, deliberation_mode="measure"),
    # --- generative sampling versus winner-take-all decoding ---
    "argmax-action": replace(REFERENCE, action_selection="argmax"),
    # --- symbol grounding: does it help to name what is actually there? ---
    # Same agent as `tb-full`; only the objective differs (see REINFORCE_OVERRIDES).
    "grounded-percepts": REFERENCE,
    "percepts-in-policy-gradient": REFERENCE,
}

# Conditions whose *objective*, not whose architecture, differs from the
# reference. Reward alone never requires the perceptual measurement to be
# accurate, and it is observed to collapse onto the instruction: the agent names
# what it is looking for rather than what it sees. These two conditions ask
# whether making the symbols mean something changes the policy.
REINFORCE_OVERRIDES: dict[str, dict[str, float | bool]] = {
    # Auxiliary cross-entropy towards the true label of the attended object.
    "grounded-percepts": {"percept_weight": 1.0},
    # Instead of grounding them, credit the perceptual measurements with the
    # reward they preceded, i.e. let the agent learn to see what is useful.
    "percepts-in-policy-gradient": {"percept_in_policy_gradient": True},
}

# Figures group conditions by the claim they address.
CLAIM_GROUPS: dict[str, tuple[str, ...]] = {
    "C1 policy and control": ("tb-full", "gru-control", "lstm-control", "argmax-action"),
    "shared bidirectional A": ("tb-full", "decoupled-feedback"),
    "C2 action index feedback": ("tb-full", "no-action-feedback", "pvm-action"),
    "C3 perceptual bottleneck": (
        "tb-full",
        "no-percept-measure",
        "no-percept-feedback",
        "grounded-percepts",
        "percepts-in-policy-gradient",
    ),
    "C4 instruction indices": ("tb-full", "cue-initial", "no-cue"),
    "C5 internal reward function": ("tb-full", "linear-critic", "no-critic"),
    "C6 deliberation depth": (
        "tb-full",
        "deliberate-2-attend",
        "deliberate-3-attend",
        "deliberate-2-measure",
    ),
    "memory and evolution": ("tb-full", "no-evolution", "evolution-qtb", "evolution-relu"),
    "index score offsets": ("tb-full", "score-softplus-bias", "score-centered"),
}
