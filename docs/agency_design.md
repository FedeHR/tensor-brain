# Agency and Action Indices: Research Design

Status: design document written before the first line of implementation, as the entry point of
the `agency` research line. Results and figures live in [`agency_results.md`](agency_results.md).

## 1. What the papers actually claim

The Tensor Brain papers say very little about agency in equations and quite a lot about it in
prose. Collecting the load-bearing statements from the July-22 QTB draft (`papers/qtb_LATEST.pdf`):

| Locus | Claim | Status in the paper |
|---|---|---|
| §11.2 | Indices exist for "entities, classes, attributes, locations, **actions** (e.g. flee, attack), predicates" | Asserted |
| §12.1.1 | "New: nodes can be action nodes !!!!" | Marginal note |
| §12.2.1 | "Any index changes the brain **and the world**; then there are specific action indices." Activating a `crocodile` index may trigger fleeing. | Asserted |
| §13.4.1 | "New: **actions are generated as any other indices!**" | Marginal note |
| §13.5.1 | "Actions are initiated by activating the corresponding indices. For survival, it is crucial that an agent reacts swiftly." | Asserted |
| §13.5.2 | Near-term planning = iterated evolution with indices activated in between (CoT-like); resulting CBS trajectories "evaluated through their interaction with internal reward functions"; explicitly compared to model-predictive control | Asserted |
| §11.4, Eq. 46 | `g(nu) = sum_k mu_k g(nu_k)` — the representation layer integrates gated input from *reward and action modules*, not only perception | Equation |
| §13.1 | The TB is a state-space model; the CBS is the latent state | Asserted |

There is **no** agentic algorithm, no environment, no credit-assignment rule, and no experiment.
Section 13 is a position statement. So the research question is not "reproduce the agentic
experiment" — there is none — but:

> **Is `k ~ softmax(a0 + A^T sigma(q))` over an action candidate group actually a workable
> policy, and does the rest of the Tensor Brain architecture (shared bidirectional `A`, index
> feedback, serial symbolic bottleneck, dynamic context, evolution-as-imagination) buy anything
> in an environment where agency is genuinely required?**

Everything below is an **experimental extension** in the sense of `docs/fidelity.md`. Nothing here
is paper-faithful in the strong sense, because there is no paper algorithm to be faithful to. What
*is* required is that the extension does not quietly change the core operations: the same `A`
scores and feeds back, `gamma = sigma(q)`, measurement/attention/evolution stay distinct, and
concept-window order stays explicit in experiment code.

## 2. Five testable claims

Restating the prose as things that can fail:

- **C1 — Actions are ordinary indices.** A generative measurement restricted to an action
  candidate group is a usable policy. *Falsified if* a TB agent cannot learn a control task that a
  size-matched conventional recurrent policy learns.
- **C2 — Action feedback is functional.** `q <- alpha q + beta a_k` writes the emitted action back
  into the CBS. This is an efference copy. *Falsified if* `beta = 0` on the action window costs
  nothing on tasks where knowing your own last action matters.
- **C3 — The symbolic perceptual bottleneck is not merely a cost.** The paper's serial ROI
  decoding forces continuous evidence through a discrete index sample before it can influence
  action. *Falsified if* removing the perceptual measurement (features straight to the action
  window) is uniformly better.
- **C4 — Shared bidirectional `A` grounds instruction.** The *same* column `a_red` is the
  perceptual label "I see red" and the top-down instruction "find red". If that is real, an agent
  should follow a cue built from index embeddings, and should recombine cue factors it was never
  trained on. *Falsified if* zero-shot recombination is at chance while a factored one-hot control
  generalizes.
- **C5 — The internal reward function can be an index.** §13.5.2 wants CBS trajectories scored by
  an "internal reward function". The cheapest TB-native realization is a reward *index*: its score
  `a_reward^T sigma(q)` is a value estimate obtained from one more column of the same matrix.
  *Falsified if* this readout is useless as a baseline/critic compared with a separate value head.

A sixth, weaker claim (**C6 — imagination**) is that unrolling `evolve` without input, activating
indices in between, and scoring the imagined CBS with the reward index yields better action
selection than reacting. This is the most speculative and is treated as a demonstration rather
than a headline result.

## 3. Pondering real environments first

The point of this section is to avoid designing a toy that flatters the architecture. What would a
*real* environment have to contain for the TB's distinctive machinery to be load-bearing rather
than decorative?

### 3.1 What the TB is actually offering

A conventional deep-RL policy is `observation -> (recurrent state) -> action logits`. The TB
differs in four ways that could matter:

1. **A single symbol layer shared across perception, memory, instruction and action.** One column
   of `A` per concept, used bottom-up for scoring and top-down for feedback.
2. **A serial attention/measurement bottleneck.** One concept per window; a scene is consumed as a
   sequence of named things, not one pooled vector.
3. **Discrete generative measurement inside the loop.** The state is repeatedly collapsed onto
   sampled symbols; stochasticity is architectural, not exploration noise bolted on.
4. **Evolution as a learned world transition** that can be unrolled without input (imagination).

So the environment must reward (a) naming things, (b) reusing names across roles, (c) attending
serially, (d) remembering, and (e) predicting. If reward depends only on a reactive mapping from
pixels to torque, none of this is exercised, and the TB should be expected to lose to PPO on a
CNN. That is a real risk and is the reason the benchmark question is deferred rather than guessed.

### 3.2 Candidate real environments

| Environment | Fit with TB machinery | Cost / risk |
|---|---|---|
| **BabyAI / MiniGrid** | Very high. Synthetic instruction language *is* a symbol sequence; the index layer can host the same vocabulary used by the instruction. Partial observability by construction. Ships compositional generalization splits. Directly tests C4. | Low visual richness — the perception boundary stays trivial, so the DINO work in this repo is unused. |
| **Crafter** | High for C5/C6: an achievement tree gives long horizons, hierarchical subgoals, and a natural place for reward indices and imagination. Small enough to iterate. | Reward is dense-ish and the symbol set is implicit; would need an ontology defined by us. |
| **ALFWorld / ALFRED** | Highest conceptual fit: household tasks with object states, language goals, and a text twin. Scene graph in, action index out — exactly the TB story, and it reuses this repo's PVSG scene-graph framing. | Heavy; strong LLM baselines make the comparison awkward; slow iteration. |
| **Habitat / ProcTHOR ObjectNav** | High and *uniquely cheap for this repo*: the PVSG pipeline already produces frozen DINOv3 features under a pinned preprocessing contract. Object-goal navigation is literally "goal index in, action indices out" with real egocentric vision. | Simulator engineering; long episodes; credit assignment over hundreds of steps. |
| **MiniHack / NetHack** | Rich symbolic state, genuinely long horizons. | Brutal exploration; would confound architecture with exploration machinery. |
| **Real robotics / BEHAVIOR** | Ultimate target of the embodiment story. | Out of scope by an order of magnitude. |

### 3.3 The path this project commits to

1. **Now — a purpose-built gridworld** (`experiments/agency/gridworld.py`), designed so that each
   of C1–C5 has a specific way to fail, and so that a full ablation grid runs in minutes on a
   laptop CPU. Its job is to establish whether the *mechanism* works at all, and to produce the
   diagnostics (index-probability traces, embedding geometry, value landscapes) that a large
   environment would make illegible.
2. **Next — BabyAI/MiniGrid**, because it converts our hand-made cue vocabulary into a real
   compositional instruction benchmark with published splits and published baselines. This is
   where C4 becomes a claim someone else can check.
3. **Then — ObjectNav with the existing frozen DINOv3 features**, because it is the only step that
   makes the perception boundary real while reusing infrastructure this repository already has,
   and because "goal index in, action index out" is the cleanest large-scale statement of the TB
   agency claim.

Crafter and ALFWorld are noted as the natural venues for C6 (imagination/planning) once C1–C5
have survived steps 1 and 2. We deliberately do not start there: a planning result on a
long-horizon environment would be uninterpretable before we know that the basic action-index
policy works.

### 3.4 The benchmark decision is deliberately deferred

Per the user's framing, step 2 is a *stopping point for discussion*, not an assumption. The
gridworld results should be read as evidence about which of C1–C5 survive, and therefore about
which real benchmark is worth the engineering.

## 4. The gridworld

`SymbolicForaging`: a batched, dependency-free, PyTorch gridworld.

- **Grid** `size x size` (default 6), border acts as a wall.
- **Objects** `num_objects` (default 3) at distinct cells, each with a `(color, shape)` attribute
  pair drawn without replacement from `colors x shapes` (default 3 x 3). The agent starts at a
  distinct cell.
- **Cue** the `(color, shape)` pair of one object, chosen uniformly. Exactly one object matches;
  the others are distractors that share at most one factor.
- **Actions** — five, and they are the action index group: `move_north`, `move_south`,
  `move_west`, `move_east`, `collect`.
- **Reward** `+1` for collecting the cued object, `-1` for collecting a distractor (both terminal),
  `-0.1` for collecting nothing, `-0.02` per step, `0` on timeout at 40 steps.
- **Observation** an egocentric `(2r+1) x (2r+1)` window (default `r = 2`) with channels
  `[colors, shapes, object_present, out_of_bounds]`. The agent's absolute position is never
  observed, so the task is a POMDP and dynamic context has something to do.

Design choices and why:

- *Distractors sharing one factor* make the task require the **conjunction** of the two cue
  indices, so a policy cannot succeed by tracking colour alone. This is what gives C4 teeth.
- *An explicit `collect` action* rather than auto-pickup keeps a non-navigational index in the
  action group, so the action layer is not merely a 4-way compass.
- *Egocentric partial observability* is what makes agency "truly needed" in the user's sense: the
  optimal behaviour is search-then-approach, which no feedforward mapping from the current view
  can express.
- *Held-out cue combinations.* Of the 9 `(color, shape)` cues, 3 form a Latin-square diagonal that
  is never used as a training cue (the objects still appear as distractors, so the visual
  statistics are unchanged). This is the zero-shot recombination test for C4.

## 5. The agent's concept-window schedule

One environment step is one *cycle* of concept windows. The schedule is written out explicitly in
`experiments/agency/agent.py`, in paper order, and never hidden behind a runner:

```text
# --- window boundary ---
q, context = tb.evolve(q, context)                      # Algorithm 1
q = tb.integrate_input(q, view_drive, input_gate=mu_v)  # Algorithm 2 / Eq. 46, perception module
q = tb.integrate_input(q, reward_drive, input_gate=mu_r)# Eq. 46, reward module (previous step)
q = q + a_cue_color + a_cue_shape                       # Eq. 47, top-down instruction indices
q, k_color, p_color = tb.measure(q, colour_group)       # Algorithm 3, perceptual naming
q, k_shape, p_shape = tb.measure(q, shape_group)        # Algorithm 3, perceptual naming
q, k_action, p_act  = tb.measure(q, action_group)       # Algorithm 3, *action* index
reward, obs = env.step(k_action)                        # the index changes the world
```

Notes:

- The action measurement is *the same call* as the perceptual ones with a different candidate
  group. That is C1 stated in code.
- The perceptual measurements name the nearest visible object (or `nothing`), which is the
  gridworld's version of the paper's serial ROI decoding.
- The reward from the previous step enters as its own gated input source, which is the concrete
  reading of Eq. 46's "reward module" and of §11.4's embodiment.
- `retain_gate`/`feedback_gate` on the action window expose HB-POVM `(1,1)`, PVM `(0,1)` and the
  no-feedback generative-RNN `(1,0)` regimes as named conditions, giving C2 directly.

### 5.1 Value as an index score

The vocabulary contains a `reward_positive` index. Its score
`v(q) = a0_reward + a_reward^T sigma(q)` is read *before* the action measurement and used as the
REINFORCE baseline / advantage critic. It costs one extra column of `A` and no new module. This is
C5. The control condition replaces it with an ordinary `nn.Linear(state_dim, 1)` value head.

## 6. Learning

Two stages, mirroring this repository's existing "overfit gate, then full run" culture.

**Stage A — behavioural cloning against a privileged oracle.** The oracle sees the true target
position and takes a greedy Manhattan step, then `collect`. Cloning uses
`selection="teacher"` on the action window and cross-entropy on the action candidate positions.
This is a *capacity* diagnostic: it asks whether the architecture can represent a cue-conditioned
policy at all, without RL variance. It is fast and it already discriminates C3 and C4.

**Stage B — REINFORCE on the generative measurement.** The measurement probability *is* the
policy, so the policy gradient is the ordinary one:

```text
loss = -sum_t log p_act[t, a_t] * (R_t - v_t) - c_H * H(p_act[t]) + c_V * (v_t - R_t)^2
```

with `R_t` the discounted return-to-go and `v_t` the reward-index value readout. No new model
component is introduced: `log p_act` comes straight out of `measure`, and `v_t` out of
`index_scores`. Entropy regularization is a standard RL necessity and is reported as such, not as
a TB claim.

Perceptual measurements are also sampled during RL. Their log-probabilities are *not* included in
the policy gradient by default — a named condition (`percept-in-pg`) does include them, which asks
whether the agent benefits from optimizing what it names for the reward it collects. This is a
genuinely interesting TB question (does an agent learn to see what is useful?) and is reported
separately.

## 7. Conditions and controls

| Name | What changes | Claim |
|---|---|---|
| `tb-full` | reference: original recurrence, `(1,1)` gates, both perceptual measurements, reward index critic | C1 |
| `no-action-feedback` | action window `feedback_gate = 0` | C2 |
| `pvm-action` | action window `retain_gate = 0` | C2 (order effect) |
| `no-percept-feedback` | perceptual windows `feedback_gate = 0` | C3 |
| `no-percept-measure` | perceptual measurements removed | C3 |
| `argmax-action` | deterministic winner-take-all action selection | C1 / exploration |
| `no-evolution` | `q` carried across steps without the evolution operator | memory |
| `evolution=qtb` / `relu` | feed-forward backends (no persistent context) | memory |
| `score=softplus-bias` / `centered` | index-score offsets | scoring |
| `linear-critic` | `nn.Linear` value head instead of the reward index | C5 |
| `cue-initial` | cue indices injected only at `t = 0` | memory + C4 |
| `gru-control` | size-matched GRU policy with factored one-hot cue, no index layer | C1 / C4 fairness |
| `lstm-control` | same, with an LSTM cell: is it recurrence in general or this recurrence? | C1 |
| `decoupled-feedback` | scoring keeps `A`; feedback gets its own trained matrix | shared bidirectional `A` |
| `deliberate-{2,3}-attend`, `deliberate-2-measure` | extra internal concept windows per environment step | C6 |

Every condition is run over the same seeds with the same environment seeds.

**The reinforcement-learning recipe was selected by watching the Tensor Brain agent learn** and
then applied unchanged to the GRU and LSTM controls. That biases the comparison in the Tensor
Brain's favour. Within-grid ablation contrasts are therefore the defensible readings; no claim
that the Tensor Brain outperforms a conventional recurrent policy follows from this grid.

### 7.1 Metrics, and why success rate alone is not enough

This was discovered by the `no-cue` control, which is what controls are for. Because collecting a
distractor is penalised but *not* terminal (Section 4), an agent that knows nothing about the
instruction can still finish most episodes: collect objects until the reward turns positive. On
one seed `no-cue` reached 0.97 success this way. Success rate therefore measures "did the episode
end well", not "did the agent follow the instruction".

The reported metrics are consequently:

- **first-choice accuracy** — was the *first* object the agent committed to the cued one? A
  cue-blind agent scores 1/3 here however many objects it collects afterwards. This is the
  primary instruction-following measure, reported beside `first_choice_rate` so that a
  never-collecting agent (which scores 0 by convention) is distinguishable from a wrong-choosing
  one;
- **mean episode return** — nets brute-force distractor collection out of success;
- **distractor collections per episode** — brute forcing made directly visible;
- **success rate** and **episode length**, retained for comparability;
- all of the above on training cues *and* on the held-out cue diagonal.

## 8. Diagnostics and figures

Quantitative:

- learning curves (success rate vs environment steps) per condition, mean ± s.e. over seeds;
- ablation bars at fixed budget, with held-out-cue success shown beside training-cue success;
- BC capacity curves (Stage A) per condition.

Qualitative, because they are what makes a symbolic architecture worth having:

- **Learning curves in both success and return**, since the two can disagree;
- **Trajectory strips**: grid renderings of a rollout annotated, per step, with the sampled
  perceptual indices and the sampled action index — i.e. the agent's own symbolic narration.
- **Index-probability rasters**: `p_act` and `p_color`/`p_shape` over an episode, showing the
  moment the target enters view and the policy commits.
- **Embedding geometry**: PCA / cosine structure of `A`, asking whether action columns separate
  from perceptual columns, and whether `a_red` sits near the CBS states in which red is both seen
  and sought.
- **Value landscape**: the reward-index score `a_reward^T sigma(q)` evaluated at every grid cell of
  a fixed layout, to see whether a single extra column of `A` learned a spatial value function.
- **Imagination rollouts** (C6): unrolled `evolve` trajectories with their reward-index scores,
  against what the environment actually returned.

## 9. What is *not* being claimed

- No claim that the TB is a competitive RL architecture. The controls exist to keep that honest.
- No claim of biological plausibility of the credit-assignment rule. REINFORCE is a tool for
  asking whether the *architecture* can carry a policy, not a model of the brain.
- No claim that the paper prescribes any of this. Section 13 is prose; this is one concrete,
  falsifiable reading of it, recorded as an experimental extension in `docs/fidelity.md`.
