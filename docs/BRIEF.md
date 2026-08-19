# Shared brief: what the Heisenberg update is, and what we are hunting

## The object

A latent state `x` is carried as a **natural parameter** `q` of an exponential
family (canonical case: independent Bernoulli, `T(x)=x`, `gamma = sigmoid(q)`).
A symbol `k` is emitted by a softmax readout over that state:

    P(k|x) = exp(a_{0,k} + a_k^T x) / Z(x),     Z(x) = sum_k exp(a_{0,k} + a_k^T x)

Exact Bayes for M observations of one unchanged state:

    log P(x | k_1..k_M) = const + (q_0 + sum_m a_{k_m})^T x  -  M log Z(x)

The **Heisenberg / additive / TB update** simply drops the last term:

    q <- q + a_k        (one O(n) addition per symbol, order-invariant, no normalization)

## The five load-bearing results (all verified numerically in this repo)

1. **Error law.** `KL(Q_H || P) ~= (M^2/2) Var[log Z(X)]`. Locally
   `= (M^2/2) sum_i gamma_i(1-gamma_i) g_i^2` with `g = A pi(x)`, i.e. the
   softmax-weighted vote sum. Practical scaling: `~ (M^2/2)(n vbar) sigma_A^2 / K_eff`
   where `K_eff = 1/sum_k pi_k^2`. More outcomes and more diverse embeddings =>
   votes cancel => the approximation gets BETTER.

2. **Exactness criterion is affine, not constant.** `log Z(x) = d + c^T x + r(x)`;
   the affine part is an **unidentified gauge** of the readout (`A -> A - c 1^T`
   leaves `P(k|x)` invariant but changes `log Z` and hence the additive update).
   So: gauge-fix the trained weights once, free, removes ~53% of belief error.
   Only the non-affine residual `r` is irreducible. A `Var[log Z]` training
   penalty buys another 53% for 0.035 nats of fit; beats L2 at matched fit cost
   (but loses to L2 in the data-limited regime).

3. **THE CANCELLATION IDENTITY (the sharpest claim).** Suppose the observation
   is not unconditional but **gated**: a window opens with probability
   `pi(x) = Z(x)/C`, and only opened windows are recorded. Then
   `p(open, k | x) = exp(a_{0,k}+a_k^T x)/C` -- **Z cancels exactly**.
   The plain additive update is then the EXACT posterior, for every M, to
   machine precision, in every regime tested (including where the unconditional
   error is 20 nats). Corollary, verified: on gated data, applying the
   principled normalizer correction is the WORST rule. "Debiasing" a
   score-gated corpus makes beliefs worse.
   Run forward in time this is stronger: under gating the exact filtering
   distribution never leaves the factorized family -- the product state is
   **sufficient, not lossy** (KL 8.9e-7 vs 1.4e-1 unconditional, 150,000x).

4. **Silence is evidence (the real limit of #3).** A window that stayed shut has
   likelihood `1 - Z(x)/C`, which is NOT affine, so no additive rule absorbs it.
   Fix: least-squares affine part applied as a constant drift `q <- q + c_s` per
   silent window; recovers 80-90% of the closable gap. Still O(n), still
   order-invariant, computed once per model.

5. **Gaussian branch => tight frames.** Mean-only Gaussian state, `y ~ N(Wx, sigma^2 I)`:
   dropped term is `-(1/2 sigma^2) x^T W^T W x`, isotropic exactly when
   `W^T W = cI` (a **tight frame**). Then a single global gain
   `beta* = 1/(1 + c tau^2/sigma^2)` makes the posterior mean exact. Diagnostic:
   frame defect `|| K K^T - cI ||_F`. Note the structural identity with modern
   linear attention: `S <- S + beta_t (v - S k) k^T`.

## Known negatives (do not re-propose these)

- No robustness gain under misspecification; worse under redundant evidence.
- More heads is monotonically WORSE (`Var[sum_h log Z_h]` grows with head count):
  prefer one wide vocabulary to many narrow ones.
- Fixed top-k retrieval does NOT give cancellation: a constant recorded count is
  conditioning on the total, which is the multinomial (not Poisson) branch and
  reinstates the normalizer. Only **variable-count / threshold / abstention**
  gates cancel.
- Vanilla POMDP belief updating (unconditional measurement every step) is the
  worst regime for this rule.
- Model merging / task arithmetic: gauge freedom has no weight-space analogue.

## Already proposed and judged not-yet-convincing by the user

MIMIC-IV clinical event streams; self-initiated measurement in RL with costly
observations; recommenders with latent-dependent exposure (KuaiRec / Yahoo!R3 /
Coat / Open Bandit); the delta-rule/linear-attention connection; neural collapse
<=> exactness over training; PVSG/COCO scene-graph annotation channels.
A diffusion parallel-decoding probe was run and killed at stage 0.

## What we are hunting

Something *truly* interesting: either (a) a modern (2025-2026) DL setting where
this makes a sharp, falsifiable, contrarian prediction on a real benchmark, or
(b) an older foundational question that these ideas actually resolve or reframe.
Prefer claims whose SIGN is opposite to the incumbent's advice, because those
are cheap to test and impossible to dismiss.

## Source documents (read these, do not take this summary as complete)

- `/Users/fede/LMU/SoSe26/master_b/tensor-brain-bayes-approximation/docs/tb_update_generalized.md`
  (the sharp claims; sections 3, 4, 6, 8b, 10, 12, 13)
- `/Users/fede/LMU/SoSe26/master_b/tensor-brain-bayes-approximation/docs/bayes_approximation.md`
- `/Users/fede/LMU/SoSe26/master_b/tensor-brain/thesis/new_version/chapters/04_derivation_intuitive_alternative.tex`
- `/Users/fede/LMU/SoSe26/master_b/tensor-brain/.claude/worktrees/heisenberg-application-candidates/docs/heisenberg_experiment_design.md`
  (section 3.3 ranks six opportunities; do not simply restate it)
