# Front: tight frames and the delta rule — state of the art (Aug 2026) and verdict

Scope: brief claim 5 (Gaussian branch => tight frames => single global gain `beta* =
1/(1 + c tau^2/sigma^2)` is Bayes-exact), tested against the 2024-2026 linear-attention
literature. Everything below was checked against arXiv this session; IDs and dates are
verbatim from the listings.

---

## 0. Verdict up front

**The theory is scooped. Comprehensively, and including the exact corollary.** Four
independent 2025-2026 lines already publish the derivation the brief proposes, and one of
them (Preconditioned DeltaNet, arXiv:2604.21100, 22 Apr 2026) states the tight-frame
corollary in its abstract, in different words: *"we derive equivalences between linear
attention and the delta rule in the exactly preconditioned case."* Exact preconditioning
means `P_t = G_t^{-1}` with `G_t = sum_i k_i k_i^T + lambda I`. Setting `G_t = cI` gives
`P_t = I/c` — a single global gain — and their Theorem 3.1 (`S_t = C_t P_t`, with
`C_t = sum v_i k_i^T` the pure linear-attention state) collapses to exactly claim 5.
A reviewer will read claim 5 as one line of that theorem.

**Worse for the pitch as framed:** the tight-frame result does not derive the *delta rule*.
It derives *plain additive linear attention* with a global gain — the thing the field spent
2024-2026 moving away from. See §5, trap 1. The brief's line "note the structural identity
with modern linear attention: `S <- S + beta_t(v - Sk)k^T`" is the wrong identification.

**What survives, and it is not nothing:** nobody has measured the frame geometry of the
accumulated keys per head in a trained delta-rule model and used it as a *diagnostic* for
where the expensive machinery is wasted. Option (c) of the mission. That is a real gap, it
is one day of work on the Mac, and it is falsifiable with a pre-registered number. It is a
thesis section or a workshop paper, **not a main-conference paper**, and the scoop window is
months (§6).

---

## 1. The 2026 landscape: state-update rules

Notation: state `S_t in R^{d_v x d_k}`, key `k_t`, value `v_t`, read `y_t = S_t q_t`.
"Delta form" = `S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T`, equivalently
`S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T`.

| Model | arXiv / date | Update | Delta form? |
|---|---|---|---|
| Linear attention | 2006.16236 (2020) | `S_t = S_{t-1} + v_t k_t^T` | No — pure additive |
| RetNet / GLA / Mamba-2 | 2307.08621 / 2312.06635 / 2405.21060 | `S_t = alpha_t S_{t-1} + v_t k_t^T`, `alpha` scalar / diagonal / data-dep. | No — additive + decay |
| **DeltaNet** | Schlag 2102.11174; parallelized Yang et al. **2406.06484** | `S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T`, `k` L2-normalized, `beta_t in (0,1)` (or `2beta_t` for negative eigenvalues) **learned** | **Yes, exactly** |
| **Gated DeltaNet** | **2412.06464** (ICLR 2025) | `S_t = alpha_t S_{t-1}(I - beta_t k k^T) + beta_t v k^T` | Yes + scalar decay |
| **Kimi Delta Attention** | **2510.26692** (Oct 2025) | as GDN but `alpha_t -> diag(a_t)` channel-wise; `beta_t` stays a scalar | Yes + diagonal decay |
| **Gated DeltaNet-2** | **2605.22791** (21 May 2026, Hatamizadeh/Choi/Kautz, NVIDIA) | channel-wise **erase** gate `b_t` and channel-wise **write** gate `w_t`, decoupled; reduces to KDA when both collapse to one scalar, to GDN when the decay also collapses | Yes, generalized |
| **RWKV-7 "Goose"** | **2503.14456** | `S_t = S_{t-1}(diag(w_t) - khat_t (a_t (*) khat_t)^T) + v_t k_t^T` | Generalized delta: removal direction != write direction, vector in-context LR `a_t` |
| **Mamba-3** | **2603.15569** (ICLR 2026) | exponential-trapezoidal discretization (3-term, second-order), complex state / data-dependent RoPE, MIMO | **No** — never subtracts a current read |
| **DeltaProduct** | **2502.10297** | `n_h` Householder factors per token; spans all of `O(d)` by Cartan-Dieudonne | Yes, `n_h` delta steps/token |
| **Longhorn** | **2407.14207** | closed form of implicit online regression: `S_t = S_{t-1} + (eps_t/(1 + eps_t ||k_t||^2)) (v_t - S_{t-1}k_t) k_t^T` | **Yes, with beta DERIVED not learned** |
| **MesaNet** | **2506.05233** | `S_t = (sum gamma v k^T)(sum gamma k k^T + lambda I)^{-1}`, CG solve at read time | No — full inverse Gram |
| **Titans** | **2501.00663** | deep MLP memory, momentum + weight decay on a surprise gradient | No — nonlinear TTT |
| **Atlas** | **2505.23735** | Omega rule: optimize memory over a *sliding window* of past tokens, Muon-style | No — higher-order |
| **Gated KalmaNet** | **2511.21016v3** (v3 17 May 2026, Peng/Chattopadhyay/Zancato/Nunez/Xia/Soatto, Penn + AWS) | `H_t = gamma H_{t-1} + k k^T`, `U_t = gamma U_{t-1} + v k^T`, `y_t = U_t · CH(H_t + lambda_t I, q_t, r)`; Chebyshev iteration, `lambda_t = a||H_t||_F` | No — exact Kalman gain, full error covariance |
| **Preconditioned DeltaNet (PDN)** | **2604.21100** (22 Apr 2026, Tumma/Loo/Rus, MIT+Liquid) | `S_t = S_{t-1} + (v_t - S_{t-1}k_t)(P_{t-1}^{norm} k_t)^T`, `P = G^{-1}`, `P^{norm} = P/(1 + k^T P k)`; diagonal approx in practice | Delta form with a **matrix** gain |
| **OSDN** | **2605.13473** | diagonal preconditioner updated online by hypergradient; "algebraically equivalent to a per-feature scaling of the write-side key" | Delta + diagonal gain |
| **Kalman Linear Attention (KLA)** | **2602.10743v2** (Shaj/Barker/Scannell/Szecsenyi/Crowley/Storkey, Edinburgh) | diagonal linear-Gaussian SSM; precision as a Mobius map `lambda_t = (alpha lambda + beta)/(gamma lambda + delta)`, forget gate `f_t = rho_t abar_t`, `rho_t = 1/(abar_t^2 + pbar_t lambda_{t-1})` | Gate derived from Bayes, diagonal |
| **Variational Linear Attention** | **2605.11196** | online regularized LS, adaptive penalty matrix via Sherman-Morrison | Delta + matrix penalty |
| Also 2026 | MDN 2605.05838 (momentum), Parallax 2605.29157, Local Linear Attention 2510.01450, Q-Delta 2606.08804, FG^2-GDN 2604.19021, Exact Flow LA 2512.12602 | | |

### Which already have an explicit Bayes / Kalman / online-regression derivation

All three of the brief's guesses check out, plus more:

- **Longhorn (2407.14207)** — confirmed. SSM design *is* online-learning-objective design;
  the per-step closed-form solution is the recurrence. **Its gate is literally the brief's
  `beta*`**: `beta_t = eps_t/(1 + eps_t ||k_t||^2)`, the same Mobius shape as
  `1/(1 + c tau^2/sigma^2)`. This is the single most damaging prior work for claim 5 as
  a *formula* claim.
- **MesaNet (2506.05233)** — confirmed. Locally optimal test-time training; solves the
  regularized cumulative least-squares problem to optimality at every step by CG.
- **Gated KalmaNet (2511.21016)** — confirmed, and stronger than the brief remembered.
  §6.4 states separately for each: *"DeltaNet approximates the KF recursion by assuming
  fixed error covariance: Sigma_t = I_n for all t"*, and for GDN, *"Like DeltaNet, GDN
  avoids tracking the evolving uncertainty Sigma_t, trading optimality for computational
  simplicity."* Same for KDA. So "DeltaNet = Kalman under identity covariance" is in print.
  Note: they never state when that assumption is *harmless*, and they report no key-covariance
  conditioning diagnostic on their own trained models. That omission is the opening.
- **Test-time regression (Wang, Shi, Fox, 2501.12352)** — confirmed as the unifying frame.
  Linear attention, SSMs, fast-weight programmers, online learners and softmax attention
  all fall out; explicitly diagnoses that *linear attention fails to capture inter-token
  correlations* (i.e. the key Gram) and gives a mathematical justification for QK-norm.
- **KLA (2602.10743)** — a full diagonal Gaussian SSM derivation of the gates. Does not
  compare its closed-form gate to DeltaNet's learned one empirically.
- **PDN (2604.21100)** and **OSDN (2605.13473)** — the second-order/curvature version:
  the correct step is preconditioned by the inverse key Gram.
- **VLA (2605.11196)** — same, in a variational dress; and it states in words the thing
  claim 5 states in symbols: *standard linear attention corresponds to a fixed, isotropic
  inverse covariance where all write directions are treated identically.*

---

## 2. Is the tight-frame condition published?

**As a named condition, no. As a fact, yes — twice over.**

What is published:
- `G_t = sum_i k_i k_i^T` is the object that the correct update must invert
  (PDN Thm 3.1; MesaNet; GKN; VLA; test-time regression). The tight-frame condition
  `G = cI` is the case where that inverse is a scalar. Nobody writes "tight frame", but
  everybody writes the matrix whose isotropy is the condition.
- PDN's abstract already claims the equivalence "linear attention <-> delta rule in the
  exactly preconditioned case". `G = cI` is the sub-case where *no* preconditioning is
  needed to get the equivalence.
- VLA states the isotropy fact in prose (above).
- **Frame-theory vocabulary is genuinely absent.** No hits for tight frame / Parseval
  frame / frame potential / Benedetto-Fickus anywhere in the attention literature. But
  vocabulary is not a contribution, and a reviewer will (correctly) call it renaming.

**The L2-norm vs tight-frame distinction: the brief is right, and the distinction is
nowhere made.** DeltaNet L2-normalizes each `k_t` (fla config for
`fla-hub/delta_net-1.3B-100B` has `qk_norm: "l2"`), which fixes `||k_t|| = 1` and hence
`tr(G/n) = 1` — it pins the *trace* of the Gram and nothing else. `G/n = I/d` additionally
requires the *collection* to be spread isotropically. Every stated justification of
L2-norm in the literature is about stability of the Householder `I - beta k k^T`
(spectral norm exactly 1; VLA Prop. 2 proves this) — never about isotropy. Papers that do
care about isotropy (PDN, OSDN, CCQ, VLA, GKN) all reach for a matrix, none of them notes
that L2-norm is the rank-1 shadow of what they want. **This distinction is a real, small,
unclaimed gap.** It is a good paragraph. It is not a paper.

Closest existing empirical work:
- **PDN Figure 1** already plots the key-Gram eigenspectrum of a *pretrained DeltaNet-340M*
  and reports eigenvectors are "fairly axis-aligned" (justifying their diagonal approx).
  They report no condition number, no isotropy measure, no per-head breakdown.
- **arXiv:2602.04852**, "The Key to State Reduction in Linear Attention: A Rank-based
  Perspective" (Feb 2026): trained linear-attention *states* are low-rank; uses
  `kappa(S) = sigma_1/sigma_d` as an anisotropy measure; notes that improving the
  conditioning of the keys improves the effective rank of the memory. This is the nearest
  neighbour to the proposed diagnostic — but it is on the *state*, not the key frame, and
  it is not per-head-predictive.
- **arXiv:2606.01294**, "Don't Read Everything: A Curvature-Conditioned Query" (31 May
  2026, Le/Nguyen/Nguyen/Luu): maintains the running key covariance `Sigma_t` per head
  and contracts the query, `q^clean = (I - lambda_t Sigma_t) q`. Measures on Qwen3-4B and
  Gated DeltaNet 500M that needle keys project less onto the top-16 key subspace than
  distractors at 100% of layers (`delta mu ~ -0.21` linear vs `-0.11` softmax). This is
  direct evidence that **trained key frames are strongly anisotropic**, which is the
  prior you should carry into §3.
- **arXiv:2607.19390** ("The Orthogonalized Read Is a Removable Training Scaffold", Jul 2026)
  runs the nearest thing to the whitening ablation, on mLSTM read-side rather than key-side,
  and finds the mechanism is a *training* scaffold that can be annealed away — and that
  "the key-Gram effective rank is indistinguishable across variants and outcomes". That
  last clause is a warning shot for §3.

---

## 3. The decisive measurement

### 3.1 Metric (finite-sample-correct — this matters)

Naive `||K K^T/n - cI||_F / c` is a **trap**: under the isotropic null with `n` unit keys in
`d` dims, `E||G/n - I/d||_F ~ 1/sqrt(n)`, so the relative defect is `~ d/sqrt(n)`, which is
2.8 at `d=128, n=2048` — the null is not near zero and the statistic is dominated by
sampling noise. Use the **frame potential / participation ratio** instead, which is exactly
the frame-theoretic object (tight frames minimize the frame potential, Benedetto-Fickus):

    G_h = (1/n) sum_{t=1..n} k_t k_t^T          (per head h; tr G_h = 1 under L2-norm)
    A_h = d * tr(G_h^2) / (tr G_h)^2  -  1       (anisotropy; 0 = tight, d-1 = rank-1)
    PR_h = 1/(1 + A_h)                           (participation ratio, 1 = tight, 1/d = rank-1)

    equivalently  d * tr(G^2) = (d/n^2) * sum_{s,t} <k_s, k_t>^2   <- literally the frame potential

Isotropic null (n unit vectors uniform on S^{d-1}): `A_null = (d-1)/n`, `PR_null = n/(d+n-1)`.
For `fla-hub/delta_net-1.3B-100B` (hidden 2048, 16 heads, `expand_k=1` => `d = 128`,
24 layers => **384 heads**) at `n = 2048`: **`A_null = 0.062`, `PR_null = 0.942`.**
Report `A_h / A_null` (the *excess* anisotropy) — this is the frame defect, made
sample-size-honest.

### 3.2 The measurement, at zero extra cost

One forward pass records, per head, (i) `G_h` (384 x 128 x 128 floats = 25 MB total —
accumulate, never store the keys) and (ii) the learned gate trajectory `beta_t^{(h)}`.
DeltaNet's `beta_t` is a scalar per head per token, so this is 384 x 2048 numbers.

Then the two statistics per head:

    A_h / A_null                                          (frame defect)
    CV_h = std_t(beta_t^{(h)}) / mean_t(beta_t^{(h)})     (how much the gate actually varies)

**Pre-registered prediction.** If claim 5 is the right lens, a head whose key frame is
near-tight has nothing for a per-token gain to do — the Bayes-optimal gain is a constant —
so its learned gate should be near-constant. Test:

    Spearman rho( A_h/A_null , CV_h ) over 384 heads.

- **Support:** `rho >= +0.35` (p < 1e-12 at n=384).
- **Kill:** `|rho| < 0.15`. Then `beta_t` is a surprise/salience signal unrelated to key
  geometry and the whole Bayesian reading of the gate is decoration.

This costs **one forward pass**, needs **no training**, and is the cheapest thing that can
kill the idea. Do this before anything else.

### 3.3 The causal confirmation (only if 3.2 passes)

Per head, replace `beta_t^{(h)}` by the constant `mean_t beta_t^{(h)}` (one head at a time,
inference-time surgery, no retraining) and measure `Delta NLL_h` on held-out text plus a
retrieval probe (MQAR or a needle set).

- **Support:** `Delta NLL` on the tightest quintile of heads `< 0.01` nats; on the least
  tight quintile `> 0.05` nats; **ratio > 5x**.
- **Kill:** ratio `< 2x`.

Cost: 384 forward passes (or 24 layer-wise passes first, then head-wise inside the extreme
layers). Still one day.

### 3.4 Feasibility on the Mac

Yes, comfortably — this is inference only, which is exactly what the local-compute rule
allows.

- Machine: M3 Pro, 19.3 GB RAM. No torch/fla currently installed in this repo's venv
  (`pyproject.toml` pins `torch==2.7.1`, with a CUDA index only for linux; on darwin you get
  the CPU/MPS wheel).
- Checkpoints, in order of preference:
  - `fla-hub/delta_net-1.3B-100B` — **the cleanest target**: pure DeltaNet, scalar learned
    `beta`, `use_beta: true`, `qk_norm: "l2"`, 24 x 16 = 384 heads, `d = 128`, ctx 2048,
    bf16 ~2.6 GB. Fits easily.
  - `fla-hub/delta_net-2.7B-100B` — same family, ~5.4 GB, also fits.
  - `linear-moe-hub/Gated-Deltanet-340M` / `-1.3B`, `m-a-p/1.3B-100B-GatedDeltaNet-hybrid-3-1`,
    `Idiap/gated-deltanet-attn-1.4B-30B` — for the gated variant and the hybrid check.
  - `fla-hub/rwkv7-0.1B-g1a` / `-0.4B-g1a` / `-1.5B-g1` — RWKV-7 replication; note the
    removal direction differs from the write direction, so define `G` on `khat_t` (the
    removal key), not on the write key.
  - Out of reach locally: `moonshotai/Kimi-Linear-48B-A3B-*` (KDA, 5.7T tokens),
    Qwen3-Next-80B-A3B. Cluster only, and only if the small-model result holds.
  - No public MesaNet or Gated KalmaNet weights found; PDN releases code, checkpoints
    unconfirmed.
- **Practical caveat:** `fla` is Triton-based. CPU/XPU/NPU extras exist but the Mac path is
  fragile. **Do not fight it** — you only need (a) the projections and (b) the recurrence.
  Write the DeltaNet recurrence in ~30 lines of plain PyTorch (`S = S @ (I - b k k^T) + b v k^T`,
  or the chunked equivalent), load the HF weights directly, and verify against a handful of
  logits. This is a smoke-test-sized job. Only the whitening *retraining* (§4, claim B) goes
  to Slurm.

---

## 4. The falsifiable contrarian claim

The mission's option (c) is the right one, and it should be stated in the sign that is
opposite to the incumbent's advice. The incumbent advice, as of mid-2026, is *unanimous*:
GKN, MesaNet, PDN, OSDN, VLA, CCQ all say **track more of the key covariance, it always
helps** (PDN: WikiText ppl 45.76 -> 43.05 for DeltaNet-340M, 31.39 -> 28.43 for GDN).

**Claim C (recommended):** the per-head excess frame defect `A_h/A_null` predicts, before
you train anything, *which heads the expensive method helps and which it does not* — and a
substantial minority of heads (predict 10-25% at `A_h/A_null < 8`) gain **nothing**, so
covariance tracking can be restricted to the anisotropic heads at a fraction of the cost
with no quality loss. Contrarian sign: "the second-order method is pointless here, and I
can tell you where from one forward pass."
Falsifier: if per-head gains from PDN/GKN are uncorrelated with `A_h`, the diagnostic is
worthless. Requires the PDN or GKN counterpart checkpoint, so it is the *second* experiment.

**Claim B (whitening) — do not lead with this.** "Explicitly whiten the keys, then one
global gain replaces the learned gate" is precisely PDN + OSDN's architecture, already
trained at 340M and 1B on SlimPajama, with numbers. Proposing it in 2026 is proposing a
published architecture. The only unclaimed residue is the *negative* half: whitening should
make the per-token gate **redundant**, so a whitened model with `beta` frozen to a constant
should match a whitened model with a learned `beta`. Nobody has run that ablation and it is
a genuine, cheap, contrarian test — but it needs two training runs (cluster).

**Claim A (gate-removal damage) is the §3.2/3.3 measurement** and is the entry ticket to
both B and C. Run it first.

---

## 5. Traps

**Trap 1 (fatal to the pitch as written): the tight-frame result derives LINEAR ATTENTION,
not the delta rule.** Batch posterior with `x ~ N(0, tau^2 I)`, `y_t = Wx + noise`,
`W^T W = cI`:

    mu_M = (1/tau^2 + Mc/sigma^2)^{-1} * (1/sigma^2) * W^T sum_t y_t

— a global scalar times a *pure accumulation* `sum_t W^T y_t`. That is
`S <- S + beta sum_t v_t k_t^T`: **Katharopoulos linear attention with a global gain**,
with no `-beta S k k^T` correction anywhere. The delta correction *is* the sequential
implementation of the whitening that the tight frame makes unnecessary. PDN Thm 3.1 says
this exactly: `S_t = C_t P_t`, and `P_t = I/c` under a tight frame, so the exactly
preconditioned delta rule *is* the linear-attention state rescaled. So the honest statement
is the reverse of the brief's: **tight frame => you do not need the delta rule.** Pitching
this as "the delta rule is the Heisenberg update" invites the reviewer to notice that the
derivation actually recovers the architecture the delta rule was invented to replace.

**Trap 2 (confirmed, and the brief is right to fear it): order-invariance must never be
sold as an advantage.** The 2024-2026 arc is explicitly about *destroying* commutativity.
DeltaNet's whole claim over additive linear attention is the non-commuting Householder
product; DeltaProduct (2502.10297) buys expressivity by *adding more* non-commuting factors
(Cartan-Dieudonne: products of Householders span `O(d)`; DeltaNet with 2 layers and an
expanded eigenvalue range solves dihedral-group word problems, `S_3` included); Mamba-3
adds complex rotations for the same reason; 2603.01959 catalogues what diagonal SSMs
*cannot* track. Any sentence containing "order-invariant" as a selling point is an
instant desk-reject in this literature.

*But note the good news, and state it explicitly, because it is the strongest technical
point available:* **a tight frame is not an orthogonal basis.** For `n > d` a tight frame is
overcomplete and its elements are not mutually orthogonal, so `G = cI` does **not** make
`prod_t (I - beta k_t k_t^T)` commute. Tightness constrains the *aggregate second moment*,
not the pairwise inner products. So making the key frame tight costs **no state-tracking
expressivity** — it only removes the need for a per-token gain. That is the one place where
frame language earns its keep over "isotropic covariance", because frame theory is exactly
the theory of overcomplete non-orthogonal spanning sets.

**Trap 3: `beta_t` is also the negative-eigenvalue knob.** DeltaNet uses `2 beta_t` so
`I - 2 beta k k^T` can have eigenvalue `-1` (a true reflection), which is what buys the
state-tracking result. Freezing `beta` to a constant may destroy state tracking for reasons
that have nothing to do with Bayes. **The §3.3 intervention must control for this**: report
the mean gate level separately from its variance, and freeze at the head's own mean so the
reflection regime is preserved.

**Trap 4: `beta` is not the only gate.** In Gated DeltaNet / KDA / GDN-2 the decay `alpha_t`
(scalar / diagonal / decoupled erase+write) carries much of the adaptivity. A finding on
DeltaNet's `beta` may not transfer. Replicate on at least one gated checkpoint before
claiming anything general — and note GDN-2 (2605.22791) has *two* channel-wise gates, so
"a single global gain" is now three architectural generations behind.

**Trap 5: the keys are computed by the model, so `G` is not exogenous.** A head could be
anisotropic *because* the gate compensates. The correlation in §3.2 is therefore
associational; the §3.3 intervention is what gives it direction. Do not skip 3.3.

**Trap 6: known negative from the repo, which bites here.** "More heads is monotonically
worse". Modern linear attention is aggressively multi-head (16 heads at 1.3B; GQA-style
splits). Whatever is claimed must be a *per-head* statement, never an aggregate one, or it
collides with the repo's own result.

**Trap 7: an unbounded `Z`.** The repo's own §8b caveat — the §6 gating/trigger
construction does not transfer to the Gaussian branch as written. So the cancellation
identity, which is the sharpest thing in the programme, does **not** come along for the
ride into linear attention. This front is disconnected from the strongest existing result.

---

## 6. Paper or chapter? And scoop risk

**Chapter, not paper.** The theoretical content is a corollary of PDN Thm 3.1, restated in
frame vocabulary, with the Kalman reading already in GKN §6.4 and the closed-form gain
already Longhorn's. A main-conference submission whose contribution is "we renamed the
isotropic case and measured it" will not survive review. As a thesis section it is
excellent: it connects the repo's Gaussian branch to the live architecture literature,
it produces a real measurement on real checkpoints, and — crucially — it lets the thesis
report a *negative* honestly (Trap 1) rather than overclaiming.

Upgrade path to a workshop paper (NeurIPS Efficient-ML / ATTRIB style): §3.2 + §3.3 + Claim C
on two architectures, with the diagnostic shown to predict per-head PDN-over-DeltaNet gain.
That is publishable as "a cheap diagnostic for when second-order linear attention is
wasted".

**Scoop risk: HIGH, and the clock is short.** PDN (Apr 2026) already plots the key-Gram
spectrum of a pretrained DeltaNet-340M — they are one scatter plot (per-head Gram anisotropy
vs per-head gain from preconditioning) away from this. OSDN (May 2026), CCQ (May 2026),
VLA (May 2026), 2602.04852 (Feb 2026) are all in the same few square metres. The arXiv rate
on this exact family is roughly one relevant paper per month through 2026. If §3.2 is not
run within weeks it is not worth running.

---

## 7. Recommendation

1. **Run §3.2 today.** One forward pass, `fla-hub/delta_net-1.3B-100B`, plain-PyTorch
   recurrence, 384 heads. Report the `A_h/A_null` histogram against `A_null = 0.062` and
   `Spearman rho(A_h/A_null, CV[beta_t^{(h)}])`.
   - Predicted: median `A_h/A_null` in **50x-500x** (i.e. `PR_h ~ 0.03-0.25`, effective
     key rank 4-32 of 128) — trained key frames are far from tight, per CCQ's needle-subspace
     evidence and PDN's diagonal-Gram success; **10-25% of heads near-tight**
     (`A_h/A_null < 8`); `rho = +0.35 to +0.6`.
   - **Kill at `|rho| < 0.15`.** Then write the negative up in three paragraphs, cite
     Trap 1, and close this front.
2. If it passes, run §3.3 (head-wise gate freezing, ratio > 5x).
3. Only then consider Claim C, which needs a preconditioned counterpart checkpoint.
4. **Reframe before writing anything up:** the result is "tight frame => the delta
   correction is unnecessary", not "the delta rule is the Heisenberg update". The second
   framing is both scooped and backwards.

## Sources

Preconditioned DeltaNet https://arxiv.org/abs/2604.21100 ·
Gated KalmaNet https://arxiv.org/abs/2511.21016 ·
Test-time regression https://arxiv.org/abs/2501.12352 ·
Longhorn https://arxiv.org/abs/2407.14207 ·
MesaNet https://arxiv.org/abs/2506.05233 ·
Kalman Linear Attention https://arxiv.org/abs/2602.10743 ·
OSDN https://arxiv.org/abs/2605.13473 ·
Variational Linear Attention https://arxiv.org/abs/2605.11196 ·
Curvature-Conditioned Query https://arxiv.org/abs/2606.01294 ·
Rank-based state reduction https://arxiv.org/abs/2602.04852 ·
Orthogonalized read scaffold https://arxiv.org/abs/2607.19390 ·
Gated DeltaNet https://arxiv.org/abs/2412.06464 ·
Gated DeltaNet-2 https://arxiv.org/abs/2605.22791 ·
Kimi Linear / KDA https://arxiv.org/abs/2510.26692 ·
RWKV-7 https://arxiv.org/abs/2503.14456 ·
Mamba-3 https://arxiv.org/abs/2603.15569 ·
DeltaProduct https://arxiv.org/abs/2502.10297 ·
Titans https://arxiv.org/abs/2501.00663 ·
Atlas https://arxiv.org/abs/2505.23735 ·
Parallelizing DeltaNet https://arxiv.org/abs/2406.06484 ·
fla-hub https://huggingface.co/fla-hub
