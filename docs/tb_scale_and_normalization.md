# Scale and normalization in the Tensor Brain

> **Revision note.** An earlier version of this document claimed a broader architectural failure
> than the evidence supports, and prescribed fixes before measurement. Reviewer feedback was
> substantially correct and this version incorporates it. The three material corrections: the
> feedback-magnitude argument was built on the wrong diagnostic (§2.2); the proposed `a0` fix is the
> first-order term of a bias QTB actually derives, and is not "fully paper-faithful" as claimed
> (§3.1); and the dimensional analysis is specific to the *sigmoid* scoring path, so it does not
> extrapolate to the original implementation (§1.2). **No behavioral change is recommended before
> the diagnostics in §5 are run.**

## Summary

| | Problem | Status |
|---|---|---|
| **P1** | Input scale: plain L2 normalization pinned `σ(q)` at `0.5 ± 0.009` | **Fixed** in the working tree; the fix is correct |
| **P2** | Feedback magnitude | **Open, not established.** Norm ratio was the wrong diagnostic — see §2.2 |
| **P3** | Score offset from zero-initialized `a0` | **Real**, reproduced. Recommended resolution: QTB's derived softplus bias plus a learned residual — see §3.2a |

**Headline recommendation (§3.2a):** use the QTB-derived bias `a0,k = −Σ_ℓ softplus(a_ℓ,k)`. Its
gradient is the Bernoulli residual `γ_ℓ − σ(a_ℓ,k)`, of which the centered score is the
small-weight limit, and its fixed point `a = logit(γ)` sits at `‖q‖` — so it resolves the offset
**and** removes the readout/write scale pressure that §1 describes. Much of the conflict analysed
below is an artifact of dropping the log-normalizer from QTB's own derivation.

The organizing hypothesis — that `A` serves a readout role and a write role whose scale
requirements differ — remains useful and is worth measuring. It is **not** a proven architectural
theorem, and §1.2 explains why it does not describe the original implementation.

---

## 1. The organizing hypothesis: the dual role of `A`

**Role 1 — readout.** `score_k = a_k^T σ(q) + a0_k` must produce a *logit*, whose scale is set by
the softmax or BCE consuming it.

**Role 2 — write.** `q ← α q + β a_k` must produce a *state increment*, whose scale is set by `q`.

These have different natural units, and nothing in the architecture reconciles them. That much is
structural. What is *not* established is that this creates a practical failure — see §2.2.

### 1.1 Dimensional analysis, scoped to the sigmoid path

With `‖q‖ = s_q√D`, `‖a‖ = s_a√D`, and `σ(q) = 0.5·1 + δ` where `δ ≈ q/4` in the linear regime:

```
score = a^T σ(q) = 0.5 · Σ_i a_i   +   a^T δ
                   └── offset ──┘   └─ signal ─┘
```

For an aligned column the signal is `D · s_a · s_δ`, so readout wants `‖a‖ ≈ 1/(√D · s_δ)` while
write wants `‖a‖ ≈ s_q√D`. The ratio is `≈ D · s_q²/4`, growing linearly in `D`:

| `D` | `‖a‖` readout | `‖a‖` write | ratio |
|---:|---:|---:|---:|
| 128 | 0.43 | 11.3 | 26× |
| 256 | 0.30 | 16.0 | 53× |
| **768** (current) | **0.17** | **27.7** | **159×** |
| 2048 | 0.11 | 45.3 | 427× |

**This applies to the sigmoid scoring path only**, because both the `0.5` offset and the `δ ≈ q/4`
slope are properties of σ. It characterizes *this repository's* paper-faithful configuration. It is
not a general property of tied bidirectional weights, and the earlier version's extrapolation to the
original model's `r = 4096` has been removed as unsound.

### 1.2 Why this does not describe the original implementation

The earlier version claimed the original avoided the conflict only because its trainable VGG encoder
absorbed it. That is incomplete, and the sigmoid-specific parts do not transfer at all. The
original's configuration differs in at least four material ways, two of them corroborated by this
repository's own fidelity ledger:

- **LeakyReLU before index scoring**, not sigmoid (ledger, "Sigmoid in the reference experiment
  path"). With LeakyReLU there is no `0.5` mean, hence **no offset term and no P3 at all**.
- **Kaiming-initialized index weights**, not unit-norm columns (ledger, "Scaled Gaussian
  initialization of `A`").
- **No index bias.**
- **A learned scalar on index feedback** (`train_scale=True` in the authors' configuration), which
  is exactly the third knob §4b.4 identifies as missing.
- Plus the trainable VGG projection.

So the original had *more* free scale parameters than the earlier draft credited, and the offset
analysis is an artifact of choosing the sigmoid-faithful path. **The conflict described here is a
property of this repository's configuration, not of the Tensor Brain as published.** That is still
worth characterizing — it is the configuration the thesis runs — but the claim must be scoped.

---

## 2. P1 and P2

### 2.1 P1 — the input scale (fixed, correct)

| | `‖drive‖` | `σ(q)` sd | `‖a_k‖` | ratio |
|---|---:|---:|---:|---:|
| Previous (plain L2) | 1.00 | **0.0090** | 1.03 | 1.03 |
| Current (`√D · L2`) | 27.71 | **0.2080** | 1.03 | 0.037 |

Under plain L2 the CBS carried almost no information. The RMS fix raises dynamic range 23-fold.

### 2.2 P2 — the norm framing was wrong

The earlier version argued from `‖a_k‖ / ‖q‖ = 0.037` that feedback is too weak to inform the state.
**Euclidean norm ratio is the wrong sole diagnostic, and the reviewer is right about this.** Because
`A` is tied, a small update aligned with `a_k` is highly legible to `A^T`. Measured at
initialization (`D = 768`, `N = 4000`):

| Quantity | Value |
|---|---:|
| `‖a_k‖ / ‖q‖` | 0.037 |
| `‖σ(q + a_k) − σ(q)‖` | 0.217 |
| Natural discriminative score spread, sd of `A^T(γ − γ̄)` | 0.210 |
| **Own-logit change** | **0.218** — about one standard deviation |
| **Other-logit change, sd** | **0.0079** |

So feedback moves its *own* readout logit by ~1 sd while barely touching unrelated logits. A 3.7%
state perturbation is not functionally negligible.

**The reformulation that survives.** The quantity the theory actually needs is *cross*-readout
influence — the paper's claim is that recognizing Sparky biases the *unary* labels and transports
information into *later* windows, not that it reinforces its own logit. That quantity is

```
other-logit change sd / discriminative spread  =  0.0079 / 0.210  =  0.037
```

— numerically identical to the norm ratio. So the norm ratio predicts cross-readout influence
correctly while understating self-readout influence.

**But this is an initialization number and proves nothing about the trained model.** At init,
columns are mutually random. After training, semantically related columns should align, and
`a_j^T a_k` for related `j` could be far above chance — which is precisely the mechanism the paper
posits. Whether it happens is empirical. **P2 is an open question to be measured, not an established
defect**, and the earlier draft's confidence was unwarranted.

### 2.3 P-SA versus P-Samp: a distinction, not only a confound

The earlier version treated the P-SA/P-Samp magnitude gap as a confound to be corrected. That was
partly wrong: a noncommittal superposition *under uncertainty* is the intended semantics of expected
attention, and equalizing magnitudes would erase part of the distinction the two operations are
meant to draw.

The right treatment is to **measure rather than correct**, reporting feedback magnitude against the
effective candidate count

```
K_eff = 1 / Σ_k p_k²
```

together with entropy and the P-SA/P-Samp feedback ratio. If the gap tracks `K_eff` it is the
semantics; if it persists at low `K_eff` it is a scale problem. `attend` still lacks the gates
`measure` exposes, so adding them remains worthwhile — but as an **experimental extension**, not
"API completion", since QTB's formal attention uses coefficient one.

---

## 3. P3 — real, reproduced, and genuinely unresolved

`a0` initializes to zeros, leaving the per-index constant `0.5 Σ_i a_ik` standing. Independently
confirmed by both analyses:

| Quantity | Value |
|---|---:|
| Offset sd | 0.502–0.505 |
| Discriminative score sd | 0.210–0.213 |
| Self-recall@1, zero `a0` | **0 / 100** |
| Self-recall@1, centered `a0` | 100 / 100 |
| Self-recall@1, QTB-derived `a0` | 100 / 100 |

Setting `q = a_k` and scoring the layer, the model does not retrieve the index it is currently
representing. That is a real finding and it stands.

### 3.1 The fix is not a one-liner, and QTB already derives one

The earlier version proposed `a0_k ← −0.5 Σ_i A[i,k]` and called it "fully paper-faithful." **That
label was wrong.** QTB Equation 31 derives a bias of a different form:

```
a0,k  =  −Σ_ℓ log(1 + exp a_ℓ,k)  =  −Σ_ℓ softplus(a_ℓ,k)
```

obtained by Jensen's approximation to Equation 30. Expanding softplus for small weights:

```
−Σ_ℓ softplus(a_ℓ)  =  −D log 2  −  ½ Σ_ℓ a_ℓ  −  ‖a‖²/8  +  O(a⁴)
```

Verified numerically at `D = 768`: exact `−533.3330`; through the linear term `−533.2007`; adding
the norm correction `−533.3331` (residual `6 × 10⁻⁵`).

So the proposed centering is exactly the **first-order term** of the QTB-derived bias, missing a
global constant (harmless — it cancels in softmax) and a **per-index norm correction `−‖a_k‖²/8`**
(not harmless — it penalizes high-norm columns, which interacts directly with everything else in
this document).

Two further objections are correct:

- A one-time *initialization* stops cancelling as soon as `A` moves, whereas `a0(A)` tracks
  automatically.
- Permanently scoring `A^T(γ − 0.5) + b` is a **reparameterization** that changes gradients, not an
  initialization choice.

### 3.2 The controlled comparison this actually needs

Treat P3 as a suspected discrepancy with four named options:

1. current independent learned `a0`, zero-initialized (baseline);
2. independent `a0` initialized to `−0.5 Σ_i A[i,k]`;
3. centered-CBS reparameterization `A^T(γ − 0.5) + b`;
4. the QTB-derived `a0,k = −Σ_ℓ softplus(a_ℓ,k)`, optionally plus a learned residual.

The primary source is `papers/qtb_current_arxiv.pdf` (Eqs. 29–31); the later WIP makes `a0` less
explicit and should not settle the choice alone.

### 3.2a Option 4 does more than fix the offset — it dissolves the scale conflict

This is the most consequential finding in this document, and it partly retires §1.

Under the softplus bias, `score_k = γ^T a_k − Σ_ℓ softplus(a_ℓ,k)`. Differentiating:

```
∂score_k / ∂a_ℓ,k  =  γ_ℓ − σ(a_ℓ,k)
```

Two things follow, both verified numerically.

**The gradient is a Bernoulli residual, and options 2–3 are its small-weight limit.** Since
`σ(a) ≈ 0.5` for small `a`, the option-4 gradient reduces to `γ_ℓ − 0.5`, which *is* the
centered-score gradient. Measured, the difference `grad₄ − grad₃` correlates with `−a/4` at
**1.0000**. So option 3 approximates option 4's gradient exactly as option 2 approximates its bias.
**Option 4 is the exact object that 2 and 3 are linearizations of.**

**Its fixed point sits at the state scale, not the logit scale.** Setting the gradient to zero gives
`a_ℓ,k = logit(γ_ℓ)`. Optimizing a single column against a realistic CBS (`‖q‖ = 27.71`) converges
to

```
‖a‖ = 27.71        max|a − logit(γ)| = 7 × 10⁻³
```

— **exactly `‖q‖`.** The embedding is driven to the same scale as the pre-CBS state, which is
precisely the *write*-role requirement that §1 claimed was incompatible with the readout role.

The reason is that §1's analysis assumed an *unnormalized* bilinear score, where logit magnitude
scales with `‖a‖` and therefore pressures `‖a‖` downward. With the log-normalizer restored, the
score is scale-calibrated: growing `‖a‖` raises the linear term and the normalizer together, so
there is no readout-side pressure toward small columns. Score differences become
`score_k − score_j = KL(γ ‖ σ(a_j))` at the fixed point — a principled, dimension-scaling logit gap.

**So the "dual-role scale conflict" of §1 is substantially an artifact of dropping the
log-normalizer from QTB's derivation.** Restoring it is not a patch; it removes the tension at
source, and it simultaneously fixes P3 and plausibly P2. This does not eliminate the need for the
§5 diagnostics — it changes what they are expected to show.

**Caveat.** A derived `a0(A)` has no freedom to absorb class priors, and both the identity and
predicate distributions are heavily imbalanced. The practical default should therefore be
**softplus bias plus a learned zero-initialized residual** `b_k`: the derived term handles the
structural normalizer, the residual learns the prior. Plain softplus remains the theoretical
reference condition.

### 3.3 Superposition capacity — a diagnostic, not a capacity constant

With the offset absorbed, `q = Σ_{k∈S} a_k` recovers `S` as the top-`m` scored indices: recall 1.000
to `m = 16`, 0.942 at 32, 0.757 at 64, 0.593 at 128.

The earlier claim that "a 768-dimensional representation layer holds 16–32 concepts" **was too
strong.** The curve depends on vocabulary size, initialization, centering, the sigmoid, candidate
restriction, and the retrieval criterion. It is a reproducible *associative-superposition
diagnostic* for one configuration. Report it as such, with those factors as controlled variables and
random-versus-trained columns as the interesting contrast.

---

## 4. Candidate resolutions

The **invariant**: top-down feedback must stay the direct embedding `q ← αq + βA[:,k]`. Any
transformation belongs on the bottom-up path.

- **C1, learned inverse temperature.** Note the earlier formulation was wrong: the paper's inverse
  temperature multiplies the *complete score before softmax*, i.e. `softmax(s·(A^Tγ + a0))`, not
  `s·A^Tγ + a0`. A sweep is justified; calling it the root-cause fix is premature.
- **C2, learned bottom-up adapter** `a_k^T φ(γ)`. Experimental extension; the ledger sanctions it.
  Comparable against the paper's §7.6 report that removing weight symmetry cost ~1%.
- **C3, cosine readout.** Caveat the earlier version missed: normalizing removes **gradient pressure
  on `‖a_k‖` entirely**, so weight decay shrinks columns and the write role degrades. Without an
  explicit column-norm or write-magnitude rule it does not solve feedback scaling — it relocates it.
- **Learnable input map.** Adds rotation and conditioning, restores the original trainable
  perception boundary, and decouples `state_dim` from DINO's width — the last of which is a
  prerequisite for the capacity diagnostic being about the Tensor Brain rather than the encoder.
  It supplies one *scale* degree of freedom, not three; see §4b in the prior revision's argument,
  which stands. Hazard: jointly learned, it can absorb the phenomenon under study.
- **Learned feedback scalar `β`.** Historically supported — the authors' configuration has
  `train_scale=True`. This is the best-motivated single addition, but as an experimental extension.

---

## 5. Recommended next step: measure, do not fix

**Make no behavioral normalization change yet.** Add diagnostics recording, per concept window:

- `q` RMS, quantiles, and CBS saturation fraction;
- offset `a0 + 0.5 A^T1` versus discriminative score `A^T(γ − 0.5)`;
- `A` column norms, column sums, and bias–column-sum correlation;
- P-SA `K_eff`, entropy, feedback norm, and the P-SA/P-Samp feedback ratio;
- `‖σ(q + f) − σ(q)‖`;
- **target-margin and full-logit displacement caused by feedback** — the §2.2 diagnostic, which is
  the one that settles P2;
- gradient norms reaching `A` through scoring versus through feedback.

Run at initialization, during tiny-data overfitting, and after convergence. **Then** compare the four
scorer-bias variants of §3.2 and a learned feedback scalar as named ablations. Temperature or an
asymmetric adapter only if those measurements show an actual downstream limitation.

One sequencing note: the zero-recall result means semantic-decoding experiments that activate a bare
index (E6, and the capacity diagnostic) will return artifactual nulls on an untrained or
lightly-trained model. Run these diagnostics before those experiments, not after.

---

## 6. What remains a contribution

Scoped honestly, three things:

- **A quantitative characterization of the readout/write scale tension in the sigmoid-faithful
  configuration**, including the `D`-linear ratio and the trilemma of §4b. Scoped to this
  configuration, with the original's LeakyReLU/Kaiming/no-bias/learned-scale setup as the contrast
  that explains why the original did not face it.
- **The zero-`a0` offset finding and the four-way bias comparison**, including the observation that
  the natural centering is the first-order term of QTB's derived `a0` and omits a per-index norm
  correction. This is a genuine suspected discrepancy with a clean experimental resolution.
- **The associative-superposition diagnostic**, reported with its dependencies rather than as a
  capacity constant, with random-versus-trained columns as the contrast.

The connection to tied input/output embeddings in language models remains a useful framing, but as
an analogy motivating the diagnostics — not as evidence that the Tensor Brain inherits a proven
defect.
