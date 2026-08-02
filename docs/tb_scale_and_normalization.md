# Scale and normalization in the Tensor Brain

## Summary

There is not one normalization problem in the current configuration. There are **three**, they have
a common root cause, and only one of them has been fixed.

| | Problem | Status |
|---|---|---|
| **P1** | Input scale: plain L2 normalization pinned `σ(q)` at `0.5 ± 0.009`, so the CBS was effectively constant | **Fixed** in the working tree |
| **P2** | Feedback magnitude: `‖a_k‖ ≈ 1` writes into `‖q‖ ≈ 27.7`, a 3.7% perturbation, and expected feedback shrinks a further 70× | Open |
| **P3** | Score offset: `a0` initializes to zero, leaving a structural per-index offset **1.9× larger than the self-recall signal** | Open, newly identified, one-line fix |

A fourth question — whether a learnable linear map between DINO and the CBS simply resolves all of
this — is treated in Section 4b. Short answer: there are **three** constraints (CBS dynamic range,
readout scale, write magnitude) and only **two** knobs (input scale, `‖a‖`). A learnable map
replaces the second knob rather than adding a third, so it moves the trade-off without removing it.
A feedback gate or a readout temperature is what adds the missing knob.

The common root cause is that **`A` is asked to serve two roles whose scale requirements are
incompatible, and the mismatch grows roughly linearly with the representation dimension.**

This document explains why the problem arises, quantifies it, treats the asymmetric-transformation
proposal in detail, and ranks the available resolutions. Section 7 argues that the analysis is
itself a thesis contribution rather than only a bug fix.

---

## 1. Why the problem arises: the dual role of `A`

The Tensor Brain uses one matrix in two directions.

**Role 1 — readout (bottom-up).** `score_k = a_k^T σ(q) + a0_k`. The output must be a *logit*: its
natural scale is set by the softmax or BCE that consumes it, so useful values are O(1)–O(10).

**Role 2 — write (top-down).** `q ← α q + β a_k`. The output must be a *state increment*: its
natural scale is set by `q` itself, so to move the CBS meaningfully `‖a_k‖` must be comparable to
`‖q‖`.

These are different units. A logit is dimensionless; a state increment lives in pre-CBS
coordinates. One matrix cannot be natively scaled for both, and nothing in the architecture
reconciles them.

### 1.1 Dimensional analysis

Let `q` have component scale `s_q` and `a` have component scale `s_a`, in `D` dimensions, so
`‖q‖ = s_q √D` and `‖a‖ = s_a √D`.

Write `σ(q) = 0.5·**1** + δ`. In the near-linear regime `δ ≈ q/4`, so `δ` has component scale
`s_δ ≈ s_q/4`, saturating toward 0.5.

Then

```
score = a^T σ(q) = 0.5 · Σ_i a_i   +   a^T δ
                   └── offset ──┘   └─ signal ─┘
```

- The **offset** `0.5 Σ_i a_i` is a *per-index constant*. For zero-mean columns it has magnitude
  `0.5 s_a √D`. See Section 3 — this is P3.
- The **signal** `a^T δ` is `√D · s_a · s_δ` for a random column, but after training `a_k` aligns
  with the `δ` produced by its own class, giving `D · s_a · s_δ`. Alignment buys a factor `√D`.

**Readout requirement.** For logits of O(1) from an aligned embedding: `D · s_a · s_δ ≈ 1`, so
`s_a ≈ 1/(D s_δ)` and `‖a‖_readout ≈ 1/(√D · s_δ)`.

**Write requirement.** `‖a‖_write ≈ ‖q‖ = s_q √D`.

**The ratio:**

```
‖a‖_write / ‖a‖_readout  ≈  s_q √D · √D · s_δ  =  D · s_q · s_δ  ≈  D · s_q² / 4
```

**The conflict grows linearly in `D`.** Measured, with `s_q = 1` (the current RMS normalization):

| `D` | `‖a‖` wanted by readout | `‖a‖` wanted by write | ratio |
|---:|---:|---:|---:|
| 128 | 0.43 | 11.3 | **26×** |
| 256 | 0.30 | 16.0 | **53×** |
| **768** (current) | **0.17** | **27.7** | **159×** |
| 2048 | 0.11 | 45.3 | **427×** |
| 4096 (original paper's `r`) | 0.075 | 64.0 | **853×** |

This is the central result. It also means the problem is **not** a PVSG artifact and **gets worse
with scale** — at the original paper's `r = 4096` the two roles want norms three orders of magnitude
apart.

### 1.2 Why the original paper did not hit this

The original used a **trainable** VGG-19 / Faster R-CNN mapping `f(·)` from image to representation
layer, fine-tuned end to end. That encoder is a free scale parameter: gradient descent can adjust
its output magnitude until `‖q‖` sits wherever `A` needs it.

The modernization to **frozen DINO features with an identity input mapping** removed exactly that
degree of freedom. The conflict was always present in the mathematics; the trainable encoder used
to absorb it silently.

**This is the precise diagnosis, and it also names the most faithful fix:** restore a learnable map
in `g`, which is what the original effectively had.

---

## 2. P1 and P2, quantified

### 2.1 P1 — the input scale (fixed)

| | `‖drive‖` | `σ(q)` sd | `‖a_k‖` | `‖a_k‖ / ‖drive‖` |
|---|---:|---:|---:|---:|
| Previous (plain L2) | 1.00 | **0.0090** | 1.03 | **1.03** |
| Current (`√D · L2`) | 27.71 | **0.2080** | 1.03 | **0.037** |

Under plain L2 the drive had component scale `1/√D ≈ 0.036`, so `σ(q)` was pinned to `0.5 ± 0.009`
and the representation layer carried almost no information. The RMS fix raises the CBS dynamic
range 23-fold. **It was a real bug and the fix is correct.**

### 2.2 P2 — feedback magnitude (open, and widened by the P1 fix)

Note the last column. *Before* the fix, drive and index column were matched at norm ≈ 1 — both too
small, but symmetric. *After*, the drive is right and the index column is 27× behind.

Expected feedback is worse still: `Σ_k π_k a_k` averages near-orthogonal unit columns, so its norm
shrinks toward `1/√K_eff`.

| Identity candidates `K` (near-uniform) | `‖feedback‖` | relative to `‖q‖ ≈ 27.7` |
|---:|---:|---:|
| 100 | 0.096 | `3.5 × 10⁻³` |
| 1,000 | 0.030 | `1.1 × 10⁻³` |
| 4,000 | 0.015 | `5.5 × 10⁻⁴` |
| **P-Samp (single column)** | **1.03** | **`3.7 × 10⁻²`** |

So P-SA and P-Samp differ in feedback *magnitude* by up to 70×, which confounds the
expected-versus-sampled question with a scale question. Compounding it, `measure` exposes
`retain_gate` and `feedback_gate` but **`attend` exposes neither**, so the weaker path is the one
that cannot be corrected through the API.

---

## 3. P3 — the score offset, and a one-line fix

This one is new, and it is the most immediately actionable.

`a0` initializes to zeros. The offset term `0.5 Σ_i a_ik` is therefore left standing. For columns
with variance `1/D` it has, independently of `D`:

```
sd(offset) = 0.5 · √D · (1/√D) = 0.5
```

while the self-recall signal from activating index `k` alone is

```
a_k^T δ  where δ = a_k/4   →   ‖a_k‖² / 4 = 0.25
```

**The structural offset is 1.9× the signal.** Measured: offset sd `0.505`, signal `0.265`.

The consequence is stark. Setting `q = a_k` and scoring the full index layer, **top-1 recall of the
index that was just activated is 0.000** at initialization. The model cannot retrieve a concept it
is currently "thinking about," because a per-index constant swamps the match.

### 3.1 The fix, and why it is fully paper-faithful

The offset is *exactly* a per-index constant, so `a0` can cancel it exactly. Initialize

```
a0_k  ←  −0.5 · Σ_i A[i, k]
```

equivalently, score on the **centered CBS**, `(σ(q) − 0.5)^T a_k`, which is algebraically identical.

`a0` is in the paper's equation and is learnable, so this is an *initialization* choice, not a
change to the model. Without it, early training is spent learning to cancel a structural offset
twice the size of the signal.

### 3.2 What it unlocks: the capacity of the global workspace

With the offset absorbed, superposition works — and its limit becomes measurable. Setting
`q = Σ_{k∈S} a_k` for a random set `S` of size `m`, and asking whether the members of `S` are the
top-`m` scored indices (`D = 768`, `N = 4000`, untrained random columns):

| `m` | top-`m` recall |
|---:|---:|
| 1–16 | **1.000** |
| 32 | 0.942 |
| 64 | 0.757 |
| 128 | 0.593 |
| 256 | 0.476 |

**A 768-dimensional representation layer holds roughly 16–32 simultaneously active concepts at high
fidelity.** That is a direct, quantitative measurement of the global-workspace bottleneck the paper
invokes throughout Section 6.6 and Section 8.5 — and it has never been reported for this model.
Section 7 argues it should be an experiment in its own right.

---

## 4. The asymmetric-transformation proposal

The suggestion — introduce a transformation on one direction only, so the two roles stop competing
for one scale — is exactly right, and it is the principled family of fixes. The fidelity ledger
already sanctions it as the "asymmetric bottom-up index adapter."

**The invariant that must be preserved:** the *top-down* path must remain the direct embedding,
`q ← α q + β A[:, k]`. That is the paper's substantive claim — the selected symbolic index injects
its own embedding into the representation layer. Any transformation therefore belongs on the
**bottom-up** path only. This asymmetry is deliberate and is what makes the proposal fidelity-safe.

Three variants, in increasing order of departure.

### C1 — Learned inverse temperature (one scalar)

```
score_k = s · (a_k^T σ(q)) + a0_k          s > 0, learned
```

One parameter fully decouples logit scale from embedding norm. `‖a_k‖` becomes free to satisfy the
*write* requirement, while `s` independently satisfies the *readout* requirement.

**This is the minimal resolution of the entire conflict**, and there is a pointed observation
attached: the fidelity ledger records a deliberate decision, *"No inverse-temperature argument
initially — revisit if calibrated sampling, temperature sweeps, or exact historical experiment
protocols require it."* The parameter that was dropped as unnecessary is precisely the parameter
that reconciles the two roles. That is a clean, reportable finding about the design.

- **Fidelity:** reasonable interpretation. Inverse temperature is standard in the paper's own
  sampling formalism, and `s` is a global scalar rather than a learned representation.
- **Cost:** one parameter. Trivial.
- **Bonus:** `s` is directly interpretable as measured confidence sharpening, and it interacts with
  the calibration experiments.

### C2 — Learned bottom-up adapter

```
score_k = a_k^T φ(σ(q)) + a0_k             φ: ℝ^D → ℝ^D linear, or a small MLP
```

The readout path may now rescale *and* rotate, while the write path stays direct. Full decoupling:
`A`'s norm is set by the write role, `φ` handles everything the readout needs.

- **Fidelity:** experimental extension, already named in the ledger. It breaks the strict
  bidirectional symmetry of `A`.
- **There is a paper data point to compare against.** Section 7.6 reports: *"We did extensive
  experiments where we removed that constraint. The result was that the performance dropped by
  about 1%."* So the paper claims symmetry is nearly free. Measuring whether an asymmetric readout
  helps *once the scale conflict is removed* directly tests whether that 1% was about symmetry or
  about scale — a nice, self-contained result.
- **Cost:** `D²` parameters for the linear version. Moderate; also the most likely to simply absorb
  the task and obscure what `A` is doing, so it should be reported alongside C1 rather than instead
  of it.

### C3 — Normalized (cosine) readout with temperature

```
score_k = s · (â_k^T γ̂) + a0_k             â = a/‖a‖,  γ̂ = centered, normalized CBS
```

Readout depends only on *direction*; magnitude is left entirely to the write role. This is the
solution the retrieval and contrastive-learning literature converged on for exactly this problem —
an embedding used both as a lookup key and as a vector.

- **Fidelity:** deliberate modernization. It changes the scoring equation's form, so it must be
  named as such, not slipped in.
- **Cost:** trivial. Very stable in practice.
- **Caution:** normalizing away `‖a_k‖` removes a degree of freedom the model may be using to encode
  concept frequency or confidence, so report a frequency-stratified comparison against C1.

---

## 4b. Would a learnable linear map between DINO and the CBS just fix this?

It is the natural first thought, it is worth adding for other reasons, and **it does not fix the
scale conflict.** The reason is worth stating carefully, because it also explains what *does* fix
it.

### 4b.1 There are three constraints, not two

The framing so far has been readout-versus-write. There is a third, and it is the one the RMS fix
addressed:

1. **CBS informativeness.** `σ(q)` must have dynamic range, which requires component scale
   `s_q = O(1)`, hence `‖q‖ ≈ s_q√D`.
2. **Readout scale.** `c · ‖a‖ · ‖δ‖ ≈ L` for a target logit `L` and alignment `c`, where
   `δ = σ(q) − mean` is the discriminative part of the CBS.
3. **Write magnitude.** `‖a‖ ≈ ‖q‖`, so that index feedback actually moves the state.

Setting `W` learnable makes `‖q‖` a free parameter. But `‖q‖` was *already* a free parameter — it is
exactly what the normalization constant chose when you picked `√D · L2`. Making it learnable lets
SGD pick that number instead of you; it does not create a new degree of freedom.

**Count the knobs.** Input scale (1) and `‖a‖` (1) — two knobs against three constraints. The system
is over-determined, and no choice of `W` changes that.

### 4b.2 What the sigmoid pins

`W` is a full 768×768 matrix, so it clearly adds *representational* freedom — rotation, shaping,
conditioning. But it adds only **one scale** degree of freedom, because the sigmoid fixes the
relationships that matter:

- `‖σ(q)‖ ≈ 0.5√D` almost regardless of `q`, since σ maps into `(0,1)` centred near 0.5;
- `‖δ‖ ≈ s_q√D / 4` in the linear regime, saturating toward `0.5√D`.

So `‖δ‖`, which sets the readout scale, is slaved to `s_q`, which sets the CBS range. You cannot
move one without the other.

### 4b.3 The trade-off, measured

Sweeping `‖q‖` at `D = 768`, with logit target `L = 5` and alignment `c = 0.5`:

| `‖q‖` | `σ(q)` sd | `‖δ‖` | `‖a‖` for readout | `‖a‖` for write |
|---:|---:|---:|---:|---:|
| 1.0 | 0.0090 | 0.25 | 40.0 | 1.0 |
| 4.0 | 0.0359 | 0.99 | 10.1 | 4.0 |
| **6.3** | **0.0561** | 1.55 | **6.44** | **6.30** ← they meet |
| 15.0 | 0.1266 | 3.51 | 2.85 | 15.0 |
| **27.7** (current) | **0.2104** | 5.83 | **1.72** | **27.7** |
| 100.0 | 0.3885 | 10.76 | 0.93 | 100.0 |

The two requirements **do** cross, at

```
‖q‖* = 2 √(L/c)          →  6.32 for L = 5, c = 0.5   (table: 6.3)
```

but notice what it costs. At the crossover `σ(q)` sd is **0.056**, against **0.208** at the current
setting — a 3.7× loss of CBS dynamic range, moving back toward the very problem the RMS fix
resolved. And since

```
σ(q) sd at the balanced point  ≈  √(L/c) / (2√D)
```

**the achievable dynamic range at the balance point falls as `1/√D`** — 0.056 at `D = 768`, 0.025 at
`D = 4096`. The trilemma tightens with dimension, just as the readout/write ratio does.

So a learnable map lets gradient descent choose *which row of that table to sit on*. Every row
forces a trade. It cannot escape the table.

### 4b.4 What does escape it

One more knob. Either of the two already discussed:

- **A feedback gate `β`:** `q ← αq + β a_k`. Write becomes `β‖a‖ ≈ ‖q‖`, decoupled from `‖a‖`.
- **A readout temperature `s`:** `score = s · a^T σ(q)`. Readout becomes `s·c·‖a‖·‖δ‖ ≈ L`,
  decoupled from `‖a‖`.

Either gives three knobs for three constraints, and the system becomes solvable. This is the precise
sense in which C1 and F1 are root-cause fixes and a learnable input map is not.

**A concrete number falls out.** Staying at the current, good `‖q‖ = 27.7` with `σ(q)` sd 0.208, the
readout row wants `‖a‖ ≈ 1.72`. Making feedback comparable to the state then needs

```
β  ≈  ‖q‖ / ‖a‖  ≈  27.7 / 1.72  ≈  16
```

Even for a deliberately sub-dominant feedback of 20–50% of the state, `β ≈ 3–8`. **The default
`feedback_gate = 1` is off by roughly an order of magnitude**, which is the sharpest argument for
learning `β` rather than fixing it.

### 4b.5 Add the linear map anyway — for different reasons

None of the above is an argument against it. It is worth adding, just not as *the fix*:

- **Fidelity.** The original had a trainable VGG/Faster-R-CNN mapping. A learnable `g` restores the
  degree of freedom the original architecture actually had, which is the honest modernization.
- **It decouples `state_dim` from 768.** This matters more than it first appears: Section 3.2
  measures the workspace as holding ~16–32 concurrent concepts at `D = 768`. Without a projection,
  the workspace capacity of the model is hostage to DINO's output width. With one, `state_dim`
  becomes an experimental variable — which is a prerequisite for the capacity experiment (X2) being
  about the Tensor Brain rather than about DINO.
- **It can absorb the CLS-versus-pooled distribution difference** between scene, object and union
  evidence, ideally as separate per-role maps.

**One methodological hazard, and it is real.** A jointly learned `W` can silently compensate for a
badly scaled feedback path by reshaping `q`, which would make the symptom disappear without
explaining it. For a thesis whose contribution includes *characterizing* this conflict, that is
backwards. So:

**Do not make the input map the first change.** Run F0 and the diagnostic on the frozen identity
map, where the phenomenon is visible, then introduce the learnable map as a named condition that is
measured rather than assumed.

| | Fix | Addresses | Fidelity | Cost | Verdict |
|---|---|---|---|---|---|
| **F0** | `a0 ← −0.5 Σ_i A[i,k]` (centered CBS) | P3 | initialization only; paper's equation unchanged | one line | **Do immediately, unconditionally** |
| **F1** | Gates on `attend`, mirroring `measure`; then learned `β` | P2 | API completion; ledger already sanctions gate values | small | **Do next.** `β` becomes a measurement |
| **F2** | Learned inverse temperature `s` (C1) | root cause | reasonable interpretation | one parameter | **Do as the primary resolution** |
| **F3** | Learnable input map `nn.Linear(768, state_dim)` in `g` | **not the root cause — see 4b**; decouples `state_dim` from 768 | most faithful to the original's trainable encoder | small | **Run as a named condition, but not first** — it can mask the phenomenon |
| **F4** | Cosine readout + temperature (C3) | root cause | deliberate modernization | trivial | Extension; strong baseline |
| **F5** | Learned adapter `φ` (C2) | root cause | experimental extension | `D²` | Extension; tests the paper's 1% symmetry claim |
| **F6** | Normalized write `q + β a_k/‖a_k‖` | P2 | reasonable interpretation | trivial | Useful ablation next to F1 |
| **F7** | Re-initialize `A` to `std = 1` | P2 only | — | trivial | **Do not.** Pushes logits toward ±400 |
| **F8** | Sweep the input component scale `s_q` | P1 tuning | preprocessing | cheap | Worth one sweep; `s_q = 1` was reasoned but never tested |

### Recommended sequence

1. **F0 now.** It is free, it is inside the paper's equations, and without it a bare index cannot
   retrieve itself.
2. **Measure.** Log `‖a_k‖` by index group, `‖feedback‖/‖q‖` per window for P-SA and P-Samp, and the
   `σ(q)` versus `σ(a_k)` histograms. If `‖a_k‖` grows to within ~3× of `‖q‖` on its own, P2 may
   resolve itself — check before acting.
3. **F1, then F2.** The gate is the paper's own generalization; the temperature is the root-cause
   fix. Together they cost one line and one scalar.
4. **F3 as a named condition,** since it restores what the original architecture actually had.
5. **F4 and F5 as extensions,** reported against the F2 baseline.

Throughout: never apply a scale correction silently. Each is an experimental condition with a
fidelity category, and the ledger should record it.

---

## 6. What to report

- The `D`-scaling table of Section 1.1, with the measured values at `D = 768` and the projection to
  `r = 4096`.
- Before/after `σ(q)` statistics for P1.
- Feedback-magnitude ratios for P-SA and P-Samp against candidate-set size.
- The offset-versus-signal ratio and the zero-recall demonstration for P3, with the fix.
- The superposition capacity curve, before and after training — the interesting question is whether
  *learned* columns superpose better than random ones.
- A comparison of F1–F5 on the same downstream task, with calibration.

---

## 7. Why this is a contribution, not only a fix

Three reasons this belongs in the thesis as a chapter rather than a footnote.

**It is a general property of tied bidirectional weights, not a PVSG artifact.** The same structure
appears wherever one matrix serves as both a readout and a write: most directly in **tied
input/output embeddings in language models**, where the identical conflict is handled by the
`√d_model` multiplier on the embedding path introduced in the original Transformer, and where
untying embeddings is a known and measurable trade-off. The Tensor Brain literature has not
connected to this, and the connection cuts both ways — it lends the TB a well-understood
precedent, and it lets the TB contribute a cleaner analysis, because its two roles are explicitly
separated operations rather than an implementation convenience.

**The `D`-linear scaling is a genuine limitation of the shared-`A` design.** A model whose two
weight roles diverge in required scale proportionally to its representation dimension does not
scale gracefully. Stating that precisely, with the table, is a real critique of the architecture
and is exactly the kind of thing a thesis should establish rather than assert.

**The workspace-capacity measurement is new.** The paper invokes the representation layer as a
global workspace throughout, and Section 8.5 argues that a single global layer forces
serialization. The number — roughly 16–32 concurrent indices at `D = 768` for random columns,
degrading smoothly beyond — is a direct quantification of that bottleneck, it costs almost nothing
to produce, and it connects to global-workspace theory and to current work on superposition and
feature capacity. It should be run as an experiment in its own right, with random versus trained
columns as the controlled variable.
