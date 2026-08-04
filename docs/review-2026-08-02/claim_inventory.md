# Claim inventory: original TB and QTB, with testability on PVSG

Built by reading both papers section by section: `papers/tb_original.pdf` (Tresp et al., *The Tensor
Brain*, arXiv 2109.13392) and `papers/qtb_LATEST.pdf` (*Bayes or Heisenberg: Who(se) Rules?*, the
July 2026 working version). Section, equation, table and figure numbers are as printed.

## How to read the columns

**Tested?** — what evidence the papers themselves provide.

| mark | meaning |
|---|---|
| **Q** | quantitative result in a table |
| **I** | illustration only — a figure, a t-SNE plot, or a hand-picked example |
| **A** | asserted analytically or conceptually; no experiment |
| **—** | not addressed experimentally at all |

**Test quality** — whether the reported evidence actually supports the claim, and where VRD-E/VRD-EX
was the wrong instrument. The recurring VRD problems:

- **VRD-EX "known entities" are affine-distorted copies of the same training image**, so
  "remembering an individual" and "matching a template" are not separable (§6.1, Fig. 4);
- **several attributes are synthetic** — `Young`/`Old` were *randomly assigned* by the authors, and
  `Dangerous` was defined as animacy (§6.1);
- **~15% of labels are the catch-all `Other`** (Table 3), so hierarchy accuracy is inflated;
- **one bounding box = one entity, mostly distinct classes**, so within-class individuation is never
  tested;
- **static images**, so every temporal claim is illustrated rather than measured.

**Infrastructure** — what it would cost *this* repository.

| mark | meaning |
|---|---|
| ✅ | ready: current manifests + existing code, needs only a runner or a metric |
| 🔧 | runner + small experiment-side model; no change to `src/tb` |
| 🏗 | needs a new core component (fast write path, decoder, allocator, gate) |
| 🚫 | out of reach on PVSG: needs an environment, agency, or new annotation |

---

## A. Perception

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| A1 | Index feedback improves perception: P-SA > P-Direct on unary labels | TB §6.5, Table 5 | **Q** (78.09 vs 77.97 on VRD-E — a 0.12 pt gap) | **Weak.** The gap is within noise, and P-Direct also sees less information and has fewer parameters | **Yes** — information-matched ladder, and the gap can be stratified by evidence quality | ✅ |
| A2 | The dynamic context layer is *essential* for binary-label prediction | TB §6.5, Table 4 (P-Direct 31.68 → P-SA 46.84 @1) | **Q** | **Confounded.** P-Direct's predicate decision sees only the union box while the BTN transports scene+subject+object. Varies information and mechanism together | **Yes** — M0→M5 ladder where each rung adds one mechanism | ✅ |
| A3 | With known entities, winner-take-all sampling (P-Samp) beats attention | TB §6.5, Table 5 VRD-EX (95.93 vs 95.12) | **Q** | **Instrument invalid.** "Known entities" are affine copies of training images, so this measures template matching | **Yes, decisively** — `blocked`/`fewshot` recur across real viewpoint, pose, scale and occlusion change | ✅ |
| A4 | Non-visual attributes are learnable *only* for already-known entities | TB §6.1, Table 5 `Y/O` (76.54 → 94.58) | **Q** | **Circular.** `Y/O` was *randomly assigned* by the authors, so it is by construction unlearnable except by memorizing the entity | **Partly** — no synthetic attribute needed: use category under degraded evidence (small mask, long gap) as the genuine deficit | ✅ |
| A5 | Zero-shot generalization of binary statements over class triples | TB §6.5, Table 6 (81.61 vs BFM 76.05) | **Q** | **Reasonable**, the strongest quantitative result in the paper | **Yes** — unseen (category, predicate, category) triples with the reviewed 4-level hierarchy and no `Other` catch-all | 🔧 |
| A6 | Perception is serial: one ROI per concept interval (one-brain hypothesis) | TB §8.6, QTB §11.5 | **A** | Architectural assumption, never ablated | **Yes** — 10.1 objects/frame lets you compare serial scan vs parallel fusion of the same evidence | 🔧 |
| A7 | The scene → subject → object → predicate order is part of the model | TB Alg. 1 | **A** | Never ablated | **Yes** — role-swap and window-permutation are evaluation-only on the pair manifests | ✅ |
| A8 | Symmetric `A` (shared bottom-up/top-down) costs ~1% versus untied | TB §7.6 | **A** ("extensive experiments", no table) | Asserted, no numbers reported | **Yes** — and it is directly interesting to a 2026 audience as the tied-embedding question | 🔧 |
| A9 | Perception uses episodic attention (EA) by default in all experiments | TB §5.3 | **A** | Never ablated; the repo's models omit EA entirely | **Yes** — episodic indices are just another named vocabulary group, no core change | 🔧 |

## B. Semantic memory

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| B1 | An index embedding is a **prototypical vector** for that concept | TB §7.5 | **I** (t-SNE, Fig. 7) | **Not tested.** A 2-D projection cannot distinguish a prototype from a discriminative classifier row | **Yes** — cosine between `a_k` and the feature centroid of `k`, tracked over training; hundreds of views per identity instead of 1–3 | ✅ |
| B2 | Embeddings organize into a conceptual space / cognitive map | TB §7.5 Fig. 7, §9.3 Fig. 8 | **I** | Qualitative only | **Yes** — embedding distance vs hierarchy tree distance, k-NN purity per level, domain separation | ✅ |
| B3 | Semantic decoding of a *class* index reveals what the class is about | TB §7.7, Table 7 | **I** (4 hand-picked rows) | Illustration, not a metric | **Yes** — run it over all 121 fine labels: is the correct basic/coarse label the argmax of its group? | 🔧 |
| B4 | Semantic memory supplies non-visual labels to perception (P-enriched) | TB §7.8, Table 9 (52.01 → 98.24) | **Q** | **Degenerate.** `Dangerous` was defined as *animacy*, which the model already predicted at ~98% | **Yes** — mask-area / recency strata give a real deficit at zero annotation cost | ✅ |
| B5 | Semantic memory achieves near-perfect recall given the entity index | TB §7.7, Table 8 (100% given entity) | **Q** | **Tautological.** Memorization by construction, as the caption itself concedes | Not worth repeating; report as a capacity ceiling instead | ✅ |
| B6 | Semantic memory = activating `ā`, or equivalently all instance indices | TB §7.1, §5.3 | **A** | Analytic identity, never checked numerically | Yes, cheaply — compare the two constructions | 🔧 |
| B7 | Sparse distributed embeddings increase memory capacity | TB §7.6 (Lasso → 70% sparsity) | **A** | Sparsity reported; **capacity never measured** | **Yes** — sweep `state_dim` × identity count × sparsity; PVSG has 7,143 identities vs VRD's 26,430 single-view ones | 🔧 |
| B8 | Semantic memory integrates multiple modalities | TB §7.8, Table 10 (VRD-S) | **Q** | On a **synthetic** social network the authors generated | **Partly** — PVSG captions/descriptions are real text, but provenance discipline is required | 🔧 |
| B9 | Generalized statements = probabilistic class-level rules | TB §3.8, Eq. 6 | **I** (Table 7) | Illustrated | **Yes** — the reviewed hierarchy gives clean `(fine, hA, basic)` statistics | 🔧 |

## C. Episodic memory

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| C1 | Activating an episodic index restores its engram and decodes the scene | TB §9.3–9.4, Table 12 (EM row), Fig. 9 | **Q + I** | Reasonable for what it is, but the "episode" is a static image | **Yes** — an episode can be a real temporal interval (the `relation_state_episode` convention already exists) | 🔧 |
| C2 | **Entity indices are required for recall** (`P-noI` ablation) | TB Table 12 bottom row (0.0 @10) | **Q** | **The paper's cleanest ablation.** Genuinely convincing | **Yes** — repeat with real recurrence rather than affine copies | 🔧 |
| C3 | Episodic embeddings form meaningful maps; similar scenes cluster | TB §9.3, Fig. 8 | **I** | Qualitative | **Yes** — same metrics as B2, plus same-video/same-scene-type retrieval precision | 🔧 |
| C4 | Recent episodic memory supplies state that is not currently perceivable ("lurking bear") | TB §9.6, Figs. 10–11 | **I** | **Illustration only — never measured.** This is one of the paper's headline ideas | **Yes, and this is the flagship opportunity.** PVSG has real occlusion: 135,440 pair records have an annotated relation while the subject's mask is *absent* | ✅ |
| C5 | Recency is implemented as a time encoding, analogous to position encoding | TB §9.6 | **A** | Proposed, never implemented | **Yes** — `score_k += λ·recency(k)`; sweep λ from similarity- to recency-driven recall | 🏗 (small) |
| C6 | Remote episodic memory retrieves *similar* past situations | TB §9.7, Fig. 12 | **I** | Qualitative | **Yes** — retrieval precision against video/scene identity, and against held-out similarity | 🔧 |
| C7 | Recall of recent vs remote memory is triggered by different mechanisms (recency+relevance vs similarity) | TB §9.6 vs §9.7 | **A** | **Not distinguishable in code at all** — indices carry no timestamp | **Yes** — the λ sweep in C5 turns the distinction into a controlled variable | 🏗 (small) |
| C8 | The value of a memory (reward, threat) is integral to the engram | TB §9.7 | **—** | Not implemented | **No.** Reward is ill-defined for a passive observer of someone else's kitchen video. Do not annotate PVSG for this | 🚫 |
| C9 | Future episodic memory: forecast events, then treat as ordinary memories | TB §9.8 | **—** | Not implemented | **Partly** — relation anticipation is well-defined on PVSG but sits on the censored-span problem | 🔧 |
| C10 | The post-observation model is a Dirichlet blend of episodic and semantic | TB §3.7, Eq. 5, Table 1 | **A** | Never evaluated as a predictive model | Yes, but low priority — it is a formal device rather than a mechanism | 🔧 |

## D. Reasoning and symbolic processing

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| D1 | Embedded symbolic reasoning: sampling `Dog` lifts `Mammal` toward 100% | TB §8.3, §5.2 | **I** (Table 7) | Illustrated on 4 rows | **Yes** — the disjoint 4-level hierarchy makes chain consistency measurable over all labels | 🔧 |
| D2 | A triple is a sequential index pattern; the brain "talks to itself" in triples | TB Table 2, §5.2, §8.7 | **A/I** | Conceptual | Yes — chain decoding on PVSG triples | 🔧 |
| D3 | Anomaly signal: a triple frequent in episodic but rare in semantic memory is "remarkable" | TB §8.1 | **—** | Never tested | **Yes** — a genuinely nice cheap experiment, and it is the closest thing to a surprise signal available without a decoder | 🔧 |
| D4 | Embedded vs symbolic vs embedded-symbolic reasoning are distinct modes | TB §8.2–8.3 | **A** | Conceptual taxonomy | No advantage from PVSG | — |

## E. Learning, consolidation, forgetting

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| E1 | New indices are established by a **fast, non-parametric** system (CLS) | TB §10.1–10.2 | **A** | **The gap between claim and implementation is total**: "establish a new index" is realized as a training run. One timescale, not two | **Yes** — `a_new ← sig(q)` normalized, one-shot, then `blocked`/`fewshot` evaluation | 🏗 |
| E2 | Self-supervised learning with pseudo-labels improves class/attribute embeddings | TB §10.3, Table 13 (74.78 → 75.08) | **Q** | Real but a **0.3 pt** effect | **Partly** — a cheap precondition test: pseudo-label precision vs confidence threshold | 🔧 |
| E3 | Catastrophic forgetting does not appear | TB §10.1 | **—** | **"Does not show up in our preliminary experiments" — no experiment is reported.** Also untestable in the regime claimed (fast write + slow replay), because there is no fast write | **Yes** — 394-video continual stream is a natural benchmark | 🏗 |
| E4 | Consolidation by **replay of index activations**, not of data | TB §10.4 | **—** | Never implemented. A real, unusual, unbenchmarked algorithm | **Yes** — no-replay vs feature-replay vs activation-replay on the video stream. Matches feature replay at a fraction of storage would be quotable independent of TB | 🏗 |
| E5 | Semantic memory forgets automatically via the `1/N_total` normalizer | TB §10.5, Eq. 4 | **A** | Analytic property of the count model, not of the neural model | Neutral | — |
| E6 | Unconsolidated MTL indices are eventually reclaimed (forgetting) | TB §10.6 | **—** | Not implemented | **Yes**, but only meaningful *after* a measured capacity curve exists | 🏗 |
| E7 | Novelty detection: an entity is new if all identity activations fall below threshold | TB footnote 6 | **—** | Never tested; it is the paper's own recruitment criterion | **Yes** — precision/recall of novelty detection on `heldout_video`, where every identity is genuinely novel. Cheap and directly falsifiable | 🔧 |

## F. Measurement theory and order effects (QTB)

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| F1 | The TB is a neural approximation to a probabilistic quantum computer (HB-POVM) | QTB §10 | **A** | Derivation, not an empirical claim | n/a | — |
| F2 | `a_{0,k}` is the **Bernoulli log-partition normalizer** `−Σ_ℓ softplus(a_{ℓ,k})` | QTB Eq. (25) applied at Eq. (34) | **A** | Derived; never compared empirically against a free bias or no bias | **Yes** — the three-way `direct` / `softplus-bias` / `learned-bias` comparison, already wired | ✅ |
| F3 | Gate regimes: `(α,β) = (0,1)` PVM, `(1,1)` HB-POVM, `(1,0)` gRNN | QTB §12.2.2 | **Q** (partly, via F4) | Only the two endpoints, on ImageNet | **Yes** — a gate phase diagram on real sequences, with intermediate values | ✅ |
| F4 | **PVM shows order effects; HB-POVM is order-invariant** | QTB §12.2.5, Table 4; App. 22 Tables 5–6 | **Q** — 100k ImageNet, 200 fine / 16 coarse, ResNet-50, KL 20.54 vs 0.304 | **The best-evidenced claim in either paper.** But: single-ROI whole-image classification, class embeddings set to *feature averages* rather than learned, and only 2 chain positions | **Yes, and this is a clear improvement**: 4 hierarchy levels instead of 2, learned embeddings instead of feature means, **chain-length scaling** (predict PVM sensitivity grows with length while HB-POVM stays flat), and *tree consistency* instead of reversal rate | 🔧 |
| F5 | Causal postselection induces an order effect even in the HB-POVM *likelihood* | QTB §12.2.3 | **A** | The text points at Tables 5–6, but those measure PVM vs HB-POVM, **not** postselection. The claim is effectively unevidenced | **Yes** — restrict fine candidates to descendants of the coarse outcome; this is also a real compute argument (7,143 → ~20 candidates) | 🔧 |
| F6 | Identifying **specific concepts before general ones** improves performance | QTB §12.2.4 | **A** ("empirically, we observe") — no table | Asserted without evidence | **Yes** — and adding *identity as level zero* is a novel hypothesis: does measuring the individual first suppress order effects even under PVM? | 🔧 |
| F7 | Relational asymmetry: `P(John likes Mary) ≠ P(Mary likes John)`, mediated by the evolution operator | QTB §12.2.6; TB §4.5 | **—** | Never tested in either paper | **Yes — the single sharpest untested claim.** Role-swap on the pair manifests is evaluation-only and falsifiable in both directions | ✅ |
| F8 | Measurement is generative: sensors excite, indices measure | QTB §12.1.1 | **A** | Interpretive | No | — |
| F9 | Brain stochasticity comes from generative sampling / stochastic winner-take-all | QTB §12.1.2 | **A** | Interpretive | No | — |
| F10 | A reservoir of unconnected indices is recruited for new memories | QTB §12.3.1 | **—** | Not implemented | **Yes** — and it is nearly free: a reserve column suffix, not an allocator | 🏗 (small) |
| F11 | Distributed indices could replace localized ones, at the cost of local optima | QTB §12.3.2 | **—** | Not tested | Defer — localized indices *are* the theory, and there is no capacity baseline to beat | — |

## G. Embodiment, grounding, multimodality

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| G1 | **Embodiment**: `ν̂_k ← g⁺(sig(a_k))`; `ν_k → a_k → ν̂_k` is an autoencoder | QTB §10.8; TB §4.1, §7.5 | **—** | **Specified with a name, a type signature and a purpose — and never built.** Currently unfalsifiable | **Yes** — decode to DINO feature space and measure nearest-neighbour retrieval. And it is nearly free: once `g` is a learned linear map, `g⁺` is its pseudo-inverse | 🏗 (small) |
| G2 | Top-down index activation reaches *earlier perceptual layers* | TB §7.5; QTB §11.2 | **—** | No observable exists | **Partly** — G1 gives the feature-space version; the pixel version is a demo, not science | 🏗 |
| G3 | Inputs from many brain regions are gated and summed: `g(ν) = Σ_k μ_k g(ν_k)` | QTB §11.4, Eq. 46 | **—** | Not tested | **Partly** — scene / object / union are three sources with learnable gates; reward and language are not available | 🔧 |
| G4 | Several indices may be active in one concept interval: `q ← Wh + g(ν) + Σ_{k∈S} a_k` | QTB Eq. (47) | **—** | Not implemented; the code injects a single index | **Yes** — inject identity *and* hierarchy labels together and measure interference | 🔧 |
| G5 | Concept grounding: an index excites modality-specific maps (appearance, sound, motor) | TB §7.4 | **A** | Neuroscience argument | Vision only | 🚫 |

## H. Architecture, workspace, consciousness

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| H1 | The representation layer cannot hold a *concatenation* of two embeddings (one-brain) | TB §8.6 | **A** | Architectural axiom | **Indirectly** — F7's role swap is its observable consequence | ✅ |
| H2 | Serial processing + central bottleneck; 3–4 items in working memory | TB §8.5 | **A** | Conceptual | **Yes** — measure how many entity identities survive to the predicate window as chain length grows. A real capacity curve for the "bottleneck" | 🔧 |
| H3 | The representation layer is the global workspace; CBS ≈ conscious experience | TB §8.4; QTB §13.6 | **A** | Philosophical | No | 🚫 |
| H4 | TB is a state-space model; RNNs are a special case; LSTM-like gating | QTB §13.1 | **A** | Analytic | Neutral; the evolution-backend comparison is the practical version | 🔧 |
| H5 | Index ↔ token, evolution net ↔ transformer stack, memory ↔ RAG | QTB §13.2 | **A** | Analogy | **Yes, as framing** — the shared `A` *is* tied input/output embeddings; A8 and B1 are that question asked of LLMs | 🔧 |

## I. Decisions, actions, planning

| # | Claim | Where | Tested? | Test quality | PVSG better? | Infra |
|---|---|---|---|---|---|---|
| I1 | **Actions are generated as any other index**; the same column recognizes and performs | QTB §12.2.1, §13.4, §13.5.1 | **—** | Strong, attractive, and entirely untested anywhere in this literature | **No.** PVSG is passive and third-person. Structurally untestable | 🚫 |
| I2 | Outcomes change the world (crocodile index → fleeing) | QTB §12.2.1 | **—** | Not tested | **No** | 🚫 |
| I3 | Near-term planning by unrolling the evolution operator (CoT analogue) | QTB §13.5.2 | **—** | Not tested | **Partly** — an unrolled `q` trajectory can be scored against *actual future frames*, but only once G1 exists to make it comparable to anything | 🏗 |
| I4 | Long-term planning over imagined scenarios and their dependencies | QTB §13.5.3 | **—** | Not yet a specification | **No** | 🚫 |
| I5 | Memory guides behaviour: remote episodic memory supports decisions | TB §9.7, §9.9; QTB §13.4.1 | **I** (the "Mary at the office" narrative) | **Narrative only** | **No** — needs counterfactuals, i.e. an environment | 🚫 |

---

## Summary of the scoring

| | count |
|---|---|
| Claims catalogued | **57** |
| With a quantitative test in either paper | **14** |
| Illustration only | **11** |
| Asserted with no experiment | **21** |
| Not addressed at all | **11** |
| Of the 14 quantitative tests, judged **compromised by the VRD instrument** | **6** (A1, A3, A4, B4, B5, and partly A2) |

**The single most useful thing this table shows:** the framework's evidential base is thinner than
its reputation. Only 14 of 57 claims have a number attached, and nearly half of those are undermined
by the dataset — most severely the memory claims, where "known entities" were affine copies and the
decisive non-visual attribute was randomly assigned.

## The eight highest-value targets

Ranked by (importance of the claim) × (improvement PVSG offers) ÷ (cost), restricted to ✅ and 🔧.

| Rank | Claim | Why |
|---|---|---|
| 1 | **F7** — relational asymmetry / role swap | Sharpest untested claim in either paper; evaluation-only; falsifiable both ways; directly tests the one-brain hypothesis (H1) |
| 2 | **C4** — recent memory supplies unperceivable state | Flagship idea, *never measured*; PVSG has 135,440 real occlusion records; already flagged in every manifest |
| 3 | **A3 / A4** — memory under real recurrence and real evidence deficits | Replaces the paper's two most compromised results (affine copies, random attributes) with genuine ones at zero annotation cost |
| 4 | **B1** — prototype versus classifier row | Settles a central interpretive claim with ~20 lines of analysis; VRD literally could not ask it (1–3 views per entity vs hundreds) |
| 5 | **F4** — order effects, extended | Best-evidenced claim, and the extension is clearly better: 4 levels not 2, learned embeddings not feature means, chain-length scaling, tree consistency |
| 6 | **A2** — dynamic context, information-matched | Repairs the paper's weakest headline with the ladder |
| 7 | **E7** — novelty detection | The paper's own recruitment criterion, never tested; `heldout_video` is exactly the right split; cheap |
| 8 | **C2** — `P-noI`, entity indices required for recall | The paper's cleanest ablation; worth re-running where recurrence is real |

## What PVSG cannot reach, and should be stated as such

C8 (memory value), I1–I2 (action indices, world effects), I4 (long-term planning), I5 (memory-guided
behaviour), G5 (multimodal grounding), H3 (consciousness). These need agency, counterfactuals, or
non-visual modalities. That is a large fraction of QTB §13 and TB §9.7–9.9. Say it once, precisely,
in future work — it is a stronger section than a list of things you ran out of time for.

## Claims that need a component first (🏗), ordered by unlock-per-cost

1. **G1 decoder `g⁺`** — small, and it is nearly free as the pseudo-inverse of a learned input map.
   Converts G1, G2, I3 and a surprise signal from unfalsifiable to measurable.
2. **F10 reserve columns + E1 one-shot write** — a reserve suffix, not an allocator. Unlocks E1, E3,
   E4, E6, C5, C7.
3. **C5 recency term on the score** — one added term; makes C7's recent/remote distinction a
   controlled variable instead of a narrative label.
