# Assessment of the repository and the experimental direction

## 1. The repository

**Quality: high, and unusually so for a thesis codebase.** This is not a courtesy sentence. The
specific things that are good:

- `src/tb` is 526 lines and I could check every operation against the paper equations without
  guessing. `attend` is `qS ← q̃S + A softmax(A^T sig(q̃S))`. `OriginalTBDynamicContext` is
  `h ← B sig[sig(h) + V sig(q)]`, `q̃ ← f(input) + W sig(h)`, with `h = 0` initialization and the
  additive input, exactly as Algorithm 1 has it. The predicate window correctly applies no feedback.
  That fidelity is the asset this project is built on, and it survived contact with a 400-video data
  pipeline without being deformed by it.
- The global-index versus candidate-position separation is a genuine correctness hazard in this
  model and it is handled explicitly, documented, and tested.
- `docs/fidelity.md`'s decision categories — paper-faithful / reasonable interpretation / deliberate
  modernization / experimental extension / suspected discrepancy — are a better methodological
  discipline than most published work has, and the ledger is actually maintained.
- The provenance discipline (pinned Hub revisions, annotation SHA-256, per-video provenance groups,
  written exclusion reasons, a materializer that refuses to overwrite a snapshot) is the part of
  this thesis a reviewer cannot attack.
- 84 tests pass, and they test equations and gradients rather than plumbing.

**The one structural weakness is asymmetry.** Roughly 3 500 lines of data machinery, 526 lines of
model, and **zero** lines of training, evaluation, or metrics. The abstraction quality of the data
layer is far ahead of the existence of the experimental layer. That is the whole problem, and it is
a sequencing problem rather than a quality problem.

## 2. Did you over-invest in data?

**Partly, and less than you think.** The honest split:

**Justified.** The extraction, materialization, exclusions, audit and protocol design are done, they
are correct as far as I can check, and they will not need to be revisited. 394 videos, 147 795
frames, 1 495 227 object observations, 8 330 261 pair observations, fully provenanced. You will not
spend another day on this. Most projects at this stage are still fighting their data.

**Premature.** The semantic property and relation inventory (49 unary values, 9 semantic relations,
grounded in CSLB/McRae/VAW/THINGSplus) was designed before a single model was trained. It is careful
work and it may not appear in this thesis at all. The four-level hierarchy is in better shape — it
will earn its place in E-A, E-B and E-C immediately — but it too was built ahead of any evidence
about which levels the experiments need.

**The generalizable lesson**, worth one line in your methods chapter: on a passive annotated
dataset, curation has no natural stopping criterion. Only a running model tells you which
distinctions matter. The fix is to get a model running early even when the data is imperfect, and to
let it tell you what to curate next.

**What I am *not* saying** is that the curation was wasted. Every artifact serves the object-first
program directly, and the pair materialization serves the one relation chapter.

## 3. The experimental direction

**PVSG over VRD is clearly right**, and for a sharper reason than "real video". VRD-EX's known
entities recurred only as affine distortions of themselves, so "the model remembers the individual"
and "the query looks like a stored template" are not separable in the original results. PVSG
individuals recur across genuine viewpoint, pose, scale and occlusion change, and the object table
carries the mask area and temporal gap needed to *stratify* by how hard each recurrence is. That is
not a better dataset; it is a different and much stronger experiment.

The second thing PVSG buys, which the existing plans underuse: **10.12 tracked objects per frame**.
Every frame is a natural sequence of ~10 perceptual acts over a shared scene — exactly the substrate
the dynamic context layer was theorized for, and exactly what VRD's single annotated pair could not
provide. It turns the paper's weakest headline claim from a two-point contrast into an accumulation
curve with a shuffled-frame control. Build the centerpiece on it.

**Three things the direction currently gets wrong**, all fixable in days:

1. **The comparison is not matched.** `PDirect` differs from `IntegralTB` in information, mechanism
   and parameter count simultaneously. The paper had this confound; there is no reason to inherit
   it when the fix is a five-rung ladder.
2. **The model does not perform the readout the memory claim is about.** Semantic labels must be
   read *after* index feedback. Currently nothing is.
3. **Feedback is numerically inert.** See issue S1. This one would have wasted a week and produced
   a confident null result.

## 4. On the three goals

**Goal 1 — test the TB's claims more rigorously.** Well served, and the plan's E-A and E-B are
strictly stronger designs than the originals. Be explicit in the write-up that PVSG masks and tracks
are *oracle*: the claim is about binding and memory *given* grouping, not about detection. Stated
plainly this is a strength; discovered by a reader it is a weakness.

**Goal 2 — extend the framework.** Two extensions are affordable and both are in the plan.
*Embodiment* becomes nearly free: once `g` is a learned linear map — which issue S1 forces you to do
anyway — its pseudo-inverse **is** the `g⁺` that QTB §10.8 specifies and nobody has built. *An
extensible index layer* is cheaper than the earlier planning assumed: you do not need an allocator,
you need to size `A` with a reserve suffix and write into it under `no_grad`. That reduces the
"keystone component" to a one-line vocabulary change and makes the one-shot Hebbian write affordable
inside three weeks.

What is *not* affordable, and should be said cleanly in future work rather than attempted: episodic
memory as its own chapter, consolidation by activation replay, forgetting, time-aware retrieval, and
anything requiring an environment. On that last point — PVSG is passive, third-person, fully
annotated video, so agency, reward, planning and "actions as ordinary indices" are **structurally**
untestable on it no matter how good the experiments are. Say that once, precisely; it is a better
future-work section than a list of things you ran out of time for.

**Goal 3 — insight for deep learning generally.** This is where the project is currently weakest in
framing and strongest in latent potential. Three claims are available, in descending order of how
much a 2026 DL audience will care:

1. **The measured cost of a discrete bottleneck.** M3 versus M4 in E-A is the same checkpoint, the
   same parameters, one line different: a soft mixture over learned embeddings versus a hard
   commitment to one of them, with feedback magnitude held matched. That is the question behind VQ,
   discrete latents and chain-of-thought-as-tokens, and in a transformer you cannot hold the
   mechanism fixed while swapping hard for soft. Here you can. This is the single most quotable
   result in the project and it is nearly free.
2. **Interventions on a genuinely symbolic bottleneck.** Enormous effort goes into *recovering*
   named, discrete, low-dimensional variables from transformers. This model has them by
   construction and nobody has intervened on them. "This model's chain of thought is a sequence of
   named symbols, and we can prove they are causally load-bearing" is a sentence no transformer
   result can state as cleanly.
3. **Tied embeddings.** The shared bidirectional `A` is structurally the same object as tied
   input/output embeddings in a language model. "Is a tied embedding a prototype or a classifier
   row, and does feeding it back into the residual stream help?" is E-C and E-A asked about LLMs.
   One paragraph, and it is the cheapest bridge from a cognitive-architecture thesis to mainstream
   relevance.

## 5. The three decisions I would make today

1. **Pivot to the object stream.** It fits in RAM (2.3 GB), it is three times larger than the usable
   pair stream, it carries nine of the paper's ten claims, and it discards every open data question
   at once. Keep relations as one chapter.
2. **Fix S1 before running anything.** Everything downstream is meaningless until
   `‖feedback‖ / ‖q‖` is in a reportable range and matched between P-SA and P-Samp.
3. **Freeze the data.** Three manifest-level regenerations (dev split, spread few-shot support,
   frame-grouped scan records), minutes of compute, then no more. The remaining risk is entirely in
   the experimental apparatus, and that is where every remaining day should go.
