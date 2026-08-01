# Missing components: what the Tensor Brain cannot currently do, and what it would take

## Purpose and relationship to the other documents

[The PVSG experiment program](pvsg_experiment_program.md) asks which *claims* are unsupported and
which experiments would settle them, using the model and data we already have.
[The QTB WIP review](qtb_wip_review_2026-07-22.md) asks which *equations* need small exact tests.

This document asks a third question: **which components does the framework not have at all, such
that a whole class of its claims is currently unfalsifiable rather than merely untested?**

That distinction matters. An untested claim needs a run. An unfalsifiable claim needs a
component built first — and building the right one changes what the project *is*, not just what
it has measured. Several of the Tensor Brain's most interesting propositions are in the second
category today, and a few of them are three days of work away from being in the first.

---

## 1. The synthesizing diagnosis

Reading the two papers against `src/tb`, the gaps are not scattered. They collapse into three
structural absences, and almost every missing component is downstream of one of them.

### 1.1 There is no write path

The Tensor Brain has a **read** path (`index → A[:,k] → q`) and a **slow learn** path (Adam on
`A`). It has no **fast write**: no way to create an index and set its embedding in one shot.

This is not an implementation detail. The entire complementary-learning-systems story of Section
10.1–10.2 — a fast non-parametric hippocampal system that establishes a new index and "copies the
episodic memory trace" into its connections, followed by slow neocortical consolidation — is a
claim about *two learning timescales*. The repository has one. Consequently:

- "establish a new episodic index" currently means "run a training job";
- episodic memory, novelty detection, one-shot identity enrollment, index recruitment,
  consolidation and forgetting are all inexpressible;
- the paper's claim that catastrophic forgetting does not appear is untestable in the regime where
  it is actually claimed to hold (fast writes, then slow replay).

Everything in Section 3 below marked *memory* depends on closing this.

### 1.2 There is no downward path

Perception runs strictly `features → q → index scores`. Nothing maps `q` or `A[:,k]` back toward
perceptual space.

This makes **embodiment unfalsifiable**. The claim in Sections 7.5 and 10.8 is that activating an
index grounds it — that top-down index activation reaches earlier processing layers and reinstates
something perceptual. With no decoder there is no observable to check. The same absence makes the
BTN-as-autoencoder framing (QTB Section 4.1) untested, and it makes imagination, future episodic
memory and planned rollouts impossible to evaluate, because an unrolled trajectory of `q` values
cannot be compared against anything.

### 1.3 There is no consequence, and therefore no surprise

Every operation in the current codebase is teacher-forced classification against an annotation.
The model never predicts something it could be *wrong about in its own terms*, and no outcome ever
changes anything outside the model.

This is the deepest gap, because **prediction error is the keystone signal for three separate
missing components at once**:

- *novelty detection* — the paper's own criterion for recruiting a new entity index (footnote 6:
  all identity activations below threshold) is a surprise signal;
- *episode boundary formation* — your perception document already anticipates that a learned
  boundary policy "may use prediction error, changes in `q` or dynamic context `h`";
- *the value of a memory* — Section 9.7 proposes that reward or threat is an integral part of an
  episodic engram.

None of the three can be built without something the model is trying to predict and something that
happens as a result. On a passive, fully-annotated dataset, both are absent by construction.

**The practical consequence of 1.1–1.3 taken together:** the Tensor Brain as implemented is an
unusually interpretable sequential classifier. The components below are what would make it a
memory system, and then an agent.

---

## 2. How to read the catalogue

Each component states what is missing, why it matters theoretically, what it unlocks, the sharpest
test it makes possible, and an honest cost. Feasibility ratings are relative to a master's project
with the current codebase and the PVSG snapshot in hand:

- **A — days.** No new data, small code, unblocks other work.
- **B — weeks.** Needs one of the A components first, or moderate new machinery.
- **C — months, or needs a new data source or environment.**
- **D — defer.** High intellectual interest, low near-term return.

---

## 3. The catalogue

### 3.1 Growable index vocabulary and a recruitment reserve — **A, keystone**

**Missing.** `TensorBrain.__init__` fixes `num_indices`, and `IndexVocabulary` is constructed once
from a manifest. There is no `grow()`, no allocation, no free list.

**Why it matters.** This is unglamorous and it silently blocks nearly every memory experiment:
few-shot enrollment, continual learning over the video stream, self-supervised index recruitment,
episodic indices, and capacity studies all require the vocabulary to change size at runtime.

**Design note.** QTB Section 12.3.1 actually describes the mechanism: "a reservoir of indices might
be recruited to form new memories. Initially they might be disconnected from the representation
layer with either no or a random connection pattern." A reserve pool — allocate `num_indices` up
front, mark a suffix as unallocated, hand out columns on request — is both the pragmatic
implementation and the paper-faithful one. It also preserves the property your fidelity ledger
already engineered for: a static prefix of `A` whose meaning does not shift when identities are
added later.

**Unlocks.** 3.2, 3.4, 3.5, and experiments B5, B7, B8 of the experiment program.

**Cost.** Small. The subtlety is optimizer state and checkpoint compatibility when columns are
allocated mid-run, not the allocation itself.

### 3.2 One-shot Hebbian write and a two-timescale `A` — **A, highest unlock-per-line**

**Missing.** A rule that sets a new index's embedding from the current state without backprop.

**Why it matters.** The scoring equation is `a_k^T σ(q) + a0_k`. The write that makes a new index
maximally responsive to the state that created it is therefore, up to normalization,
`a_new ← σ(q)`. That is not a heuristic bolted on: it is the direct reading of Section 10.2's
"quickly stored by establishing a new index and its connections to the representation layer,
copying the episodic memory trace." The framework's own mathematics names the write rule; nobody
has implemented it.

Doing so partitions `A` into a **slow region** (static semantic, hierarchy and predicate columns,
trained by gradient descent) and a **fast region** (episodic and identity columns, written in one
shot, optionally consolidated later). That is complementary learning systems expressed as a
property of one matrix, which is a genuinely elegant and genuinely novel thing to have.

**Sharpest test.** Write an identity column in one shot from a single observation, then ask three
questions that have never been asked of this model: (a) how does one-shot Hits@1 compare to
gradient-trained Hits@1 at k = 1, 5, 25 exposures; (b) does slow replay-based consolidation move
the fast-written column *toward* where gradient descent would have put it, or somewhere else; (c)
is the one-shot write a better initialization for consolidation than random?

**Modern context.** This is the Tensor Brain's version of fast weights and of modern Hopfield
memories — QTB Section 12.3.2 name-checks Ramsauer et al. itself. Framing it as "one matrix, two
timescales, symbolic addressing" is a clean and currently unoccupied position.

**Cost.** Roughly a day of code, plus the normalization decision (write `σ(q)`, or
`σ(q) − mean`, or a unit-norm version — this matters because column scale directly sets that
index's logit, and an unnormalized write will dominate the softmax).

**Risk to name.** A one-shot write is trivially good at recognizing the *exact* state that wrote
it. The evaluation must be temporally separated (the `blocked` protocol) or it will measure
template matching, exactly as VRD-EX did.

### 3.3 A decoder `g⁺` from state back to perceptual features — **A, converts three unfalsifiable claims at once**

**Missing.** Any map from `q` or `A[:,k]` back toward DINO feature space.

**This one is not even an inference — QTB specifies the component and it is simply not built.**
Section 10.8 states: "The (approximate) inverse mapping from index `k` to CBS `q = a_k` and back to
`ν_k` is an embodiment process. It could be realized by a top-down network implementing a function
where `ν̂_k ← g⁺(sig(a_k))` is the embodiment of the index `k`. The map `ν_k → a_k → ν̂_k` forms an
autoencoder structure that visually explains to the brain what an index is all about and that might
be useful for self-supervised learning." The extended original paper likewise describes the BTN
observation model as having "the characteristics of an autoencoder" (Sections 4.1 and 4.6), and
QTB Section 15.9 lists "Autoencoder Learning" as a named thesis project. The function has a name,
a type signature and a stated purpose in the source material. Nobody has written it.

**Why it matters.** Adding a small decoder — to the 768-dimensional DINO space, *not* to pixels —
simultaneously makes three separate claims measurable:

1. **Embodiment / grounding.** Activate the `dog` index with no visual input, decode, compare to
   the mean DINO feature of dogs and to held-out dog observations. If index embeddings are
   grounded prototypes as Section 7.5 claims, this should work; if they are arbitrary linear
   classifier rows, it should not. Nobody knows which is true today.
2. **The autoencoder framing.** Both papers describe the BTN observation model as an autoencoder,
   and QTB explicitly proposes `g⁺` as the missing half of it. Reconstruction quality has never
   been reported.
3. **Imagination and future episodic memory.** An unrolled evolution trajectory becomes a sequence
   of *predicted features* that can be scored against the actual future frames PVSG contains. This
   is the only way "reasoning about the future" becomes a number rather than a narrative.

**And it produces the keystone signal.** A decoder gives per-step prediction error, which is
precisely what novelty detection and episode-boundary policies need (see 1.3). Build this and two
later components stop being blocked.

**Cost.** Very low. A linear or two-layer head on cached features; no new data; trains in minutes.
Cosine and MSE against held-out features, plus a nearest-neighbour retrieval check (does the
decoded vector retrieve the right object?), which is a much more meaningful metric than raw MSE.

**Design boundary.** Keep it outside `src/tb`, exactly as the encoder is. It is part of `g`, not
part of the Tensor Brain.

### 3.4 Time-aware indices and a recency-relevance retrieval prior — **B**

**Missing.** Indices carry no timestamp, no age, no decay. `a0` is a static bias.

**Why it matters.** Sections 9.6 and 9.7 distinguish *recent* from *remote* episodic memory, and
state that recall is triggered by recency and relevance in one case and by similarity in the other.
With no temporal metadata on indices, **the two are not distinguishable in code at all** — the
distinction that carries much of Section 9's argument currently has no implementation.

**Design note.** The paper points at the mechanism: "this emphasis on episodic closeness can be
implemented as time encoding, in a similar way as position encoding is used in the attention
literature." The minimal version keeps the scoring equation intact and adds a term:
`score_k = a_k^T σ(q) + a0_k + λ · recency(k)`. Then `λ = 0` is remote/similarity-driven recall and
large `λ` is recency-driven recall, and the retrieval regime becomes a controlled variable rather
than a narrative label.

**Sharpest test.** On PVSG, sweep `λ` and measure which regime supports which task: state
maintenance across occlusion should favour recency; analogical retrieval of a similar past
situation should favour similarity. If one setting wins everything, the recent/remote distinction
is decorative, which is itself a finding.

**Cost.** Small, but it requires 3.1 and 3.2 to exist first so there are episodic indices to time-
stamp.

### 3.5 Consolidation by replay of index activations — **B, distinctive and modern**

**Missing.** Any consolidation mechanism.

**Why it matters.** Section 10.4 proposes something specific and unusual: activate an episodic
index, let the representation layer hold `a_t`, and let a *new* index learn from that activation by
Hebbian learning — explicitly noting the advantage that "there is no need for direct interactions
of indices in both storage sites, only indirect interactions by a shared activation of the
representation layer."

That is **replay of activations rather than replay of data**, and it is a real algorithm that has
never been implemented or benchmarked. It is also unusually well positioned for 2026: generative
replay, dataset distillation and privacy-preserving continual learning all want exactly this
property — rehearsal without retaining inputs.

**Sharpest test.** Run the continual-learning stream over the 394 PVSG videos with three
conditions: no replay, replay of stored features, and replay of index activations. If activation
replay matches feature replay at a fraction of the storage, that is a clean, quotable result that
does not depend on any Tensor Brain-specific claim being true.

**Cost.** Moderate, and it depends on 3.2.

### 3.6 Forgetting and eviction — **B**

**Missing.** Section 10.6 argues that unconsolidated episodic indices are eventually reclaimed
because MTL capacity is limited. No eviction policy exists.

**Why it matters.** It converts capacity from a fixed hyperparameter into a managed resource, and
it is the natural pair to the capacity-scaling experiment. Policies worth comparing: least
recently used, least frequently retrieved, lowest consolidation score, lowest downstream utility.

**Honest assessment.** Interesting, but it is only meaningful *after* a measured capacity curve
shows where interference actually begins. Sequence it after the capacity experiment, not before.

### 3.7 Action indices and an actuator boundary — **C, highest ceiling**

**Missing.** Any notion of an index whose measurement does something.

**Why it matters.** QTB is emphatic and repeated on this point — "actions are generated as any
other indices", "when nodes are action nodes then labels influence the real world", and Section
12.2.1's observation that activating a crocodile index may trigger fleeing while a log index does
not. The claim is strong and attractive: **no separate policy head is needed, because measurement
over a candidate group of action indices already is a policy, and the same column of `A` serves
both to recognize an action and to perform it.**

That shared-column claim is testable and, as far as I can tell, untested anywhere in this
literature. It also carries a real risk of being false, which is what makes it worth testing.

**What it requires.** An environment. This is the honest cost, and it is discussed separately in
Section 5 below, because it is the single most consequential strategic decision available here.

### 3.8 Reward and valence as an input modality and a memory tag — **C**

**Missing.** No reward signal, and no value attached to an episodic index.

**Why it matters.** Section 9.7 proposes that the value of a memory is integral to it, and that
remote episodic memory guides behaviour by retrieving similar past situations with their outcomes.
Equation 46's gated input sum already accommodates a reward channel and `integrate_input` already
supports it mechanically — what is missing is a signal and a value-weighted retrieval rule.

**Feasibility split worth stating clearly.** In a synthetic environment this is nearly free. On
PVSG it requires new annotation of dubious validity, since "reward" for a passive observer of
someone else's kitchen video is not well defined. Do not annotate PVSG for reward.

### 3.9 A learned write / episode-boundary policy — **C, the natural capstone**

**Missing.** Episodes are predetermined from annotations (`relation_state_episode`), as your
perception document specifies.

**Why it matters.** Deciding *when* to commit an experience to memory is the interesting version of
episodic memory, and your document already frames the comparison correctly: learned policy versus
annotation-derived boundaries versus fixed windows, controlled for number and duration of stored
memories, judged by downstream utility rather than by agreement with human boundaries.

**Why it is late, not early.** It is a composite. It needs a surprise signal (3.3), a write
operation (3.2), an allocator (3.1), and a downstream task whose utility can be measured. Attempting
it before those exist produces an unfalsifiable policy trained on a proxy objective. Recognize it
as the capstone it is.

### 3.10 Outcome-conditioned candidate construction — **A/B, undersold**

**Missing.** Candidate groups are static named sets. Nothing conditions the next candidate set on
the previous outcome.

**Why it matters, in two registers.** Theoretically, this *is* causal postselection, the mechanism
QTB says induces a likelihood order effect even in the HB-POVM regime, and it is the mechanism
behind the inattentional-blindness phenomenon. For a hierarchy it is nearly free to implement:
restrict the fine candidates to the descendants of the coarse outcome.

But there is a second register that the papers do not emphasize and that a deep-learning audience
will care about more: **it is an efficiency mechanism.** With ~7,143 identity columns, scoring the
full index layer at every window is the dominant cost. Outcome-conditioned restriction turns that
into scoring twenty candidates. So the same mechanism is simultaneously a cognitive phenomenon, a
source of a predicted order effect, and a real compute argument. That triple framing is the
strongest version of this component and it is currently absent from all three documents.

**Cost to name.** Error propagation. A wrong coarse outcome makes the correct fine label
unreachable, and the size of that penalty is exactly what the experiment should report.

### 3.11 An intervention and causal-analysis harness — **A, strategically underrated**

**Missing.** Tooling to intervene on intermediate symbolic outcomes and measure downstream effect:
ablate a measured index, substitute a wrong one, shuffle the order, or clamp a feedback vector.

**Why this may be the most strategically valuable cheap component.** The Tensor Brain's structural
peculiarity is that its intermediate variables are **named, discrete, low-dimensional and human
readable by construction**. A very large amount of contemporary interpretability work exists to
*recover* variables with those properties from transformers — activation patching, causal tracing,
sparse autoencoders, circuit analysis. The Tensor Brain has them for free and has never run the
corresponding experiments.

The claim that becomes available is sharp: *this model's chain of thought is a sequence of named
symbols, and we can prove they are causally load-bearing rather than decorative, by intervening on
them.* No transformer result can be stated that cleanly. That is a strong position, it costs a
small harness, and it makes every other experiment in the program more interpretable at no extra
compute.

**Sharpest tests.** Substitute the correct identity outcome with a wrong one and measure downstream
predicate degradation — if it is small, index feedback is decorative and the paper's central
mechanism is in trouble. Compare against the same intervention on a flat fusion model's hidden
state, where the intervention cannot even be *specified*.

### 3.12 A sparsity mechanism on `A` — **A, cheap and tied to an existing claim**

**Missing.** Section 7.6 reports that applying Lasso to all parameters yielded 70% sparsity and
argues, citing Rolls and Ma et al., that sparse distributed representations increase memory
capacity. There is no sparsity mechanism in the repository and the capacity claim was never
measured.

**Why it matters.** It pairs directly with the capacity experiment: sweep dimensionality and
identity count with and without sparsity, and test whether the paper's stated rationale for sparse
embeddings survives contact with data. Cheap, quantitative, and it either confirms or retires a
claim the paper makes in passing.

### 3.13 Multi-timescale evolution — **B**

**Missing.** One evolution operator applies at concept-window boundaries.

**Why it matters.** PVSG already gives three natural time bases: the concept window (subject →
object → predicate), the frame (5 FPS), and the relation-state episode. Section 9.8's note that
useful future memories may avoid long iterative rollouts is essentially a request for a coarse
transition that skips ahead. Comparing a fast per-frame transition, a slow per-episode transition,
and a learned shortcut to a future state is a well-posed architectural question with an obvious
video substrate.

**Cost.** Moderate. It fits cleanly behind the existing evolution contract, which is the point.

### 3.14 A language boundary at the index layer — **C, high ceiling, high scope risk**

**Missing.** No text path. QTB Section 11.8 is a stub.

**Why it is interesting.** The natural design is unusual and worth stating precisely: let a language
model read and write the **symbolic index layer**, not the representation layer. The Tensor Brain
emits a triple sequence and an LM verbalizes it; or an LM emits triples that enter as
teacher-forced measurements. That is a clean modular interface between a subsymbolic memory system
and a language system, at the symbol boundary rather than through a learned adapter — which is
precisely the interface most current neurosymbolic work fails to obtain.

PVSG has captions and descriptions, and your semantic inventory already specifies the provenance
discipline needed to use them without leaking future events into a causal protocol.

**Honest caution.** This is where scope explodes. The cheap version — use captions as an additional
supervision signal on index scores, with observation-time provenance — is a contained, useful
experiment. The closed-loop version is a separate thesis. Do not start with the closed loop.

### 3.15 Distributed indices — **D, defer with a sharper reason than "hard"**

QTB Section 12.3.2 raises replacing localized index neurons with population codes, and immediately
worries about local optima and slow convergence.

The standard reason to defer is difficulty. There is a better reason: **localized indices are the
theory.** One column per symbol is what makes the one-shot Hebbian write (3.2) well defined, what
lets indices carry timestamps and eviction state (3.4, 3.6), what makes intervention meaningful
(3.11), and what grounds the symbolic-feedback claim at all. Distributing the code sacrifices all
of those simultaneously in exchange for better parameter scaling.

That trade should only be evaluated once there is a measured capacity curve for the localized
version to beat. Until then, a distributed variant cannot be assessed — there is no baseline number
it would be trying to improve.

---

## 4. Feasibility ranking

| # | Component | Feasibility | Payoff | Blocks / unblocks |
|---|---|---|---|---|
| 3.1 | Growable vocabulary and reserve pool | **A** | enabling | gates 3.2, 3.4, 3.5, 3.6 and three experiments |
| 3.2 | One-shot Hebbian write, two-timescale `A` | **A** | very high | the whole memory program |
| 3.3 | Decoder `g⁺` to feature space | **A** | very high | embodiment, autoencoding, imagination, surprise |
| 3.11 | Intervention harness | **A** | high, strategic | makes every other result interpretable |
| 3.12 | Sparsity on `A` | **A** | moderate | pairs with capacity experiment |
| 3.10 | Outcome-conditioned candidates | **A/B** | high | order effects, blindness, and real compute savings |
| 3.4 | Time-aware indices, recency prior | **B** | high | recent vs remote distinction |
| 3.5 | Consolidation by activation replay | **B** | high | continual learning story |
| 3.13 | Multi-timescale evolution | **B** | moderate | video-specific architecture question |
| 3.6 | Forgetting and eviction | **B** | moderate | after capacity curve exists |
| 3.7 | Action indices and actuator | **C** | highest ceiling | requires an environment |
| 3.8 | Reward and valence | **C** | high in an environment | do not annotate PVSG for this |
| 3.9 | Learned write / boundary policy | **C** | high | capstone; needs 3.1, 3.2, 3.3 |
| 3.14 | Language boundary at the index layer | **C** | high, risky | scope discipline required |
| 3.15 | Distributed indices | **D** | unknown | needs a localized baseline first |

**The four A-tier components (3.1, 3.2, 3.3, 3.11) are together perhaps two weeks of work, need no
new data, and convert roughly half of the framework's currently unfalsifiable claims into
measurable ones.** If nothing else in this document is acted on, those four are the recommendation.

---

## 5. The PVSG ceiling, and the case for a small environment

This deserves to be stated plainly because none of the existing documents say it.

**PVSG cannot test agency.** It is passive, third-person, fully annotated video. The agent does not
act, nothing responds to it, and there are no counterfactuals. Therefore the following claims are
*structurally* untestable on PVSG, no matter how good the experiments are:

- remote episodic memory guiding behaviour (Section 9.7);
- memory-supported decision making (Section 9.9, Section 13.4.1);
- near-term and long-term planning (Sections 13.5.2–13.5.3);
- actions as ordinary indices, and the shared recognize/execute column (Section 3.7 above);
- the value of a memory (Section 3.8 above);
- any intervention on the world, as opposed to intervention on the model.

That is a large fraction of QTB Section 13 and of the original paper's Section 9. If those claims
are part of the intended contribution, no amount of PVSG work will reach them.

**The complement is cheap.** A small partially-observable gridworld — objects with stable
identities, a few relations, occluders, an agent that moves and manipulates, and optionally a
reward — is a contained piece of engineering, and it provides exactly what PVSG cannot:

- ground-truth counterfactuals, so "would the model have acted differently without that memory?"
  is answerable;
- controllable occlusion and controllable recurrence intervals, so object permanence and
  recency curves can be measured at chosen difficulty rather than at whatever PVSG happens to
  contain;
- consequences, hence reward, hence surprise, hence a principled novelty and boundary signal;
- arbitrarily long episode streams for capacity, forgetting and consolidation, without the 394-video
  ceiling.

The right framing is **complementary, not competitive**: PVSG supplies realism, visual difficulty
and genuine individuals; the environment supplies agency, counterfactuals and unlimited controlled
streams. A result that holds in both is far stronger than either alone, and the pairing is itself a
defensible methodological contribution.

**Sequencing caution.** Do not build the environment first. Build the A-tier components against
PVSG, where the perception problem is real and the data already exists, and add the environment
when the memory mechanisms work and the action claims become the binding constraint. An environment
built too early tends to become the project.

---

## 6. What I would build, in order

**Weeks 1–2 — the A tier, all four.** Growable vocabulary and reserve pool; one-shot Hebbian write
with a two-timescale `A`; the feature decoder; the intervention harness. No new data. At the end
of this, episodic memory, embodiment, imagination and causal analysis are all expressible.

**Weeks 3–5 — make them earn their place.** One-shot versus gradient-trained identity enrollment
under the `blocked` protocol. Decoder-based grounding of index embeddings. Intervention on measured
identity outcomes to test whether index feedback is load-bearing. Capacity sweep with and without
sparsity. These are cheap, and any of them could produce a result worth reporting on its own.

**Weeks 6–10 — the B tier where it pays.** Time-aware retrieval and the recency/similarity sweep;
consolidation by activation replay against the video stream; outcome-conditioned candidate sets
feeding the order-effect experiment.

**Then decide.** At that point you will know whether the memory mechanisms actually work. If they
do, the environment (3.7, 3.8) becomes the highest-value next investment and the planning and
decision claims come into reach. If they do not, that is a substantial negative result about a
published framework, and it is better to have it from cheap experiments than after building an
environment.

---

## 7. What to deliberately not build yet

- **Distributed indices.** No baseline exists to justify the trade. See 3.15.
- **Pixel-level decoding.** The feature decoder answers the scientific questions; pixels are a
  demo and cost far more.
- **A generic protocol runner or schedule composer.** Your own repository policy already forbids
  this, and it is worth restating here, because every component above invites one. The explicit
  schedules are load-bearing research documentation.
- **Reward annotation on PVSG.** Ill-defined for a passive observer. Reward belongs to the
  environment.
- **The closed-loop language interface.** Start with captions as supervision; the loop is a
  separate project.
- **Long-term planning.** Section 13.5.3 is not yet a specification, and near-term planning via
  evolution rollout has to work first.

---

## 8. The one-sentence version

The Tensor Brain currently has a read path, a slow learn path, and annotations; adding a fast write
path, a downward decode path, and — eventually — consequences would turn roughly half of its
published claims from narrative into measurement, and the first two of those cost about two weeks
and no new data.
