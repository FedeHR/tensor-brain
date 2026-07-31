# PVSG semantic property and relation inventory

## Boundary

`experiments/pvsg/semantic_inventory.json` defines candidate indices, not property assignments.
Each property value and semantic-relation predicate can receive one ordinary column of `A`.
Facts linking a PVSG category or identity to these indices require a separate versioned artifact
with evidence and split provenance; this inventory never turns an omitted label into a negative.

Taxonomic labels remain in the [object hierarchy](pvsg_object_hierarchy.md). This follows the
[CSLB concept property norms](https://link.springer.com/article/10.3758/s13428-013-0420-4),
which record semantic features separately from tagged taxonomic features, and the earlier
[McRae feature norms](https://pubmed.ncbi.nlm.nih.gov/16629288/). It also preserves the Tensor
Brain distinction between unary attributes and relational triples.

## Unary properties

The perceptual vocabulary is a compact normalization of attribute types in
[Visual Attributes in the Wild](https://openaccess.thecvf.com/content/CVPR2021/html/Pham_Learning_To_Predict_Visual_Attributes_in_the_Wild_CVPR_2021_paper.html).
VAW's explicit positive, explicit negative, and unlabeled distinction motivates our unknown
policy. The affordances follow the normed object dimensions in
[THINGSplus](https://pmc.ncbi.nlm.nih.gov/articles/PMC10991023/).

| Family | Values | Stability | Intended question |
|---|---:|---|---|
| color | 11 | identity-stable | Can appearance be decoded or recalled for a known entity? |
| material | 12 | identity-stable | Does visual evidence support stable composition knowledge? |
| shape | 10 | identity-stable | Which geometric properties transfer across categories? |
| closure state | 2 | observation-transient | Can episodic memory retain open versus closed state? |
| content state | 2 | observation-transient | Can it retain empty versus full state? |
| surface condition | 5 | observation-transient | Can it retain wet, dry, clean, dirty, or damaged state? |
| power state | 2 | observation-transient | Can it retain powered-on versus powered-off state? |
| affordance | 4 | category-typical | Can semantic memory supply motion and manipulation knowledge? |
| risk | 1 | category-typical | Can semantic memory enrich perception with possible physical harm? |

Color, material, shape, and transient state require explicit instance annotations before they
become targets. Multiple colors, materials, shapes, or surface conditions may be true at once.
Single-label state families are only supervised when one listed value is explicitly known.

The four affordance labels are `can_move_independently`, `movable_by_person`,
`graspable_by_hand`, and `holdable_by_hand`. We do not include THINGSplus dimensions already
represented by the hierarchy, or subjective dimensions such as preciousness, pleasantness, and
arousal in the first inventory.

Risk is a positive, nonvisual semantic fact named `can_cause_physical_harm`. It deliberately
does not reproduce the original paper's `Dangerous` for every living thing and `Harmless` for
every nonliving thing. Absence of the risk fact means unknown, not harmless. Contextual hazards
such as boiling water can later be represented compositionally using state and risk facts.

## Semantic relations

Concept-level relations use the established CSLB/ConceptNet forms `made_of`, `used_for`, and
`capable_of`; see the [ConceptNet 5 paper](https://arxiv.org/abs/1612.03975). Identity-level
relations use a compact subset of Tensor Brain examples and Wikidata property conventions:

| Canonical fact | Inverse | Stability |
|---|---|---|
| `made_of` | none | category-typical |
| `used_for` | none | category-typical |
| `capable_of` | none | category-typical |
| `owned_by` | `owns` | time-varying |
| `parent_of` | `child_of` | identity-stable |
| `partner_of` | itself | time-varying and symmetric |
| `sibling_of` | itself | identity-stable and symmetric |

Ownership and kinship are not unary attributes. They connect two identities and therefore use
the same triple structure as other semantic statements. `parent_of` is gender-neutral; more
distant kinship should be derived rather than separately enumerated. `loved_by` is omitted
because it is subjective and cannot be established reliably from PVSG video or captions.

No identity relation is currently asserted. A future fact record must include subject,
predicate, object, evidence source, confidence, validity interval where appropriate, and the
earliest observation time at which the fact is available. Caption- or VLM-derived facts must be
named as a separate modality and may not leak future video content into a causal protocol.

## Ground-truth construction

The inventory is a selectable superset. A run includes only property families for which the
chosen split has audited facts; defining 49 values does not require activating all 49 in every
checkpoint. Candidate restriction also keeps each measurement local to one family.

The initial Section 6 experiments therefore proceed without these semantic labels. A later,
versioned annotation pass can obtain category-typical and caption-supported facts from a simple
LM followed by human verification, while identity-stable visual properties can use a VLM over a
temporally spread mosaic of masked crops, again with abstention and human review. Transient state
requires interval evidence rather than a single identity mosaic. These generated labels remain
separate from the source annotations and enter an experiment only after their provenance and
review status have been recorded.

Ground truth is constructed at the coarsest defensible granularity:

1. Category-typical affordance and risk facts are proposed once for the reviewed fine classes,
   then human-reviewed. No frame-level VLM call is needed.
2. Identity-stable color, material, and shape use a temporally spread mosaic of high-quality
   masked crops for one tracked identity. A VLM may propose only values from the closed
   inventory and must be allowed to abstain. Agreement across views and prompts is evidence;
   disagreement remains unknown or enters human review. Accepted values propagate only within
   that identity, never to every member of its category.
3. Observation-transient state is annotated only for applicable objects and candidate change
   intervals, using before/after frames or short clips. Predicate spans and visual change may
   nominate intervals, but do not themselves prove a state. No state is copied across an
   unobserved interval.
4. Ownership and kinship candidates come first from explicit caption phrases whose numbered
   mentions resolve to tracked PVSG identities. Deterministic high-precision patterns precede a
   text model. Unmentioned siblings or owners are not invented; synthetic hidden entities, if
   later useful, form a separately named controlled experiment.

Each accepted unary fact records subject granularity (`category`, `identity`, or frame interval),
property value, explicit polarity, evidence addresses, annotator and revision, prompt version,
confidence, review status, and earliest availability time. Relation records additionally store
subject, predicate, object, inverse derivation, and validity interval. Full-video descriptions
may help create offline evaluation truth, but facts exposed to a causal model are restricted to
the observation prefix.

Before scaling, a small stratified pilot estimates abstention, human-verified precision, runtime,
and cost per identity and per state transition. Weak VLM labels may support training only after
this audit; headline evaluation uses a human-verified subset so that it does not merely measure
agreement with the annotating model.

## Current index space

The reviewed inventory contains 49 unary property values and 9 relation predicates. The loader
validates namespaces, global uniqueness, family cardinality, stability metadata, provenance
conventions, and inverse/symmetry consistency. Unknown and catch-all property targets are
forbidden.
