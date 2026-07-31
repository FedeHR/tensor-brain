# PVSG object hierarchy

## Purpose and boundary

The hierarchy supplies semantic targets for PVSG object observations. It does not add a model
layer or change Tensor Brain scoring. The reviewed paths become three named groups in the
existing `IndexVocabulary`, and every label receives one ordinary global column of `A`.

Each included observation has four distinct readout labels:

1. `fine`: the most specific defensible interpretation of the source object;
2. `basic`: a strict, recognizable, and usually visual superclass of the fine label;
3. `coarse`: a strict visual-semantic superclass of the basic label;
4. `domain`: the broad semantic world containing the coarse class.

The first three labels form strict “is a kind of” paths. Association, location, material,
part-whole, and typical use are not category edges. `domain` is explicitly a broad partition for
semantic generalization rather than a claim of lexical hypernymy. When a generic source name
denotes a narrower visual sense in PVSG, the fine label records that reviewed sense. When the
tracks do not support one honest path, the source category is left outside semantic supervision.

The complete fine, basic, coarse, and domain label sets are pairwise disjoint, and their sizes
must satisfy `domain < coarse < basic < fine`. This keeps each concept at one fixed level and
prevents an experiment from measuring the same target twice. There is no universal `object`,
`thing`, `stuff`, `other`, or miscellaneous root.

## Source and review policy

`experiments/pvsg/object_hierarchy.json` follows the exact 115 `thing` and 11 `stuff` category
order in the pinned `pvsg.json` with SHA-256
`2cbc23b060386ccf090475f90cb0282a3e96cefb29c69772f9f8fd916995ba08`. The source segmentation
distinction between `thing` and `stuff` is not used as a semantic hierarchy level.

Open English WordNet and the Open Images hierarchy may propose senses and parents, but neither
is accepted automatically. Every shipped path is a human-reviewed semantivisual decision. This
follows the basic-level emphasis and avoidance of abstract padding in
[Free-Grained Hierarchical Visual Recognition](https://openaccess.thecvf.com/content/CVPR2026/html/Park_Free-Grained_Hierarchical_Visual_Recognition_CVPR_2026_paper.html).

Representative paths include:

- `adult` -> `human` -> `living_being` -> `natural_and_living_world`;
- `dog` -> `animal` -> `living_being` -> `natural_and_living_world`;
- `potted_plant` -> `plant_life` -> `living_being` -> `natural_and_living_world`;
- `pillow` -> `soft_bedding` -> `household_textile` ->
  `built_and_domestic_environment`;
- `beverage` -> `liquid_food` -> `food` -> `food_and_kitchen`;
- `camera` -> `portable_electronic_device` -> `electronic_device` ->
  `personal_recreation_and_mobility`;
- `wall` -> `room_surface` -> `architectural_element` ->
  `built_and_domestic_environment`;
- `rock` -> `geological_material` -> `natural_material` ->
  `natural_and_living_world`.

`human`, `animal`, and `plant_life` are the three basic branches under `living_being`.
`animal` is never used as a parent of `human`, while this avoids a narrower class such as
`vascular_plant` above a generic `plant` label.

The domain mapping is stored once per coarse label rather than repeated in every path. Every
domain contains at least two coarse classes, so a held-out coarse class can be evaluated against
other training classes from the same domain:

| Domain | Coarse classes | Fine labels | Retained tracks |
|---|---:|---:|---:|
| `natural_and_living_world` | 3 | 17 | 1,702 |
| `food_and_kitchen` | 2 | 25 | 1,001 |
| `built_and_domestic_environment` | 5 | 29 | 2,659 |
| `tools_containers_and_appliances` | 5 | 23 | 993 |
| `personal_recreation_and_mobility` | 7 | 27 | 788 |

The label support is balanced without forcing unrelated coarse classes together merely to equal
track counts. PVSG's scene distribution remains uneven, so domain experiments must use
class-aware training and report macro and per-domain metrics in addition to accuracy.

## Reviewed source corrections

The source annotations and captions resolve several non-obvious labels:

- `ballon` and `vaccum` are normalized to `balloon` and `vacuum_cleaner`;
- `cover` denotes lids or caps, and `glass` denotes a drinking glass;
- `simmering` masks food or a dish and maps to `simmering_food -> prepared_food -> food`;
- `can` includes paint cans as well as kitchen cans, so its path is
  `can -> cylindrical_container -> container`, not `food_container`;
- `fan` maps through `air_moving_appliance`; material such as metal belongs in an attribute
  experiment rather than in the object taxonomy;
- the source `plant` category denotes potted or cultivated plants and is normalized to
  `potted_plant -> plant_life -> living_being`;
- furniture, household textiles, furniture components, natural materials, and lighting devices
  are separated instead of being joined by associative labels such as `furnishing` or
  `natural_environment`.

The retained source category `bat` contains four senses. All 23 retained tracks are explicitly
resolved by stable identity:

- 9 baseball bats;
- 10 golf clubs;
- 3 tennis rackets;
- 1 badminton racket.

These fine labels share `striking_sports_equipment -> sports_equipment`. The separate PVSG
source category `racket` contains three table-tennis paddles and maps to
`table_tennis_paddle -> striking_sports_equipment -> sports_equipment`.

No hierarchy target is supplied for `board`, `gift`, `others`, `paper`, `powder`, `rack`,
`ring`, or `stand`. `gift` is a contextual role spanning wrapped packages and presented items;
`rack` spans freestanding storage and several kinds of attached component. The remaining labels
are overloaded, unresolved, or train-unseen as documented in the JSON artifact. Their tracks
remain available for identity and relation experiments. They require track-level review rather
than a forced parent.

## Validation

The loader validates exact source order, complete coverage of every identity belonging to a
refined source category, three distinct category labels per path, complete coarse-to-domain
coverage, consistent parents, pairwise-disjoint level vocabularies, strictly decreasing level
sizes, exclusion reasons, and absence of catch-all concepts. It currently yields 121 fine, 78
basic, 22 coarse, and 5 domain labels: 226 disjoint static columns in `A`.

The hierarchy is semantically reviewed first and visually audited second. Frozen DINO features
should test level-wise linear-probe and nearest-centroid accuracy, within-parent versus
between-parent confusion, and same-parent feature similarity using training videos only. These
diagnostics may flag an edge for human revision; they must not silently define the taxonomy.
