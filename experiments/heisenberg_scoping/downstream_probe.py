"""Does posterior fidelity translate into downstream decision quality?

Scores every update rule on the ontology-grounded scene corpus, under both
annotation processes, on task metrics (NLL / accuracy / ECE) rather than KL.
This is the shape the real-data experiment will take.
"""

import torch

from experiments.bayes_approximation import scene as S

torch.manual_seed(0)

ont = S.build_ontology(num_categories=12)
gen = torch.Generator().manual_seed(0)
mdl = S.build_scene_model(ont, generator=gen)

for gated in (False, True):
    label = "saliency-gated" if gated else "unconditional"
    g = torch.Generator().manual_seed(7)
    corpus = S.scene_dataset(
        mdl, ont, num_scenes=400, num_named=4, saliency_gated=gated, generator=g
    )
    summary = S.evaluate_corpus(mdl, corpus)
    print(f"\n=== corpus: {label} (400 scenes, 4 named objects) ===")
    print(f"{'rule':<38} {'NLL':>8} {'acc':>8} {'ECE':>8}")
    for name in sorted(summary, key=lambda k: summary[k]["nll"]):
        row = summary[name]
        print(f"{name:<38} {row['nll']:>8.3f} {row['accuracy']:>8.3f} {row['ece']:>8.3f}")

# how much does each rule move when the naming order is permuted?
g = torch.Generator().manual_seed(11)
sc, _ = S.sample_scene(ont, generator=g)
named = S.annotate(mdl, sc, 5, saliency_gated=False, generator=g)
print("\n=== max belief shift under re-ordering of the same evidence ===")
for name, value in S.order_invariance(mdl, named).items():
    print(f"  {name:<38} {value:.3e}")
