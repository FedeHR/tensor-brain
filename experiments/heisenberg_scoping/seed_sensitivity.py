"""How stable is the TB-vs-exact-Bayes ordering across corpus seeds?

The documented scene result (150 scenes, 4 named) reports exact Bayes ahead of
TB on the unconditional corpus and behind it on the gated one. Repeat over many
corpus seeds and report the spread, to size the replication the real experiment
needs.
"""

import statistics

import torch

from experiments.bayes_approximation import scene as S

ont = S.build_ontology(num_categories=12)
gen = torch.Generator().manual_seed(0)
mdl = S.build_scene_model(ont, generator=gen)

RULES = ["TB", "TB-affine", "Exact Bayes (unconditional model)"]

for num_scenes in (150, 600):
    for gated in (False, True):
        label = "gated" if gated else "uncond"
        rows = {r: [] for r in RULES}
        for seed in range(12):
            g = torch.Generator().manual_seed(500 + seed)
            corpus = S.scene_dataset(
                mdl, ont, num_scenes=num_scenes, num_named=4,
                saliency_gated=gated, generator=g,
            )
            summary = S.evaluate_corpus(mdl, corpus)
            for r in RULES:
                rows[r].append(summary[r]["nll"])

        print(f"\n=== {label}, {num_scenes} scenes, 12 corpus seeds ===")
        for r in RULES:
            v = rows[r]
            print(f"  {r:<36} {statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}")
        # paired difference: TB minus exact Bayes, per seed
        diff = [a - b for a, b in zip(rows["TB"], rows["Exact Bayes (unconditional model)"])]
        wins = sum(1 for d in diff if d < 0)
        print(f"  paired TB - ExactBayes: {statistics.mean(diff):+.3f} "
              f"± {statistics.stdev(diff):.3f}   TB better in {wins}/12 seeds")
