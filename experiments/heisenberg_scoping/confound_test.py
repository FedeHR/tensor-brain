"""Separate two confounded causes in the scene A/B result.

The documented claim is that the TB-vs-exact-Bayes ranking flips because of the
MEASUREMENT PROCESS (unconditional vs saliency-gated). But scene_dataset draws
scenes from a situation mixture while the model assumes an independent Bernoulli
prior, so the PRIOR is misspecified too. Disentangle by rebuilding the corpus
from the model's own prior.
"""

import statistics

import torch

from experiments.bayes_approximation import scene as S

DTYPE = torch.float64

ont = S.build_ontology(num_categories=12)
gen = torch.Generator().manual_seed(0)
mdl = S.build_scene_model(ont, generator=gen)

RULES = ["TB", "TB-affine", "Exact Bayes (unconditional model)"]


def prior_corpus(num_scenes, num_named, gated, generator):
    """Same as scene_dataset but scenes come from the MODEL's factorized prior."""
    gamma = torch.sigmoid(mdl.q_prior)
    log_z_max = float(mdl.log_partition(mdl.states()).max())
    corpus = []
    guard = 0
    while len(corpus) < num_scenes and guard < 500 * num_scenes:
        guard += 1
        scene = (torch.rand(ont.num_categories, generator=generator, dtype=DTYPE)
                 < gamma).to(DTYPE)
        if gated:
            log_accept = num_named * (float(mdl.log_partition(scene)) - log_z_max)
            if float(torch.rand(1, generator=generator, dtype=DTYPE).log()) > log_accept:
                continue
        named = S.annotate(mdl, scene, num_named, saliency_gated=gated, generator=generator)
        corpus.append((scene, named, "prior"))
    return corpus


for source, builder in [
    ("situation mixture (as shipped)",
     lambda ns, nn, g, gt: S.scene_dataset(mdl, ont, num_scenes=ns, num_named=nn,
                                           saliency_gated=gt, generator=g)),
    ("model's own factorized prior",
     lambda ns, nn, g, gt: prior_corpus(ns, nn, gt, g)),
]:
    for gated in (False, True):
        rows = {r: [] for r in RULES}
        for seed in range(10):
            g = torch.Generator().manual_seed(900 + seed)
            corpus = builder(300, 4, g, gated)
            summary = S.evaluate_corpus(mdl, corpus)
            for r in RULES:
                rows[r].append(summary[r]["nll"])
        diff = [a - b for a, b in zip(rows["TB"], rows["Exact Bayes (unconditional model)"])]
        wins = sum(1 for d in diff if d < 0)
        tag = "gated" if gated else "uncond"
        print(f"\n--- scenes from {source} | {tag} ---")
        for r in RULES:
            print(f"    {r:<36} {statistics.mean(rows[r]):.3f} ± {statistics.stdev(rows[r]):.3f}")
        print(f"    paired TB - ExactBayes: {statistics.mean(diff):+.3f}"
              f" ± {statistics.stdev(diff):.3f}   TB better in {wins}/10")
