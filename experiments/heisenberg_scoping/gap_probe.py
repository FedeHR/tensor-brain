"""Is there a measurable logZ gap when A has realistic ontology structure?

Uses the existing PVSG-ontology scene model (structured A: strong evidence for
own category, weak for domain siblings, negative otherwise) and reports the
diagnostics that drive the whole error law.
"""

import torch

from experiments.bayes_approximation import general as G
from experiments.bayes_approximation import inference as I
from experiments.bayes_approximation import model as M
from experiments.bayes_approximation import scene as S

torch.manual_seed(0)

for num_cat in (10, 12):
    ont = S.build_ontology(num_categories=num_cat)
    gen = torch.Generator().manual_seed(0)
    mdl = S.build_scene_model(ont, generator=gen)

    x = mdl.states()
    w = torch.softmax(mdl.log_prior(x), dim=0)
    lz = mdl.log_partition(x)
    mean = float((w * lz).sum())
    var = float((w * (lz - mean) ** 2).sum())

    c, stats = G.affine_correction(mdl)
    gauged = G.gauge_fix(mdl)
    lzg = gauged.log_partition(x)
    meang = float((w * lzg).sum())
    varg = float((w * (lzg - meang) ** 2).sum())

    print(f"\n=== {num_cat} categories, K={ont.num_objects} objects ===")
    print(f"  Var[logZ]            {var:.4f}   (sd {var**0.5:.3f})")
    print(f"  affine fraction      {stats['fraction_affine']:.3f}")
    print(f"  Var[residual]        {stats['var_residual']:.4f}")
    print(f"  Var[logZ] gauge-fixed{varg:9.4f}")

    # predicted error law vs measured, at a few M
    for num_named in (2, 4, 6):
        kls, kls_aff = [], []
        for trial in range(40):
            g2 = torch.Generator().manual_seed(100 + trial)
            sc, _ = S.sample_scene(ont, generator=g2)
            named = S.annotate(mdl, sc, num_named, saliency_gated=False, generator=g2)
            exact = I.exact(mdl, named)
            tb = I.tb(mdl, named)
            aff = G.tb_affine(mdl, named, c)
            n = mdl.state_dim
            kls.append(float(torch.sum(exact.joint * (exact.joint.clamp_min(1e-300).log()
                                                      - tb.as_joint(n).clamp_min(1e-300).log()))))
            kls_aff.append(float(torch.sum(exact.joint * (exact.joint.clamp_min(1e-300).log()
                                                          - aff.as_joint(n).clamp_min(1e-300).log()))))
        pred = 0.5 * num_named**2 * var
        print(f"  M={num_named}: KL(TB) {sum(kls)/len(kls):.4f}  "
              f"KL(TB-affine) {sum(kls_aff)/len(kls_aff):.4f}  "
              f"[law predicts {pred:.4f}]")
