### E1 - first-choice accuracy by observation noise (escaped seeds)

| condition | eps=0 | eps=0.3 | eps=0.6 |
|---|---|---|---|
| `tb-full` | 0.797 ± 0.125 (3/4) | 0.360 ± 0.019 (2/4) | 0.329 ± 0.012 (4/4) |
| `gru-control` | nan ± nan (0/4) | 0.832 ± 0.000 (1/4) | nan ± nan (0/4) |
| `lstm-control` | nan ± nan (0/4) | nan ± nan (0/4) | nan ± nan (0/4) |
| `decoupled-feedback` | 0.900 ± 0.080 (4/4) | 0.422 ± 0.080 (4/4) | 0.353 ± 0.004 (4/4) |

### E2 - first-choice accuracy by prior weight and volatility

| hazard | `alpha-1.0` | `alpha-0.5` | `alpha-0.0` | `alpha-learned` |
|---|---|---|---|---|
| 0 | 0.346 ± 0.007 (4/4) | 0.493 ± 0.072 (4/4) | 0.688 ± 0.040 (4/4) | 0.370 ± 0.030 (4/4) |
| 0.05 | 0.403 ± 0.018 (4/4) | 0.431 ± 0.023 (3/4) | 0.771 ± 0.017 (4/4) | 0.381 ± 0.024 (4/4) |
| 0.2 | 0.406 ± 0.016 (4/4) | 0.550 ± 0.019 (4/4) | 0.731 ± 0.039 (4/4) | 0.426 ± 0.013 (4/4) |
