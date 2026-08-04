| condition | escaped seeds | first choice, train cues | first choice, held-out cues | return | distractor/ep | steps |
|---|---|---|---|---|---|---|
| `tb-full` | 5 / 6 | 0.793 ± 0.097 | 0.545 ± 0.054 | +0.697 | 0.20 | 8.9 |
| `gru-control` | 1 / 6 | 0.999 ± 0.000 | 0.989 ± 0.000 | +0.896 | 0.00 | 5.2 |
| `lstm-control` | 0 / 6 | — | — | — | — | never escaped |
| `decoupled-feedback` | 6 / 6 | 0.906 ± 0.055 | 0.683 ± 0.048 | +0.790 | 0.09 | 8.1 |
| `no-action-feedback` | 6 / 6 | 0.777 ± 0.115 | 0.545 ± 0.054 | +0.712 | 0.22 | 8.6 |
| `pvm-action` | 6 / 6 | 0.967 ± 0.004 | 0.719 ± 0.026 | +0.811 | 0.03 | 7.9 |
| `no-percept-measure` | 6 / 6 | 0.893 ± 0.085 | 0.722 ± 0.061 | +0.786 | 0.11 | 7.9 |
| `no-percept-feedback` | 5 / 6 | 0.848 ± 0.072 | 0.646 ± 0.063 | +0.745 | 0.15 | 8.6 |
| `no-cue` | 5 / 6 | 0.324 ± 0.005 | 0.330 ± 0.004 | +0.480 | 0.68 | 9.7 |
| `cue-initial` | 6 / 6 | 0.337 ± 0.004 | 0.329 ± 0.006 | +0.517 | 0.66 | 9.2 |
| `no-evolution` | 0 / 6 | — | — | — | — | never escaped |
| `evolution-qtb` | 6 / 6 | 0.721 ± 0.085 | 0.484 ± 0.045 | +0.734 | 0.28 | 7.6 |
| `evolution-relu` | 2 / 6 | 0.341 ± 0.008 | 0.326 ± 0.008 | +0.572 | 0.66 | 8.0 |
| `score-softplus-bias` | 0 / 6 | — | — | — | — | never escaped |
| `score-centered` | 6 / 6 | 0.461 ± 0.050 | 0.370 ± 0.025 | +0.527 | 0.52 | 9.8 |
| `linear-critic` | 6 / 6 | 0.672 ± 0.066 | 0.513 ± 0.043 | +0.626 | 0.32 | 9.7 |
| `no-critic` | 4 / 6 | 0.511 ± 0.064 | 0.406 ± 0.025 | +0.602 | 0.49 | 9.1 |
| `deliberate-2-attend` | 6 / 6 | 0.968 ± 0.008 | 0.755 ± 0.060 | +0.740 | 0.03 | 8.9 |
| `deliberate-3-attend` | 6 / 6 | 0.992 ± 0.004 | 0.803 ± 0.050 | +0.822 | 0.01 | 8.1 |
| `deliberate-2-measure` | 6 / 6 | 0.966 ± 0.003 | 0.728 ± 0.038 | +0.812 | 0.03 | 8.1 |
| `argmax-action` | 0 / 6 | — | — | — | — | never escaped |
| `grounded-percepts` | 1 / 6 | 0.456 ± 0.000 | 0.080 ± 0.000 | -0.362 | 0.23 | 22.5 |
| `percepts-in-policy-gradient` | 0 / 6 | — | — | — | — | never escaped |
