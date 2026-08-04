### gotolocal (`BabyAI-GoToLocal-v0`)

| condition | seeds | success | held-out missions | return | steps |
|---|---|---|---|---|---|
| `tb-full` | 3 | 0.793 ± 0.007 | 0.824 ± 0.027 | 0.574 | 28.9 |
| `gru-control` | 3 | 0.885 ± 0.003 | 0.927 ± 0.009 | 0.690 | 21.2 |
| `lstm-control` | 3 | 0.863 ± 0.011 | 0.896 ± 0.011 | 0.667 | 22.7 |
| `deliberate-3-attend` | 3 | 0.790 ± 0.009 | 0.850 ± 0.011 | 0.569 | 29.1 |
| `no-cue` | 3 | 0.772 ± 0.004 | 0.807 ± 0.015 | 0.561 | 29.6 |
| `no-percept-measure` | 3 | 0.780 ± 0.020 | 0.828 ± 0.041 | 0.578 | 28.4 |
| `decoupled-feedback` | 3 | 0.780 ± 0.012 | 0.823 ± 0.038 | 0.579 | 28.4 |
| `pvm-action` | 3 | 0.768 ± 0.037 | 0.809 ± 0.044 | 0.553 | 30.2 |
| `score-softplus-bias` | 3 | 0.490 ± 0.020 | 0.531 ± 0.032 | 0.322 | 44.6 |
| `cue-gain-8` | 3 | 0.776 ± 0.020 | 0.839 ± 0.019 | 0.558 | 29.8 |
| `cue-gate-learned` | 3 | 0.784 ± 0.006 | 0.812 ± 0.032 | 0.593 | 27.4 |
| `normalized-drive` | 3 | 0.783 ± 0.017 | 0.844 ± 0.015 | 0.576 | 28.6 |

### doorkey (`MiniGrid-DoorKey-6x6-v0`)

| condition | seeds | success | return | steps |
|---|---|---|---|---|
| `tb-full` | 3 | 1.000 ± 0.000 | 0.966 | 13.8 |
| `gru-control` | 3 | 1.000 ± 0.000 | 0.965 | 14.0 |
| `lstm-control` | 3 | 1.000 ± 0.000 | 0.966 | 13.7 |
| `deliberate-3-attend` | 3 | 1.000 ± 0.000 | 0.967 | 13.1 |
| `no-percept-measure` | 3 | 1.000 ± 0.000 | 0.965 | 14.0 |
| `decoupled-feedback` | 3 | 1.000 ± 0.000 | 0.966 | 13.6 |
| `pvm-action` | 3 | 1.000 ± 0.000 | 0.966 | 13.5 |
| `score-softplus-bias` | 3 | 0.204 ± 0.058 | 0.099 | 328.6 |
