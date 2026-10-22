# Commands

## Setup

```
pip install -e .
pip install -e ../spatial-association-rules   # only while the library changes here
```

The first line takes the mining from PyPI; the second points it at the working copy instead.

## Mining flow

### Configuration

Everything lives in `constants.py`: `SETTINGS` (how to mine) and the plain constants below
it (`N_SHUFFLES`, `N_CONDITIONAL_SHUFFLES`, `SAVE_PERMUTATION_METRICS`, `RANDOM_SEED`, `LABELS_KEPT_FIXED`,
`MAX_INDIVIDUAL_FDR`, `MIN_LIFT_GAIN`, `MIN_CONSEQUENT_CONVICTION_GAIN`, `WORKERS`).
`DEBUG=True` there switches to `debug_run` on a few FOVs.

Conditional tests use `N_CONDITIONAL_SHUFFLES` shuffles per distinct fixed-type batch,
so they can take substantially longer than the individual tests. Set it to `None` to
disable them. Both FDR calculations use the library defaults.

These settings can be overridden per run without editing the file:

| override | what it sets |
| --- | --- |
| `WEIGHTING=weighted\|binary` | how much a neighbour counts (distance decay, or presence) |
| `METHOD=CN\|KNN_R` | how cells are grouped into patches |
| `MAX_ITEMS_PER_RULE=2` | longest rule to build |
| `LABELS_KEPT_FIXED=Epithelial,Muscle` | labels that never move when shuffling (`Epithelial*` also matches `Epithelial_anything`) |
| `SAVE_PERMUTATION_METRICS=true` | save shuffle metrics; `1` / `yes` also enable it, otherwise it is off |

```
# Linux/Mac
WEIGHTING=binary METHOD=KNN_R python run_association_mining.py

# Windows
set WEIGHTING=binary && set METHOD=KNN_R && python run_association_mining.py
```

Each run replaces its own results folder, which is named after the weighting and method,
so a weighted CN run and a binary KNN_R run do not overwrite each other. Progress goes to
the console — redirect to keep it:

```
python run_association_mining.py > run.log 2>&1
```

### Output

```
results/<run_type>/<weighting>_<method>_<n>_items/
  run_config.json                         every setting the run used
  data/results_<METHOD>.csv               rules with p-values and FDR values
  data/results_<METHOD>_RAW.csv           rules before final mining filters, without p-values
  data/results_<METHOD>_comparisons.csv   individual conditional comparisons
  permutations/sample_<FOV>/*.npz         optional shuffle metrics
```

All three CSVs retain every returned column and include FOV and biopsy metadata;
empty tables are saved with headers. Existing column names such as `FOV`, `Lift`, and
`Individual_FDR` stay the same; additional columns keep their library names.
Join comparisons to their complex rules on `["FOV", "rule_idx"]`; `simpler_rule` names
the shorter cell-type rule each one was compared with.
Raw rules still reflect earlier search pruning; they are not every possible rule.
The rules CSV is not cut by FDR; selection happens in analysis.

Set the environment variable `SAVE_PERMUTATION_METRICS=true` to record every shuffle's
lift and conviction. Each FOV gets `regular_p_value.npz` and one `conditional_0001.npz`, etc.,
per conditional batch actually shuffled. FOV names are URL-encoded in folder names.
Recording defaults to off and does not change the p-values.
Arrays are buffered on disk per batch, then compressed; recording needs extra disk space.

Each archive contains `rule_idx`, `lift`, `conviction`, and metadata `sample_id`,
`test`, `fixed_types`. Individual tests also contain `passed_mining_filters`.
Metric arrays have shape `(permutations, rules)`; their columns follow `rule_idx`,
and the row number is the permutation number. Undefined metrics stay `NaN`;
infinite conviction is preserved. Simpler-rule links come from the comparisons CSV,
and gains can be calculated from the metrics in the same conditional batch.

```python
import numpy as np

with np.load("results/<run>/permutations/sample_<FOV>/individual.npz") as data:
    column = np.flatnonzero(data["rule_idx"] == rule_idx)[0]
    shuffled_lifts = data["lift"][:, column]
```

### Steps

1. **Data check** — `python data_exploration/check_data_bias.py`
2. **Mining** — `python run_association_mining.py`
3. **Analysis** — the notebooks in `visualization/notebooks/` and `result_exploration/`,
   each one self-contained, and `python result_exploration/analyze_ratios.py`
   (needs step 1 to have been run first)

## Using the library directly

The mining is the spatial-association-rules package, installed from PyPI: it takes
coordinates and labels and returns DataFrames, and knows nothing about this repo.
`pip install spatial-association-rules`. Its own README has the API and the full
parameter table.
