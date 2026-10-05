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
it (`N_SHUFFLES`, `N_CONDITIONAL_SHUFFLES`, `RANDOM_SEED`, `LABELS_KEPT_FIXED`,
`MAX_INDIVIDUAL_FDR`, `MIN_LIFT_GAIN`, `MIN_CONSEQUENT_CONVICTION_GAIN`, `WORKERS`).
`DEBUG=True` there switches to `debug_run` on a few FOVs.

Conditional tests use `N_CONDITIONAL_SHUFFLES` shuffles per distinct fixed-type batch,
so they can take substantially longer than the individual tests. Set it to `None` to
disable them. Both FDR calculations use the library defaults.

Four of these can be overridden per run without editing the file:

| override | what it sets |
| --- | --- |
| `WEIGHTING=weighted\|binary` | how much a neighbour counts (distance decay, or presence) |
| `METHOD=CN\|KNN_R` | how cells are grouped into patches |
| `MAX_ITEMS_PER_RULE=2` | longest rule to build |
| `LABELS_KEPT_FIXED=Epithelial,Muscle` | labels that never move when shuffling (`Epithelial*` also matches `Epithelial_anything`) |

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
  data/results_<METHOD>.csv               classified rules with p-values and FDR values
  data/results_<METHOD>_RAW.csv           rules before final mining filters, without p-values
  data/results_<METHOD>_comparisons.csv   individual conditional comparisons
```

All three CSVs retain every returned column and include FOV and biopsy metadata;
empty tables are saved with headers. Existing column names such as `FOV`, `Lift`, and
`Individual_FDR` stay the same; additional columns keep their library names.
Join comparisons to classified rules on `["FOV", "rule_idx"]`, and to their simpler
rules on `["FOV", "simpler_idx"]` matched to `["FOV", "rule_idx"]`.
Raw rules still reflect earlier search pruning; they are not every possible rule.
The classified CSV is not cut by FDR or `Adds_Information`; selection happens in analysis.

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
