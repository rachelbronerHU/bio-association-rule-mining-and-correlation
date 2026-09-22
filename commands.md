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
it (`N_SHUFFLES`, `RANDOM_SEED`, `LABELS_KEPT_FIXED`, `MAX_FDR`, `MIN_LIFT_GAIN`,
`KEEP_TOP_RULES`, `WORKERS`). `DEBUG=True` there switches to `debug_run` on a few FOVs.

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
  run_config.json                      every setting the run used
  data/results_<METHOD>.csv            final rules, joined to the biopsy metadata
  data/results_<METHOD>_RAW.csv        the same before the MAX_FDR cut (SAVE_RAW_RULES=True)
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
