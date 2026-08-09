import os as _os

from fpgrowth_rule_mining import Method, Settings, Weighting


def _parse_list_env(env_var: str, default: list) -> list:
    raw = _os.environ.get(env_var)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# Debugging Configurations
DEBUG = True # Set to True for quick test
DEBUG_FOVS_PER_GROUP = 10

# How much a neighbour counts, and how cells are grouped. Everything else is identical.
# Override per-run via: WEIGHTING=binary METHOD=KNN_R python run_association_mining.py
WEIGHTING = Weighting(_os.environ.get("WEIGHTING", Weighting.WEIGHTED.value))
METHOD = Method(_os.environ.get("METHOD", Method.CN.value))

MAX_ITEMS_PER_RULE = int(_os.environ.get("MAX_ITEMS_PER_RULE", 4))
WORKERS = None if DEBUG else _os.cpu_count()   # None runs in this process

# Cell types whose labels stay put while shuffling, so the tissue keeps its structure.
# An exact name matches that label only. "Epithelial*" matches anything starting with it.
LABELS_KEPT_FIXED = tuple(_parse_list_env("LABELS_KEPT_FIXED", ["Epithelial"]))

# Path Configuration
DATA_DIR = 'data/'
MIBI_GUT_DIR_PATH = DATA_DIR + 'MIBIGutCsv/'

RESULTS_BASE_DIR = 'results/'
RESULTS_DIR = RESULTS_BASE_DIR + ('debug_run/' if DEBUG else 'full_run/')

# The method is in the name so a CN run and a KNN_R run do not overwrite each other
if LABELS_KEPT_FIXED and len(LABELS_KEPT_FIXED) > 0:
    run_name = f"{WEIGHTING.value}_{METHOD.value}_{MAX_ITEMS_PER_RULE}_items_fixed_{'-'.join(LABELS_KEPT_FIXED)}"
else:
    run_name = f"{WEIGHTING.value}_{METHOD.value}_{MAX_ITEMS_PER_RULE}_items"

RESULTS_ALGO_DIR = RESULTS_DIR + run_name + '/'
RESULTS_DATA_DIR = RESULTS_ALGO_DIR + 'data/'
RESULTS_PLOTS_DIR = RESULTS_ALGO_DIR + 'plots/'


# --- Mining ---
# One set of numbers for both weightings. Only `weighting` changes the maths.
SETTINGS = Settings(
    weighting=WEIGHTING,
    method=METHOD,
    radius=25.0,
    min_support=0.01,
    min_lift=1.2,
    max_items_per_rule=MAX_ITEMS_PER_RULE,

    bandwidth=15.0,             # at this distance a neighbour counts ~0.6. Omit to use the radius
    min_cells_per_patch=2,
    max_one_type_share=0.9,     # a patch this dominated by one type says nothing

    # Counted in weight, so this is exactly 10 patches under BINARY and somewhat
    # more than 10 real patches under WEIGHTED, a far neighbour counting as less.
    min_patches=10,
    strong_confidence=0.9,      # above this confidence a lower support is allowed
    min_support_when_strong=0.005,

    min_confidence=0.5,
    min_leverage=0.0005,
    min_conviction=1.3,

    include_avoidance_rules=True,   # search for cell types that keep apart, as well as together
    avoidance_max_lift=0.8,
    avoidance_max_leverage=-0.0025,
    avoidance_min_expected_meetings=10,  # expect at least this many meetings before "they don't meet" counts

    min_label_count=5,          # ignore rules naming a cell type this rare in the sample
)

# --- Significance ---
# Passed to the calls that use them. Nothing here corrects or cuts: p-values come out
# raw, and you correct at the point you make a claim (see the library README).
N_SHUFFLES = 5 if DEBUG else 1000
RANDOM_SEED = 42                     # each FOV derives its own seed from this
MIN_LIFT_GAIN = 1.1                  # a longer rule must beat its shorter version by this much

