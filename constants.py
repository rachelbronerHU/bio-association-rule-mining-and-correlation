import os as _os

# Debugging Configurations
DEBUG = False # Set to True for quick test
DEBUG_FOVS_PER_GROUP = 10
SAVE_RAW_RULES = True

# Algorithm Selection: "fpgrowth" | "weighted_fpgrowth"
# Change _DEFAULT_ALGO to switch default. Override per-run via: ALGO=weighted_fpgrowth python ...
_DEFAULT_ALGO = "weighted_fpgrowth"
ALGO = _os.environ.get("ALGO", _DEFAULT_ALGO)

# Path Configuration
DATA_DIR = 'data/'
MIBI_GUT_DIR_PATH = DATA_DIR + 'MIBIGutCsv/'

RESULTS_BASE_DIR = 'results/'
if DEBUG:
    RESULTS_DIR = RESULTS_BASE_DIR + 'debug_run/'
else:
    RESULTS_DIR = RESULTS_BASE_DIR + 'full_run/'

RESULTS_ALGO_DIR = RESULTS_DIR + ALGO + '/'
RESULTS_DATA_DIR = RESULTS_ALGO_DIR + 'data/'
RESULTS_PLOTS_DIR = RESULTS_ALGO_DIR + 'plots/'

TRANSACTION_DATA_DIR = RESULTS_DATA_DIR + 'transaction_data/'
RARE_FILTERING_STATS_DIR = RESULTS_DATA_DIR + 'rare_filtering_stats/'
RESULT_EXPLORATION_DIR = 'result_exploration/'
CONSENSUS_RESULTS_EXPLORATION_DIR = RESULTS_DATA_DIR + '/consensus_tables/'

RESULTS_ML_DIR = RESULTS_DATA_DIR + 'ml_refined_robust_benchmarks/'
RESULTS_ML_DATA_DIR = RESULTS_ML_DIR + 'data/'
RESULTS_ML_DATA_DIR_NO_SELF = RESULTS_ML_DIR + 'data_no_self/' # New No-Self Directory
RESULTS_ML_PLOTS_DIR = RESULTS_ML_DIR + 'plots/'

RESULTS_SIMPLE_STATS_DIR = RESULTS_DATA_DIR + 'simple_stats_benchmarks/'
RESULTS_SIMPLE_STATS_DATA_DIR = RESULTS_SIMPLE_STATS_DIR + 'data/'
RESULTS_SIMPLE_STATS_DATA_DIR_NO_SELF = RESULTS_SIMPLE_STATS_DIR + 'data_no_self/' # New No-Self Directory
RESULTS_SIMPLE_STATS_PLOTS_DIR = RESULTS_SIMPLE_STATS_DIR + 'plots/'

RESULTS_CLINICAL_CORRELATION_PLOTS_DIR = RESULTS_PLOTS_DIR + 'clinical_correlation_report/'

# Param Configuration
MIN_P_VALUE = 0.05

# --- Algorithm-specific Configuration ---
# Switch ALGO above to change everything below automatically.

if ALGO == "weighted_fpgrowth":
    METHODS = ["CN", "KNN_R"]  # BAG has no center cell — not supported by weighted_fpgrowth
    CONFIG = {
        "RADIUS": 25.0,
        "K_NEIGHBORS": 30,
        "BANDWIDTH": 15.0,          # Gaussian decay (µm). At d=BANDWIDTH weight ≈ 0.6. Defaults to RADIUS if absent.
        "MIN_SUPPORT": 0.01,       # Lower than binary: weighted support uses min(item weights), harder to achieve
        "MIN_ABS_SUPPORT": 10,      # Rule must hold in at least this many transactions (absolute floor)
        "MIN_CONFIDENCE": 0.7,      # Same as binary — weighted confidence already requires intensity match, not just presence
        "MIN_LIFT": 1.2,            # Slightly stricter to compensate for finer-grained support scale
        "MIN_LEVERAGE": 0.005,
        "MAX_NEGATIVE_LEVERAGE": -0.001,
        "MIN_CONVICTION": 1.3,
        "MIN_REDUNDANCY_LIFT_IMPROVEMENT": 1.1,
        "MAX_NEGATIVE_LIFT": 0.8,
        "MAX_RULE_LENGTH": 4,
        "TARGET_CELLS": 30,
        "MIN_CELLS_PER_PATCH": 2,
        "N_PERMUTATIONS": 5 if DEBUG else 1000,
        "N_TOP_RULES": 100 if DEBUG else 2000,
        "MIN_CELL_TYPE_FREQUENCY": 5,
        "MIN_CELL_TYPE_PERCENTAGE": 0.01,
    }
else:
    METHODS = ["BAG", "CN", "KNN_R"]
    CONFIG = {
        "RADIUS": 25.0,
        "K_NEIGHBORS": 30,
        "GRID_WINDOW_SIZE": 30.0,
        "WINDOW_STEP_FRACTION": 0.5,
        "MIN_SUPPORT": 0.05,
        "MIN_CONFIDENCE": 0.7,
        "MIN_LIFT": 1.2,
        "MIN_LEVERAGE": 0.005,
        "MAX_NEGATIVE_LEVERAGE": -0.001,
        "MIN_CONVICTION": 1.3,
        "MIN_REDUNDANCY_LIFT_IMPROVEMENT": 1.1,
        "MAX_NEGATIVE_LIFT": 0.8,
        "MAX_RULE_LENGTH": 4,
        "TARGET_CELLS": 30,
        "MIN_CELLS_PER_PATCH": 2,
        "N_PERMUTATIONS": 5 if DEBUG else 1000,
        "N_TOP_RULES": 100 if DEBUG else 2000,
        "MIN_CELL_TYPE_FREQUENCY": 5,
        "MIN_CELL_TYPE_PERCENTAGE": 0.01,
    }
