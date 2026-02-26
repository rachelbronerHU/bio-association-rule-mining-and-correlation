# Debugging Configurations
DEBUG = False # Set to True for quick test
DEBUG_FOVS_PER_GROUP = 10
SAVE_RAW_RULES = True

# Path Configuration
DATA_DIR = 'data/'
MIBI_GUT_DIR_PATH = DATA_DIR + 'MIBIGutCsv/'

RESULTS_BASE_DIR = 'results/'
if DEBUG:
    RESULTS_DIR = RESULTS_BASE_DIR + 'debug_run/'
else:
    RESULTS_DIR = RESULTS_BASE_DIR + 'full_run/'

RESULTS_DATA_DIR = RESULTS_DIR + 'data/'
RESULTS_PLOTS_DIR = RESULTS_DIR + 'plots/'

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
