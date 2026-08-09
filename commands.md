##### MINING FLOW #####

# --- Configuration ---
# Everything lives in constants.py: SETTINGS (how to mine) and the plain constants
# below it (N_SHUFFLES, RANDOM_SEED, LABELS_KEPT_FIXED, MAX_FDR, MIN_LIFT_GAIN,
# KEEP_TOP_RULES, WORKERS). DEBUG=True there switches to debug_run on a few FOVs.
#
# Four of these can be overridden per run without editing the file:
#   WEIGHTING=weighted|binary      how much a neighbour counts (distance decay, or presence)
#   METHOD=CN|KNN_R                how cells are grouped into patches
#   MAX_ITEMS_PER_RULE=2           longest rule to build
#   LABELS_KEPT_FIXED=Epithelial,Muscle    labels that never move when shuffling
#                                          ("Epithelial*" also matches Epithelial_anything)
#
# Linux/Mac:  WEIGHTING=binary METHOD=KNN_R python run_association_mining.py
# Windows:    set WEIGHTING=binary && set METHOD=KNN_R && python run_association_mining.py
#
# Each run replaces its own results folder, which is named after the weighting and
# method, so a weighted CN run and a binary KNN_R run do not overwrite each other.
# Progress goes to the console — redirect to keep it:
#   python run_association_mining.py > run.log 2>&1

# --- Output ---
# results/<run_type>/<weighting>_<method>_<n>_items/
#   run_config.json                      every setting the run used
#   data/results_<METHOD>.csv            final rules, joined to the biopsy metadata
#   data/results_<METHOD>_RAW.csv        the same before the MAX_FDR cut (SAVE_RAW_RULES=True)

# Step no. 1: Data check
* python data_exploration/check_data_bias.py

# Step no. 2: Mining
* python run_association_mining.py

# Step no. 3: Analysis
* The notebooks in visualization/notebooks/ and result_exploration/. Each one is self-contained.
* python result_exploration/analyze_ratios.py   (needs step 1 to have been run first)

##### USING THE LIBRARY DIRECTLY #####

# fpgrowth_rule_mining/ has no dependency on this repo — it takes coordinates and
# labels and returns DataFrames. See fpgrowth_rule_mining/README.md for the API and
# the full parameter table.
