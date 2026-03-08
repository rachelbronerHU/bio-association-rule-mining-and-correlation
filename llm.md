# Project Structure:
* data/ - where the data (input of flow) is stored
* results/ - where the results of the workflow is stored:
    - full_run for the prod run (all data), debug_run for debug data (fraction of the data. you can find the flag in constants.py)
    - results/<run_type>/<algo>/data/ and /plots/ — algo is "fpgrowth" or "weighted_fpgrowth"
* algos/ - mining algorithm implementations (fpgrowth.py, weighted_fpgrowth.py). each exposes a run() entry point
* utils/ - shared utilities: spatial.py (neighborhoods), rules.py (filtering), stats.py (permutation test), config_validation.py
* data_explorations - scripts for exploring the data
* results_explorations - some script for first exploration of the results
* visualization - scripts for creating plots from the result data
* check_rule_correlation_with_disease - sub flow for checking correlation of rules from the base flow (association rule mining) with clinical stage metadata

# Important Files - read them before you start working:
* commands.md - how to use this repo
* constants.py, where constants and paths are stored
* README.md - our final report of the results
* data_exploration/data_inspection_report.md
* data_exploration/check_data_bias.py

# Instructions:
* Never give code first when we start discussion about on new feature / design / bug fix. Always give a clear, not too long and smart design suggestion/s (no code, only design / plan) and only after approval give the code changes