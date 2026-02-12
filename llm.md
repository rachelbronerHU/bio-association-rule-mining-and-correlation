# Project Structure:
* data/ - where the data (input of flow) is stored
* results/ - where the results of the workflow is stored:
    - full_run for the prod run (all data), debug_run for debug data (fraction of the data. you can find the flag in constants.py)
    - data / plots - data for results stored data for forward investigations, plots - results of visualization scripts on result data
* data_explorations - scripts for exploring the data
* results_explorations - some script for first exploration of the results
* visualization - scripts for creating plots from the result data
* check_rule_correlation_with_disease - sub flow for checking correlation of rules from the base flow (association rule mining) with clinical stage metadata

# Important Files:
* constants.py, where constants and paths are stored
* README.md - our final report of the results

# Instructions:
* Never give code first. always give a clear, not too long and smart design suggestion/s, and only after approval give the code