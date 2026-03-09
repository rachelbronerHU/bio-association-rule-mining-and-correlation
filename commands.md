##### MINING FLOW #####

# --- Algorithm Selection ---
# Option A — edit constants.py:
#   ALGO = "fpgrowth"           → runs on BAG, CN, KNN_R
#   ALGO = "weighted_fpgrowth"  → runs on CN, KNN_R only
# Option B — env variable override (no file edit needed): see Step 2 commands below.
# DEBUG=True/False also toggleable via constants.py (debug_run vs full_run).
# CONFIG and METHODS are set automatically in constants.py based on ALGO.

# --- Output Files (per method) ---
# results/<run_type>/<algo>/data/results_<METHOD>.csv         — final rules with metadata
# results/<run_type>/<algo>/data/results_<METHOD>_RAW.csv     — raw rules before filtering (if SAVE_RAW_RULES=True)
# results/<run_type>/<algo>/data/stats_<METHOD>.json          — transaction statistics
# results/<run_type>/<algo>/data/transaction_data/            — per-FOV transaction distributions
# results/<run_type>/<algo>/data/rare_filtering_stats/        — rare cell filtering logs

# Step no. 1:
* python data_exploration/check_data_bias.py (results/full_run/plots/data_bias_report)

# Step no. 2:
* Mining: python run_association_mining.py
* Mining (Linux/Mac, override algo): ALGO=weighted_fpgrowth python run_association_mining.py
* Mining (Windows, override algo):   set ALGO=weighted_fpgrowth && python run_association_mining.py

# Step no. 3: Result Exploration
* python result_exploration/pipeline_efficiency.py [--update-readme]
* python result_exploration/show_top_rules.py [--update-readme]

# Step no. 4: Result Visualization
* python visualization\visualize_mining_results.py (results/full_run/plots/mining_report)
* python visualization\visualize_raw_rules.py (results/full_run/plots/raw_rules_report)
* python visualization\visualize_top_rules_spatial.py [--top_n 20] [--exclude_shared_items / --exclude_self_loops] [--show_tissue_background] (results/full_run/plots/top_rules_spatial)
* python visualization\visualize_transaction_distributions.py (results/full_run/plots/transaction_distributions)

# Step no. 5: Consensus Report:
**(Run each one with and without --no_self flag. Auto-discovers and processes all organs by default.)**
* *step 1* python result_exploration/generate_consensus_tables.py [--top_n 100] [--no_self] [--organs Colon Duodenum] (tables under: results/full_run/data/consensus_tables, organ-stratified)
  - Omit --top_n to save ALL rules (recommended for complete analysis)
  - Use --top_n N to save only top N rules (faster, but may miss stage-specific rules)
* *step 2* python visualization\visualize_consensus.py [--no_self] [--organs Colon Duodenum] (results/full_run/plots/consensus_report, organ-stratified)

##### RULE CORRELATION FLOW #####

# Step no. 1: Run discovery:
# Each script logs to a file named after the script + algo (e.g. run_correlation_pipeline_fpgrowth.log).
* *Dry Run (preview stratification plan)*: python check_rule_correlation_with_disease/run_correlation_pipeline.py --dry_run
* *Both flows*: python check_rule_correlation_with_disease/run_correlation_pipeline.py [--no_self]
* *Simple only*: python check_rule_correlation_with_disease/run_correlation_pipeline.py --only simple [--no_self]
* *ML only*: python check_rule_correlation_with_disease/run_correlation_pipeline.py --only ml [--no_self]
* *(or run individually — unchanged)*
* *Simple Discovery*: python check_rule_correlation_with_disease/run_robust_simple_stats.py [--no_self]
* *Advanced Discovery*: python check_rule_correlation_with_disease/advanced_discovery.py [--no_self]
* *(parallel algos — run in separate terminals)*
* *Linux/Mac*: ALGO=weighted_fpgrowth python check_rule_correlation_with_disease/run_correlation_pipeline.py [--only simple|ml] [--no_self]
* *Windows*:   set ALGO=weighted_fpgrowth && python check_rule_correlation_with_disease/run_correlation_pipeline.py [--only simple|ml] [--no_self]

# Step no. 2: Comparison
* python visualization/clinical_correlations/plot_performance_comparisons.py (results/full_run/plots/clinical_correlation_report)
* python visualization/clinical_correlations/plot_rule_target_value_heatmap.py [--num_strategies 5] [--no_self] [--method KNN_R / CN / BAG] (results/full_run/plots/clinical_correlation_report)




