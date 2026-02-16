##### MINING FLOW #####

# Step no. 1:
* python data_exploration/check_data_bias.py (results/full_run/plots/data_bias_report)

# Step no. 2:
* Mining: python run_association_mining.py

# Step no. 3: Result Exploration
* python result_exploration/pipeline_efficiency.py [--update-readme]
* python result_exploration/show_top_rules.py [--update-readme]

# Step no. 4: Result Visualization
* python visualization\visualize_mining_results.py (results/full_run/plots/mining_report)
* python visualization\visualize_mining_results.py (results/full_run/plots/raw_rules_report)
* python visualization\visualize_top_rules_spatial.py [--top_n 20] [--exclude_shared_items / --exclude_self_loops]

# Step no. 5: Consensus Report:
**(Run each one with and without --no_self flag)**
* *step 1* python result_exploration/generate_consensus_tables.py [--top_n 100] [--no_self] (tables under: results/full_run/data/consensus_tables)
* *step 2* python visualization\visualize_consensus.py [--no_self] (results/full_run/plots/consensus_report)

##### RULE CORRELATION FLOW #####

# Step no. 1: Run discovery:
* *Simple Discovery*: python check_rule_correlation_with_disease/run_robust_simple_stats.py [--no_self]
* *Advanced Discovery*: python check_rule_correlation_with_disease/advanced_discovery.py [--no_self]

# Step no. 2: Comparison
* python visualization/clinical_correlations/plot_performance_comparisons.py (results/full_run/plots/clinical_correlation_report)
* python visualization/clinical_correlations/plot_rule_target_value_heatmap.py [--num_strategies 5] [--no_self] [--method KNN_R / CN / BAG] (results/full_run/plots/clinical_correlation_report)




