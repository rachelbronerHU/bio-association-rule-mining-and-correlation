import sys
import subprocess
import argparse
from pathlib import Path


def run_cmd(command, repo_root):
    print(" ".join(command))
    result = subprocess.run(command, cwd=str(repo_root))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run post-mining exploration and visualization pipeline.")
    parser.add_argument("--subset_rule_items_eq", type=int, default=None)
    parser.add_argument("--subset_min_support", type=float, default=None)
    args = parser.parse_args()

    subset_args = []
    if args.subset_rule_items_eq is not None:
        subset_args += ["--subset_rule_items_eq", str(args.subset_rule_items_eq)]
    if args.subset_min_support is not None:
        subset_args += ["--subset_min_support", str(args.subset_min_support)]

    # This script contains all steps executed after the association mining is done.
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    print(">>> Running Result Exploration...")
    run_cmd([py, "result_exploration/pipeline_efficiency.py", "--update-readme"], repo_root)
    run_cmd([py, "result_exploration/show_top_rules.py", "--update-readme"], repo_root)

    print("\n>>> Running Mining Visualizations...")
    run_cmd([py, "visualization/visualize_mining_results.py", *subset_args], repo_root)
    run_cmd([py, "visualization/visualize_raw_rules.py", *subset_args], repo_root)
    run_cmd([py, "visualization/visualize_transaction_distributions.py"], repo_root)

    run_cmd([py, "visualization/visualize_top_rules_spatial.py", "--top_n", "20", *subset_args], repo_root)
    run_cmd([py, "visualization/visualize_top_rules_spatial.py", "--top_n", "20", "--exclude_shared_items", *subset_args], repo_root)
    run_cmd([py, "visualization/visualize_top_rules_spatial.py", "--top_n", "20", "--exclude_self_loops", *subset_args], repo_root)

    print("\n>>> Running New Spatial Visualizations...")
    run_cmd([py, "visualization/visualize_dynamic_spatial_evolution.py", "--top_n", "3", "--method", "CN", "--min_n_stages", "3", *subset_args], repo_root)

    print("\n>>> Generating Consensus Reports...")
    # With self-loops (default)
    run_cmd([py, "result_exploration/generate_consensus_tables.py", *subset_args], repo_root)
    run_cmd([py, "visualization/visualize_consensus.py", *subset_args], repo_root)
    # Without self-loops (no_self)
    run_cmd([py, "result_exploration/generate_consensus_tables.py", "--no_self", *subset_args], repo_root)
    run_cmd([py, "visualization/visualize_consensus.py", "--no_self", *subset_args], repo_root)

    print("\n>>> Generating Variant Rules...")
    run_cmd([py, "visualization/visualize_stage_marker_rules.py", "--top_n", "20", "--min_n_stages", "3", *subset_args], repo_root)
    run_cmd([py, "visualization/visualize_stage_marker_rules.py", "--top_n", "20", "--no_self", "--min_n_stages", "3", *subset_args], repo_root)


if __name__ == "__main__":
    main()
