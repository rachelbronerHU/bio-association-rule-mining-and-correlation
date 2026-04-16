import sys
import subprocess
from pathlib import Path


def run_cmd(command, repo_root):
    print(" ".join(command))
    result = subprocess.run(command, cwd=str(repo_root))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    # This script contains all steps executed after the association mining is done.
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    print(">>> Running Result Exploration...")
    run_cmd([py, "result_exploration/pipeline_efficiency.py", "--update-readme"], repo_root)
    run_cmd([py, "result_exploration/show_top_rules.py", "--update-readme"], repo_root)

    print("\n>>> Running Mining Visualizations...")
    run_cmd([py, "visualization/visualize_mining_results.py"], repo_root)
    run_cmd([py, "visualization/visualize_raw_rules.py"], repo_root)
    run_cmd([py, "visualization/visualize_transaction_distributions.py"], repo_root)

    run_cmd([py, "visualization/visualize_top_rules_spatial.py", "--top_n", "20"], repo_root)
    run_cmd([py, "visualization/visualize_top_rules_spatial.py", "--top_n", "20", "--exclude_shared_items"], repo_root)
    run_cmd([py, "visualization/visualize_top_rules_spatial.py", "--top_n", "20", "--exclude_self_loops"], repo_root)

    print("\n>>> Running New Spatial Visualizations...")
    run_cmd([py, "visualization/visualize_dynamic_spatial_evolution.py", "--top_n", "3", "--method", "CN"], repo_root)

    print("\n>>> Generating Consensus Reports...")
    # With self-loops (default)
    run_cmd([py, "result_exploration/generate_consensus_tables.py"], repo_root)
    run_cmd([py, "visualization/visualize_consensus.py"], repo_root)
    # Without self-loops (no_self)
    run_cmd([py, "result_exploration/generate_consensus_tables.py", "--no_self"], repo_root)
    run_cmd([py, "visualization/visualize_consensus.py", "--no_self"], repo_root)

    print("\n>>> Generating Variant Rules...")
    run_cmd([py, "visualization/visualize_stage_marker_rules.py", "--top_n", "20"], repo_root)
    run_cmd([py, "visualization/visualize_stage_marker_rules.py", "--top_n", "20", "--no_self"], repo_root)


if __name__ == "__main__":
    main()
