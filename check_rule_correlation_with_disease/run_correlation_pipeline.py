"""
Main runner for the correlation-with-disease pipeline.

Usage:
    python run_correlation_pipeline.py                  # run both flows
    python run_correlation_pipeline.py --dry_run        # preview stratification plan
    python run_correlation_pipeline.py --debug          # use results/debug_run as input
    python run_correlation_pipeline.py --no_self        # exclude self-interacting rules
    python run_correlation_pipeline.py --only simple    # run only simple-stats flow
    python run_correlation_pipeline.py --only ml        # run only ML (advanced) flow
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants
from data_exploration.check_data_bias import load_stratified_biopsies
from check_rule_correlation_with_disease.stratified_utils import CONTROLS_ELIGIBLE, TARGETS, filter_viable_stratum

import run_robust_simple_stats
import advanced_discovery

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FOVS_PATH = os.path.join(PROJECT_ROOT, constants.MIBI_GUT_DIR_PATH, "fovs_metadata.csv")

# ── Dry-run ───────────────────────────────────────────────────────────────────

def run_dry_run():
    """Print the stratification plan without running any models."""
    print("\nDry Run Mode - No modeling will be performed")
    print("=" * 56)

    df_biopsies = load_stratified_biopsies()

    df_fovs = pd.read_csv(FOVS_PATH)
    df_fovs = df_fovs[~df_fovs["FOV"].astype(str).str.startswith("S_")]
    biopsy_to_organ = df_biopsies.set_index("Biopsy_ID")["Organ"]
    df_fovs["Organ"] = df_fovs["Patient"].map(biopsy_to_organ)
    fov_counts_by_organ = df_fovs.groupby("Organ").size()

    organs = sorted([o for o in df_biopsies["Organ"].unique() if o != "Unknown"])

    total_run = 0
    total_skip = 0

    for organ in organs:
        df_organ = df_biopsies[df_biopsies["Organ"] == organ]
        n_biopsies = len(df_organ)
        n_fovs = fov_counts_by_organ.get(organ, 0)
        n_gvhd = int((~df_organ["Is_Control"]).sum())
        n_controls = int(df_organ["Is_Control"].sum())

        print(f"\nOrgan: {organ} ({n_biopsies} biopsies, {n_fovs} FOVs)")
        print(f"  GVHD: {n_gvhd}, Controls: {n_controls}")

        for target in TARGETS:
            if target not in df_biopsies.columns:
                continue

            include_controls = CONTROLS_ELIGIBLE.get(target, True)
            df_t = df_organ if include_controls else df_organ[~df_organ["Is_Control"]]
            population_label = "GVHD + Controls" if include_controls else "GVHD only"
            eligibility_label = "Yes" if include_controls else "No (GVHD only)"

            df_filtered, note = filter_viable_stratum(df_t, target)

            print(f"\n  Target: {target}")
            if df_filtered is not None:
                class_counts = dict(
                    df_filtered[df_filtered[target].notna()][target].value_counts().sort_index()
                )
                print(f"    ✓ VIABLE - Will run")
                print(f"    Population: {population_label} (eligibility: {eligibility_label})")
                print(f"    Classes: {class_counts}")
                if note:
                    print(f"    Note: {note}")
                total_run += 1
            else:
                print(f"    ✗ SKIP - {note}")
                total_skip += 1

    print(f"\n{'=' * 56}")
    print(f"Summary: {total_run} analyses will run, {total_skip} will be skipped\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation-with-disease pipeline runner")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview stratification plan without running models.")
    parser.add_argument("--no_self", action="store_true",
                        help="Exclude rules with any shared cell types (strict no-self).")
    parser.add_argument("--only", choices=["simple", "ml"], default=None,
                        help="Run only one flow: 'simple' or 'ml'.")
    args = parser.parse_args()

    # Ensure relative paths (e.g. in load_stratified_biopsies) resolve from project root
    os.chdir(PROJECT_ROOT)

    if args.dry_run:
        run_dry_run()
        sys.exit(0)

    if args.only in (None, "simple"):
        print("=== Running Simple Stats flow ===")
        run_robust_simple_stats.run_pipeline(no_self=args.no_self)

    if args.only in (None, "ml"):
        print("=== Running ML (Advanced Discovery) flow ===")
        advanced_discovery.run_pipeline(no_self=args.no_self)
