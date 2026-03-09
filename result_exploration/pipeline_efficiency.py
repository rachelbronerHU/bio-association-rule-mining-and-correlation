import pandas as pd
import os
import warnings
import sys

# Suppress DtypeWarning for large mixed-type datasets
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import RESULTS_DATA_DIR, METHODS, CONFIG

_MIN_LIFT = CONFIG["MIN_LIFT"]
_MAX_NEG_LIFT = CONFIG["MAX_NEGATIVE_LIFT"]

DATA_DIR = RESULTS_DATA_DIR

def _fmt_lift(raw_val, final_val):
    if pd.isna(raw_val) and pd.isna(final_val):
        return "N/A"
    r = f"{raw_val:.2f}" if not pd.isna(raw_val) else "N/A"
    f_ = f"{final_val:.2f}" if not pd.isna(final_val) else "N/A"
    return f"{r}->{f_}"


def calculate_summary_stats():
    print("\n=== PIPELINE EFFICIENCY REPORT (Raw vs Final) ===\n")
    # Updated Header
    header = f"{'METHOD':<8} | {'RAW Count':<9} | {'FINAL Count':<11} | {'RETENTION %':<11} | {'RAW/FOV':<7} | {'FINAL/FOV':<9} | {'AVG LIFT+(R->F)':<16} | {'AVG LIFT-(R->F)':<16} | {'SIG (FDR<0.01)':<18} | {'SIG (P<0.01)':<18}"
    print(header)
    print("-" * len(header))

    for method in METHODS:
        final_path = os.path.join(DATA_DIR, f"results_{method}.csv")
        raw_path = os.path.join(DATA_DIR, f"results_{method}_RAW.csv")
        
        # Initialize Metrics
        raw_count = 0
        final_count = 0
        raw_fov_avg = 0
        final_fov_avg = 0
        raw_lift_pos = raw_lift_neg = float("nan")
        final_lift_pos = final_lift_neg = float("nan")
        raw_n_pos = raw_n_neg = 0
        final_n_pos = final_n_neg = 0
        
        raw_sig_fdr = "N/A"
        final_sig_fdr = "N/A"
        raw_sig_p = "N/A"
        final_sig_p = "N/A"
        
        # Load Raw
        if os.path.exists(raw_path):
            try:
                df_raw = pd.read_csv(raw_path, low_memory=False)
                if not df_raw.empty:
                    raw_count = len(df_raw)
                    raw_fov_avg = raw_count / df_raw['FOV'].nunique()
                    if 'Lift' in df_raw.columns:
                        pos = df_raw[df_raw['Lift'] >= _MIN_LIFT]
                        neg = df_raw[df_raw['Lift'] <= _MAX_NEG_LIFT]
                        raw_n_pos, raw_n_neg = len(pos), len(neg)
                        raw_lift_pos = pos['Lift'].mean() if raw_n_pos else float("nan")
                        raw_lift_neg = neg['Lift'].mean() if raw_n_neg else float("nan")
                    
                    if "FDR" in df_raw.columns:
                        n_sig = len(df_raw[df_raw["FDR"] < 0.01])
                        raw_sig_fdr = f"{(n_sig / raw_count * 100):.1f}%"
                        
                    if "p_value" in df_raw.columns or "P_Value" in df_raw.columns:
                        col = "p_value" if "p_value" in df_raw.columns else "P_Value"
                        n_sig_p = len(df_raw[df_raw[col] < 0.01])
                        raw_sig_p = f"{(n_sig_p / raw_count * 100):.1f}%"
            except: pass

        # Load Final
        if os.path.exists(final_path):
            try:
                # low_memory=False resolves DtypeWarning
                df_final = pd.read_csv(final_path, low_memory=False)
                if not df_final.empty:
                    final_count = len(df_final)
                    final_fov_avg = final_count / df_final['FOV'].nunique()
                    if 'Lift' in df_final.columns:
                        pos = df_final[df_final['Lift'] >= _MIN_LIFT]
                        neg = df_final[df_final['Lift'] <= _MAX_NEG_LIFT]
                        final_n_pos, final_n_neg = len(pos), len(neg)
                        final_lift_pos = pos['Lift'].mean() if final_n_pos else float("nan")
                        final_lift_neg = neg['Lift'].mean() if final_n_neg else float("nan")
                    
                    if "FDR" in df_final.columns:
                        n_sig = len(df_final[df_final["FDR"] < 0.01])
                        final_sig_fdr = f"{(n_sig / final_count * 100):.1f}%"

                    if "p_value" in df_final.columns or "P_Value" in df_final.columns:
                        col = "p_value" if "p_value" in df_final.columns else "P_Value"
                        n_sig_p = len(df_final[df_final[col] < 0.01])
                        final_sig_p = f"{(n_sig_p / final_count * 100):.1f}%"
            except: pass
            
        # Derived
        retention = (final_count / raw_count * 100) if raw_count > 0 else 0

        lift_pos_str = _fmt_lift(raw_lift_pos, final_lift_pos)
        lift_neg_str = _fmt_lift(raw_lift_neg, final_lift_neg)

        print(f"{method:<8} | {raw_count:<9} | {final_count:<11} | {retention:<11.2f} | {raw_fov_avg:<7.1f} | {final_fov_avg:<9.1f} | {lift_pos_str:<16} | {lift_neg_str:<16} | {raw_sig_fdr:>5}->{final_sig_fdr:<5}      | {raw_sig_p:>5}->{final_sig_p:<5}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    calculate_summary_stats()