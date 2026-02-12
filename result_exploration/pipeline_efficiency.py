import pandas as pd
import os
import warnings

# Suppress DtypeWarning for large mixed-type datasets
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

DATA_DIR = "results/full_run/data"
METHODS = ["BAG", "CN", "KNN_R"]

def calculate_summary_stats():
    print("\n=== PIPELINE EFFICIENCY REPORT (Raw vs Final) ===\n")
    # Updated Header
    header = f"{'METHOD':<8} | {'RAW Count':<9} | {'FINAL Count':<11} | {'RETENTION %':<11} | {'RAW/FOV':<7} | {'FINAL/FOV':<9} | {'AVG LIFT':<10} | {'SIG (FDR<0.01)':<18} | {'SIG (P<0.01)':<18}"
    print(header)
    print("-" * len(header))

    METHODS = ["BAG", "CN", "KNN_R"]
    DATA_DIR = "results/full_run/data"

    for method in METHODS:
        final_path = os.path.join(DATA_DIR, f"results_{method}.csv")
        raw_path = os.path.join(DATA_DIR, f"results_{method}_RAW.csv")
        
        # Initialize Metrics
        raw_count = 0
        final_count = 0
        raw_fov_avg = 0
        final_fov_avg = 0
        raw_lift = 0
        final_lift = 0
        
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
                        raw_lift = df_raw['Lift'].mean()
                    
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
                        final_lift = df_final['Lift'].mean()
                    
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
        
        print(f"{method:<8} | {raw_count:<9} | {final_count:<11} | {retention:<11.2f} | {raw_fov_avg:<7.1f} | {final_fov_avg:<9.1f} | {raw_lift:.2f}->{final_lift:.2f} | {raw_sig_fdr:>5}->{final_sig_fdr:<5}      | {raw_sig_p:>5}->{final_sig_p:<5}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    calculate_summary_stats()