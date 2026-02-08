import pandas as pd
import os
import warnings

# Suppress DtypeWarning for large mixed-type datasets
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

DATA_DIR = "results/full_run/data"
METHODS = ["BAG", "CN", "KNN_R", "WINDOW", "GRID"]

def calculate_summary_stats():
    print("\n=== PIPELINE EFFICIENCY REPORT (Raw vs Final) ===\n")
    header = f"{'METHOD':<10} | {'RAW Count':<10} | {'FINAL Count':<11} | {'RETENTION %':<11} | {'RAW/FOV':<9} | {'FINAL/FOV':<9} | {'AVG LIFT (R->F)':<15}"
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
        raw_lift = 0
        final_lift = 0
        
        # Load Final
        if os.path.exists(final_path):
            try:
                # low_memory=False resolves DtypeWarning
                df_final = pd.read_csv(final_path, low_memory=False)
                final_count = len(df_final)
                final_fov_avg = final_count / df_final['FOV'].nunique() if not df_final.empty else 0
                final_lift = df_final['Lift'].mean() if not df_final.empty else 0
            except: pass
            
        # Load Raw
        if os.path.exists(raw_path):
            try:
                df_raw = pd.read_csv(raw_path, low_memory=False)
                raw_count = len(df_raw)
                raw_fov_avg = raw_count / df_raw['FOV'].nunique() if not df_raw.empty else 0
                raw_lift = df_raw['Lift'].mean() if not df_raw.empty else 0
            except: pass
            
        # Derived
        retention = (final_count / raw_count * 100) if raw_count > 0 else 0
        
        print(f"{method:<10} | {raw_count:<10} | {final_count:<11} | {retention:<11.2f} | {raw_fov_avg:<9.1f} | {final_fov_avg:<9.1f} | {raw_lift:.2f} -> {final_lift:.2f}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    calculate_summary_stats()