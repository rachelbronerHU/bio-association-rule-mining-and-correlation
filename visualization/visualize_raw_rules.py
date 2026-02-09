import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

# Add project root to sys.path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants

# --- Configuration ---
RESULTS_DIR = constants.RESULTS_DATA_DIR
OUTPUT_DIR = os.path.join(constants.RESULTS_PLOTS_DIR, 'raw_rules_report')
METHODS = ["BAG", "CN", "KNN_R"]

sns.set_theme(style="whitegrid")

def load_raw_data():
    """Loads RAW results CSVs."""
    raw_dfs = {}
    print(f"Loading RAW data from: {RESULTS_DIR}")
    
    for m in METHODS:
        # Load Raw CSV
        csv_path = os.path.join(RESULTS_DIR, f"results_{m}_RAW.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                raw_dfs[m] = df
            except Exception as e:
                print(f"Error loading {m} RAW CSV: {e}")
        else:
            print(f"Warning: {csv_path} not found.")
            
    return raw_dfs

def plot_raw_volcano(dfs, output_dir):
    """Volcano Plot Grid for Raw Rules: Lift vs -log10(FDR)."""
    print("Generating Raw Rules Volcano Grid...")
    
    # Filter methods that have FDR
    valid_methods = [m for m in dfs.keys() if "FDR" in dfs[m].columns]
    
    if not valid_methods:
        print("Skipping Raw Volcano: 'FDR' column not found in raw data.")
        return

    # Fixed Color Map for Stages
    stage_colors = {
        0: "forestgreen",
        1: "gold",
        2: "darkorange",
        3: "firebrick"
    }
    
    # Setup Grid
    n_plots = len(valid_methods)
    cols = 3
    rows = (n_plots // cols) + (1 if n_plots % cols > 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
    if n_plots == 1: axes = [axes]
    axes = np.array(axes).flatten()
    
    for i, m in enumerate(valid_methods):
        ax = axes[i]
        df = dfs[m].copy()
        
        # Sample if huge to avoid crashing plotter
        if len(df) > 20000:
            df = df.sample(20000, random_state=42)
            title_suffix = " (Sampled 20k)"
        else:
            title_suffix = ""
            
        # Log FDR
        df["log_fdr"] = -np.log10(df["FDR"] + 1e-10)
        
        # Color Logic
        hue_col = "Pathological stage" if "Pathological stage" in df.columns else "Group"
        palette = stage_colors if hue_col == "Pathological stage" else "Spectral"
        if hue_col not in df.columns: 
            hue_col = None
            palette = None
        
        sns.scatterplot(
            data=df, 
            x="Lift", 
            y="log_fdr", 
            hue=hue_col, 
            palette=palette, 
            alpha=0.6,
            s=20,
            ax=ax,
            legend='brief'
        )
        
        ax.axhline(-np.log10(0.05), color='red', linestyle='--', label="FDR=0.05", linewidth=1)
        ax.axvline(1.0, color='gray', linestyle='--', label="Lift=1.0", linewidth=1)
        
        # Calculate Statistics
        n_total = len(df)
        p_vals = df["FDR"]
        
        stats_subtitle = ""
        if not p_vals.empty:
            pcts = np.percentile(p_vals, [10, 25, 50, 75, 90])
            stats_subtitle = (
                f"N={n_total} | FDR P-Values: "
                f"10%:{pcts[0]:.1e}, 50%:{pcts[2]:.1e}, 90%:{pcts[4]:.1e}"
            )

        ax.set_title(f"Raw Rules: {m}{title_suffix}\n{stats_subtitle}", fontsize=10)
        ax.set_ylabel("-log10(FDR)")
        ax.set_xlabel("Lift")

        # Remove individual legends to create a shared one later
        if ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            # Store handles from the first plot for the shared legend
            if i == 0:
                shared_handles = handles
                shared_labels = labels
            ax.get_legend().remove()

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    # Add Shared Legend to the Figure
    if 'shared_handles' in locals():
        fig.legend(shared_handles, shared_labels, loc='center right', title="Stage/Group")
        # Adjust layout to make room for legend (Manual adjustment since constrained_layout handles internal spacing)
        # Note: constrained_layout and subplots_adjust don't mix well, but figure legend is usually fine.
        # We can disable constrained layout for this figure or just rely on the legend placement.
        
    plt.suptitle("Raw Rules Volcano Plots: Significance vs Strength (Colored by Stage)", fontsize=16)
    plt.savefig(f"{output_dir}/raw_volcano_grid.png")
    plt.close()
    print(f"Saved raw volcano plot to {output_dir}/raw_volcano_grid.png")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created directory: {OUTPUT_DIR}")

    raw_dfs = load_raw_data()
    
    if not raw_dfs:
        print("No raw data found. Exiting.")
        return

    plot_raw_volcano(raw_dfs, OUTPUT_DIR)

if __name__ == "__main__":
    main()
