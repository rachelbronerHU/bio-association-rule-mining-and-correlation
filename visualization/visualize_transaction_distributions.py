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
TRANSACTION_DATA_DIR = constants.TRANSACTION_DATA_DIR
OUTPUT_DIR = os.path.join(constants.RESULTS_PLOTS_DIR, 'transaction_distributions')
METHODS = ["BAG", "CN", "KNN_R"]

def load_transaction_data(method):
    """
    Loads all transaction data for a given method.
    Returns a list of cell counts across all FOVs.
    """
    method_dir = os.path.join(TRANSACTION_DATA_DIR, method)
    
    if not os.path.exists(method_dir):
        print(f"Warning: Directory not found for {method}: {method_dir}")
        return []
    
    all_cell_counts = []
    fov_files = [f for f in os.listdir(method_dir) if f.endswith('.csv')]
    
    print(f"Loading {len(fov_files)} FOV files for {method}...")
    
    for fov_file in fov_files:
        file_path = os.path.join(method_dir, fov_file)
        try:
            df = pd.read_csv(file_path)
            if 'num_cells' in df.columns:
                all_cell_counts.extend(df['num_cells'].tolist())
        except Exception as e:
            print(f"Warning: Failed to load {fov_file}: {e}")
    
    return all_cell_counts

def plot_method_distribution(ax, cell_counts, method_name):
    """
    Plots the distribution of cells per transaction for a single method.
    """
    if not cell_counts:
        ax.text(0.5, 0.5, f"No data available for {method_name}", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    # Calculate statistics
    mean_cells = np.mean(cell_counts)
    median_cells = np.median(cell_counts)
    min_cells = np.min(cell_counts)
    max_cells = np.max(cell_counts)
    
    # Create histogram
    ax.hist(cell_counts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add vertical lines for mean and median
    ax.axvline(mean_cells, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cells:.1f}')
    ax.axvline(median_cells, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_cells:.1f}')
    
    # Labels and title
    ax.set_xlabel('Number of Cells per Transaction', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{method_name} Method\n'
                 f'Min: {min_cells} | Max: {max_cells} | Total Transactions: {len(cell_counts):,}',
                 fontsize=13, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

def plot_comparison_boxplot(ax, data_dict):
    """
    Creates a boxplot comparing distributions across methods.
    """
    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    
    for method in METHODS:
        if method in data_dict and data_dict[method]:
            data_to_plot.append(data_dict[method])
            labels.append(method)
    
    if not data_to_plot:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    # Create boxplot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    # Color the boxes
    colors = ['steelblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Number of Cells per Transaction', fontsize=12)
    ax.set_title('Comparison Across Methods', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

def plot_comparison_violin(ax, data_dict):
    """
    Creates a violin plot comparing distributions across methods.
    """
    # Prepare data for violin plot
    plot_data = []
    
    for method in METHODS:
        if method in data_dict and data_dict[method]:
            for count in data_dict[method]:
                plot_data.append({'Method': method, 'Cells': count})
    
    if not plot_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create violin plot
    sns.violinplot(data=df_plot, x='Method', y='Cells', ax=ax, 
                   palette=['steelblue', 'lightcoral', 'lightgreen'])
    
    ax.set_ylabel('Number of Cells per Transaction', fontsize=12)
    ax.set_xlabel('')
    ax.set_title('Distribution Comparison (Violin Plot)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

def create_summary_stats_table(data_dict):
    """
    Creates a summary statistics table for all methods.
    """
    stats_data = []
    
    for method in METHODS:
        if method in data_dict and data_dict[method]:
            cell_counts = data_dict[method]
            stats_data.append({
                'Method': method,
                'Transactions': len(cell_counts),
                'Mean': np.mean(cell_counts),
                'Median': np.median(cell_counts),
                'Std Dev': np.std(cell_counts),
                'Min': np.min(cell_counts),
                'Max': np.max(cell_counts),
                'Q25': np.percentile(cell_counts, 25),
                'Q75': np.percentile(cell_counts, 75)
            })
    
    df_stats = pd.DataFrame(stats_data)
    return df_stats

def main():
    print("="*60)
    print("Transaction Cell Count Distribution Analysis")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data for all methods
    data_dict = {}
    for method in METHODS:
        cell_counts = load_transaction_data(method)
        data_dict[method] = cell_counts
        print(f"{method}: {len(cell_counts):,} transactions loaded")
    
    print("\n" + "="*60)
    
    # Create individual distribution plots
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(12, 5 * len(METHODS)))
    if len(METHODS) == 1:
        axes = [axes]
    
    for idx, method in enumerate(METHODS):
        plot_method_distribution(axes[idx], data_dict[method], method)
    
    plt.tight_layout()
    individual_plot_path = os.path.join(OUTPUT_DIR, 'transaction_distributions_individual.png')
    plt.savefig(individual_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {individual_plot_path}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_comparison_boxplot(axes[0], data_dict)
    plot_comparison_violin(axes[1], data_dict)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(OUTPUT_DIR, 'transaction_distributions_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {comparison_plot_path}")
    
    # Create and save summary statistics table
    df_stats = create_summary_stats_table(data_dict)
    stats_path = os.path.join(OUTPUT_DIR, 'transaction_statistics_summary.csv')
    df_stats.to_csv(stats_path, index=False, float_format='%.2f')
    print(f"Saved: {stats_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(df_stats.to_string(index=False))
    print("="*60)
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
