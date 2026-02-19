"""
One-time script to filter existing results and remove rules involving rare cell types.
This script applies the same rare cell type filtering logic to already-mined results.
"""

import pandas as pd
import numpy as np
import os
import sys
from collections import Counter

# Add project root to path
sys.path.append(os.path.dirname(__file__))
import constants

# Configuration (matching run_association_mining.py)
MIN_CELL_TYPE_FREQUENCY = 5
MIN_SUPPORT = 0.01

# Paths
CELL_TABLE_PATH = os.path.join(constants.MIBI_GUT_DIR_PATH, 'cell_table.csv')
RESULTS_DIR = constants.RESULTS_DATA_DIR
METHODS = ["BAG", "CN", "KNN_R"]

def load_cell_type_counts_per_fov(cell_table_path):
    """
    Load cell table and compute cell type counts for each FOV.
    Returns: dict[FOV] -> Counter(cell_type -> count)
    """
    print(f"Loading cell table from {cell_table_path}...")
    df_cells = pd.read_csv(cell_table_path)
    
    fov_cell_counts = {}
    for fov in df_cells['fov'].unique():
        df_fov = df_cells[df_cells['fov'] == fov]
        cell_types = df_fov['cell type'].values
        fov_cell_counts[fov] = Counter(cell_types)
    
    print(f"Loaded cell type counts for {len(fov_cell_counts)} FOVs")
    return fov_cell_counts

def calculate_threshold(n_cells, min_absolute=MIN_CELL_TYPE_FREQUENCY, min_percentage=MIN_SUPPORT):
    """Calculate adaptive threshold for rare cell filtering."""
    return max(min_absolute, int(min_percentage * n_cells))

def extract_cell_types_from_rule(antecedents_str, consequents_str, method):
    """
    Extract cell types from rule strings, removing _CENTER/_NEIGHBOR suffixes.
    
    For BAG: ['Muscle', 'Epithelial'] -> ['Muscle', 'Epithelial']
    For CN/KNN_R: ['Muscle_CENTER', 'Epithelial_NEIGHBOR'] -> ['Muscle', 'Epithelial']
    """
    import ast
    
    try:
        ant_list = ast.literal_eval(antecedents_str)
        con_list = ast.literal_eval(consequents_str)
    except:
        return []
    
    all_types = ant_list + con_list
    
    # Remove suffixes for CN/KNN_R methods
    if method in ['CN', 'KNN_R']:
        clean_types = [t.replace('_CENTER', '').replace('_NEIGHBOR', '') for t in all_types]
    else:
        clean_types = all_types
    
    return clean_types

def filter_results_file(results_path, fov_cell_counts, method):
    """
    Filter a results CSV file to remove rules with rare cell types.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {results_path}")
    print(f"Method: {method} (handles _CENTER/_NEIGHBOR: {method in ['CN', 'KNN_R']})")
    print(f"{'='*60}")
    
    if not os.path.exists(results_path):
        print(f"File not found, skipping.")
        return
    
    # Load results
    df_results = pd.read_csv(results_path)
    n_original = len(df_results)
    print(f"Original rules: {n_original:,}")
    
    if n_original == 0:
        print("No rules to filter.")
        return
    
    # Sample a few rules to show what we're processing
    if n_original > 0:
        sample_rule = df_results.iloc[0]
        sample_types = extract_cell_types_from_rule(
            sample_rule['Antecedents'], 
            sample_rule['Consequents'], 
            method
        )
        print(f"Example rule: {sample_rule['Antecedents']} -> {sample_rule['Consequents']}")
        print(f"  Extracted cell types: {sample_types}")
    
    # Filter rules
    keep_mask = []
    filtered_stats = {
        'total': 0,
        'filtered_rare': 0,
        'fov_missing': 0
    }
    
    rare_rule_examples = []  # Store examples for reporting
    
    for idx, row in df_results.iterrows():
        fov = row['FOV']
        
        # Check if we have cell counts for this FOV
        if fov not in fov_cell_counts:
            # Keep the rule but note we couldn't check it
            keep_mask.append(True)
            filtered_stats['fov_missing'] += 1
            continue
        
        filtered_stats['total'] += 1
        
        # Get cell types involved in this rule (handles _CENTER/_NEIGHBOR)
        cell_types_in_rule = extract_cell_types_from_rule(
            row['Antecedents'], 
            row['Consequents'],
            method
        )
        
        if not cell_types_in_rule:
            keep_mask.append(True)
            continue
        
        # Get FOV stats
        fov_counts = fov_cell_counts[fov]
        n_cells = sum(fov_counts.values())
        threshold = calculate_threshold(n_cells)
        
        # Check if any cell type in rule is below threshold
        rare_types = []
        for ct in cell_types_in_rule:
            count = fov_counts.get(ct, 0)
            if count < threshold:
                rare_types.append((ct, count))
        
        has_rare = len(rare_types) > 0
        
        if has_rare:
            keep_mask.append(False)
            filtered_stats['filtered_rare'] += 1
            
            # Save first few examples
            if len(rare_rule_examples) < 5:
                rare_rule_examples.append({
                    'fov': fov,
                    'rule': f"{row['Antecedents']} -> {row['Consequents']}",
                    'rare_types': rare_types,
                    'threshold': threshold,
                    'n_cells': n_cells,
                    'lift': row.get('Lift', 'N/A'),
                    'support': row.get('Support', 'N/A')
                })
        else:
            keep_mask.append(True)
    
    # Apply filter
    df_filtered = df_results[keep_mask].copy()
    n_filtered = len(df_filtered)
    n_removed = n_original - n_filtered
    
    print(f"\nFiltering Results:")
    print(f"  Rules checked: {filtered_stats['total']:,}")
    print(f"  Rules with rare cell types removed: {filtered_stats['filtered_rare']:,}")
    print(f"  Rules from unknown FOVs (kept): {filtered_stats['fov_missing']:,}")
    print(f"  Final count: {n_filtered:,} rules ({n_removed:,} removed, {100*n_removed/n_original:.1f}%)")
    
    # Show examples of filtered rules
    if rare_rule_examples:
        print(f"\n  Examples of removed rules:")
        for ex in rare_rule_examples[:3]:
            print(f"    FOV: {ex['fov']} (threshold={ex['threshold']}, n_cells={ex['n_cells']})")
            print(f"    Rule: {ex['rule']}")
            print(f"    Rare types: {ex['rare_types']}")
            print(f"    Lift: {ex['lift']}, Support: {ex['support']}")
            print()
    
    if n_removed > 0:
        # Create backup
        backup_path = results_path.replace('.csv', '_BACKUP_BEFORE_FILTER.csv')
        if not os.path.exists(backup_path):
            df_results.to_csv(backup_path, index=False)
            print(f"  ✓ Backup saved: {backup_path}")
        
        # Save filtered results
        df_filtered.to_csv(results_path, index=False)
        print(f"  ✓ Filtered results saved: {results_path}")
    else:
        print(f"  → No changes needed, file unchanged.")

def main():
    print("="*60)
    print("RARE CELL TYPE FILTER - EXISTING RESULTS CLEANUP")
    print("="*60)
    print(f"Configuration:")
    print(f"  MIN_CELL_TYPE_FREQUENCY: {MIN_CELL_TYPE_FREQUENCY}")
    print(f"  MIN_SUPPORT: {MIN_SUPPORT}")
    print(f"  Adaptive threshold: max({MIN_CELL_TYPE_FREQUENCY}, {MIN_SUPPORT} * n_cells)")
    print(f"\nNote: For CN/KNN_R methods, _CENTER and _NEIGHBOR suffixes")
    print(f"      are automatically stripped when matching cell types.")
    print("="*60)
    
    # Load cell type counts
    fov_cell_counts = load_cell_type_counts_per_fov(CELL_TABLE_PATH)
    
    # Process each method's results (including RAW)
    for method in METHODS:
        # Filter main results
        results_path = os.path.join(RESULTS_DIR, f"results_{method}.csv")
        filter_results_file(results_path, fov_cell_counts, method)
        
        # Filter raw results if they exist
        raw_results_path = os.path.join(RESULTS_DIR, f"results_{method}_RAW.csv")
        if os.path.exists(raw_results_path):
            filter_results_file(raw_results_path, fov_cell_counts, method)
    
    print("\n" + "="*60)
    print("✓ FILTERING COMPLETE")
    print("="*60)
    print("\nBackup files created with suffix: _BACKUP_BEFORE_FILTER.csv")
    print("You can delete these backups once you verify the filtered results.")
    print("\nTo restore original files if needed:")
    print("  - Rename *_BACKUP_BEFORE_FILTER.csv files back to original names")

if __name__ == "__main__":
    main()
