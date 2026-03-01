import pandas as pd
import ast

df = pd.read_csv('results/full_run/plots/data_bias_report/bias_summary_by_organ.csv')

print('='*80)
print('CURRENT STATUS (min=3, max_majority=80%):')
print('='*80)
viable = df[df['Viable'] == '✓']
print(f'Viable: {len(viable)}/{len(df)} ({len(viable)/len(df)*100:.1f}%)')

# Calculate ratios
results = []
for _, row in df.iterrows():
    # Parse counts dict safely
    counts_str = row['Class_Counts']
    counts_dict = ast.literal_eval(counts_str)
    counts = list(counts_dict.values())
    ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    
    results.append({
        'Target': row['Target'],
        'Organ': row['Organ'],
        'Viable': row['Viable'],
        'Ratio': ratio,
        'Min': min(counts),
        'Max': max(counts),
        'N_Total': row['N_Total']
    })

rdf = pd.DataFrame(results)

print('\n' + '='*80)
print('CURRENTLY VIABLE - DETAILED RATIOS:')
print('='*80)
viable_results = rdf[rdf['Viable'] == '✓'].sort_values('Ratio', ascending=False)
print(viable_results[['Target', 'Organ', 'Min', 'Max', 'Ratio']].to_string(index=False))

print('\n' + '='*80)
print('RATIO THRESHOLD IMPACT ON VIABLE STRATA:')
print('='*80)
for threshold in [3, 4, 5, 6, 7, 10]:
    would_pass = rdf[(rdf['Viable'] == '✓') & (rdf['Ratio'] <= threshold)]
    print(f'  Max {threshold}:1 → {len(would_pass)}/{len(viable)} pass ({len(would_pass)/len(viable)*100:.0f}%)')

print('\n' + '='*80)
print('WHICH STRATA WOULD FAIL WITH DIFFERENT THRESHOLDS:')
print('='*80)
for threshold in [4, 5]:
    would_fail = rdf[(rdf['Viable'] == '✓') & (rdf['Ratio'] > threshold)]
    if not would_fail.empty:
        print(f'\nFails at {threshold}:1:')
        for _, row in would_fail.iterrows():
            print(f"  • {row['Target']:25s} × {row['Organ']:10s} ({row['Min']:2.0f}:{row['Max']:2.0f}, ratio={row['Ratio']:.1f})")
