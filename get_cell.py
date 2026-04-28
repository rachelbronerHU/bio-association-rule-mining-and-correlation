import json
with open('visualization/notebooks/investigation_02_bubble_metrics.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if 'HIGHLIGHT_RULES' in src and 'pattern' in src and 'display(stats_df)' in src:
            print(f'--- CELL {i} ---')
            print(src)
