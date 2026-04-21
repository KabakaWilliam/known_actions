#!/usr/bin/env bash
# check_traces.sh — print a pivot table of trace counts per agent × dataset.
# Cells flagged with ! are below 90% of the collection target.
cd "$(dirname "$0")/traces"

python3 -c "
import os, glob
agents = sorted(d for d in os.listdir('.') if os.path.isdir(d) and d not in ('models','classifiers','legacy_classifiers'))
datasets = ['2wikimultihop','frames','webshop','deepshop','webgames']
ds_splits = {
    '2wikimultihop': ['2wikimultihop_train','2wikimultihop_val','2wikimultihop_test'],
    'frames':        ['frames_test'],
    'webshop':       ['webshop_train','webshop_val','webshop_test'],
    'deepshop':      ['deepshop_ood'],
    'webgames':      ['webgames_train','webgames_val','webgames_test'],
}
targets = {'2wikimultihop':300,'frames':824,'webshop':300,'deepshop':150,'webgames':150}
print(f'{\"Agent\":<25}', '  '.join(f'{ds[:10]:<10}' for ds in datasets))
print('-'*85)
for agent in agents:
    row = f'{agent:<25}'
    for ds in datasets:
        total = sum(len(glob.glob(f'{agent}/{split}/**/*.json', recursive=True)) for split in ds_splits[ds])
        flag = '!' if total < targets[ds] * 0.9 else ' '
        row += f'  {flag}{total:<9}'
    print(row)
"
