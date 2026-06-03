#!/bin/bash
# Download WikiText-103 raw dataset for training
# ~516MB, ~103M tokens after tokenization
# Same format as WikiText-2 — zero integration work needed
# Uses HF datasets library (pip install datasets)

set -euo pipefail

DATA_DIR="data/training/active"
OUTFILE="$DATA_DIR/wikitext103_train.txt"

if [ -f "$OUTFILE" ]; then
    echo "WikiText-103 already exists at $OUTFILE"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "=== Downloading WikiText-103 Raw via HF datasets ==="

source .venv/bin/activate
python -c "
from datasets import load_dataset
import os

print('Downloading WikiText-103-raw-v1...')
ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
print(f'Downloaded: {len(ds)} training examples')

outpath = '$OUTFILE'
print(f'Writing to {outpath}...')
with open(outpath, 'w') as f:
    for item in ds:
        f.write(item['text'] + '\n')

size_mb = os.path.getsize(outpath) / (1024*1024)
lines = sum(1 for _ in open(outpath))
print(f'Done: {lines} lines, {size_mb:.1f} MB')
"

echo "=== Done ==="
