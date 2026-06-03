#!/bin/bash
# Wait for step_003000 checkpoint, then run MMLU + HellaSwag evals
CKPT="data/checkpoints/cycle22_extend/step_003000.ckpt"
TOK="data/training/tokenizer_4k.bin"
LOGDIR="data/experiments/exp_cycle22_extend"

echo "Waiting for $CKPT..."
while [ ! -f "$CKPT" ]; do sleep 30; done
echo "Checkpoint found at $(date)!"

# Give it a moment to finish writing
sleep 5

source .venv/bin/activate

echo "=== Running MMLU eval on step 3000 ==="
python tools/eval_mmlu.py --checkpoint "$CKPT" --tokenizer "$TOK" --max-per-subject 5 2>&1 | tee "$LOGDIR/mmlu_step3000.txt"

echo "=== Running HellaSwag eval on step 3000 ==="
python tools/eval_hellaswag.py --checkpoint "$CKPT" --tokenizer "$TOK" --max-items 100 2>&1 | tee "$LOGDIR/hellaswag_step3000.txt"

echo "=== Evals complete ==="
