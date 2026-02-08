#!/bin/bash
VOCAB_SIZE=32000 BATCH_SIZE=128 MAX_STEPS=30000 WARMUP_ITERS=2000 COSINE_CYCLE_ITERS=30000 MAX_LR=6e-4 MIN_LR=3e-4 WEIGHT_DECAY=0.05 bash run_training.sh train artifacts/owt_tokens/train.npy artifacts/owt_tokens/valid.npy mps
