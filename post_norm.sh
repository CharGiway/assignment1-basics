#!/bin/bash

NORM_STYLE=post BATCH_SIZE=128 MAX_STEPS=5000 WARMUP_ITERS=2000 COSINE_CYCLE_ITERS=5000 MAX_LR=6e-4 MIN_LR=3e-4 WEIGHT_DECAY=0.05 VOCAB_SIZE=10000 bash run_training.sh train artifacts/tinystories_tokens/train.npy artifacts/tinystories_tokens/valid.npy mps