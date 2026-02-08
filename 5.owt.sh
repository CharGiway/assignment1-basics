#!/bin/bash
VOCAB_SIZE=32000 BATCH_SIZE=128 MAX_STEPS=15000 WARMUP_ITERS=3000 COSINE_CYCLE_ITERS=15000 MAX_LR=3e-4 MIN_LR=1e-4 WEIGHT_DECAY=0.05 DROPOUT_P=0.1 SAVE_BEST=1 PATIENCE=0 MIN_DELTA=0.0 bash run_training.sh train artifacts/owt_tokens/train.npy artifacts/owt_tokens/valid.npy mps
