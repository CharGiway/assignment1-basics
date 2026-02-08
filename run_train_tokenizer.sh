#!/bin/bash
set -e

if command -v nproc >/dev/null 2>&1; then
  NWORKERS=$(nproc)
else
  NWORKERS=$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi

# uv run cs336_basics/train_bpe_tinystories.py \
#   --input_path data/TinyStoriesV2-GPT4-train.txt \
#   --vocab_size 10000 \
#   --out_dir artifacts \
#   --n_workers "$NWORKERS" \
#   --profile

uv run cs336_basics/train_bpe_tinystories.py \
  --input_path data/owt_train.txt \
  --vocab_size 32000 \
  --out_dir artifacts/owt_32k \
  --n_workers "$NWORKERS" \
  --profile
