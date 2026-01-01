#!/bin/bash
set -e

uv run cs336_basics/train_bpe_tinystories.py \
  --input_path data/TinyStoriesV2-GPT4-train.txt \
  --vocab_size 10000 \
  --out_dir artifacts \
  --n_workers $(nproc) \
  --profile