#!/bin/bash
set -euo pipefail
VOCAB_PATH="${VOCAB_PATH:-artifacts/owt_32k/tinystories_bpe_vocab.json}"
MERGES_PATH="${MERGES_PATH:-artifacts/owt_32k/tinystories_bpe_merges.json}"
TRAIN_TEXT="${TRAIN_TEXT:-data/owt_train.txt}"
VALID_TEXT="${VALID_TEXT:-data/owt_valid.txt}"
OUT_DIR="${OUT_DIR:-artifacts/owt_tokens}"
TRAIN_OUT="${TRAIN_OUT:-${OUT_DIR}/train.npy}"
VALID_OUT="${VALID_OUT:-${OUT_DIR}/valid.npy}"
# 硬编码：按文本体积估算 token 数的 1/10，采用 4.8 bytes/token
TRAIN_BYTES="$(stat -f%z "${TRAIN_TEXT}")"
VALID_BYTES="$(stat -f%z "${VALID_TEXT}")"
TRAIN_LIMIT="$(awk -v b="${TRAIN_BYTES}" 'BEGIN { printf "%d", b/4.8/10.0 }')"
VALID_LIMIT="$(awk -v b="${VALID_BYTES}" 'BEGIN { printf "%d", b/4.8/10.0 }')"
mkdir -p "${OUT_DIR}"
uv run cs336_basics/encode_corpus.py --text_path "${TRAIN_TEXT}" --vocab_path "${VOCAB_PATH}" --merges_path "${MERGES_PATH}" --out_path "${TRAIN_OUT}" --limit_tokens "${TRAIN_LIMIT}" --dtype uint16 --device auto
uv run cs336_basics/encode_corpus.py --text_path "${VALID_TEXT}" --vocab_path "${VOCAB_PATH}" --merges_path "${MERGES_PATH}" --out_path "${VALID_OUT}" --limit_tokens "${VALID_LIMIT}" --dtype uint16 --device auto
