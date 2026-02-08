#!/usr/bin/env bash
# =============================================================================
# CS336 Basics - Training & LR Sweep Runner
#
# 用途：
# - 一键运行语言模型训练（train 模式）
# - 一键运行学习率扫参（sweep 模式）
# - 默认参数适配 Mac（CPU/MPS）低资源场景；可按需修改
# -----------------------------------------------------------------------------
# 使用示例：
# 1) 训练（推荐 MPS，自动检测设备）：
#    bash run_training.sh train /path/to/train.npy /path/to/valid.npy auto
#
# 2) 训练（CPU）：
#    bash run_training.sh train artifacts/lr_sweep_demo/train.npy artifacts/lr_sweep_demo/valid.npy cpu
#
# 3) 学习率扫参（推荐在 mps 或 cpu）：
#    bash run_training.sh sweep /path/to/train.npy /path/to/valid.npy mps
#    # 可在脚本下方修改 LR_LIST 等参数
# -----------------------------------------------------------------------------
# 说明：
# - 脚本通过 uv 运行 Python 入口（cs336_basics/train_lm.py 与 cs336_basics/lr_sweep.py）
# - 数据需是 token ID 的 numpy 数组（建议 uint16）；可用 np.load(..., mmap_mode='r')
# - 训练与扫参日志写入 artifacts 目录，便于画曲线与写报告
# =============================================================================

set -euo pipefail

MODE="${1:-}"
TRAIN_TOKENS="${2:-}"
VALID_TOKENS="${3:-}"
REQ_DEVICE="${4:-auto}"    # auto | mps | cpu

if [[ -z "${MODE}" || -z "${TRAIN_TOKENS}" || -z "${VALID_TOKENS}" ]]; then
  echo "用法：bash run_training.sh <train|sweep> <train_tokens.npy> <valid_tokens.npy> [device]"
  echo "示例：bash run_training.sh train /path/to/train.npy /path/to/valid.npy auto"
  exit 1
fi

# -----------------------------------------------------------------------------
# 设备选择：auto 时优先 mps；否则使用用户指定
# -----------------------------------------------------------------------------
detect_device() {
  local req="${1}"
  if [[ "${req}" == "auto" ]]; then
    if uv run python - <<'PY' | grep -q '^True$'; then
import torch; print(torch.backends.mps.is_available())
PY
      echo "mps"
    else
      echo "cpu"
    fi
  else
    echo "${req}"
  fi
}

DEVICE="$(detect_device "${REQ_DEVICE}")"
echo "Using device: ${DEVICE}"

# -----------------------------------------------------------------------------
# 通用数据类型与输出路径
# -----------------------------------------------------------------------------
TOKENS_DTYPE="uint16"                              # 词表 ≤ 65,536 时建议 uint16
TS="$(date +%Y%m%d-%H%M%S)"
ART_DIR="artifacts/run_${MODE}_${TS}"
mkdir -p "${ART_DIR}"

# -----------------------------------------------------------------------------
# 训练默认超参数（低资源 / Mac 友好）
# 可按需修改：vocab_size / model dims / steps / LR 等
# -----------------------------------------------------------------------------
VOCAB_SIZE="${VOCAB_SIZE:-10000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
D_MODEL="${D_MODEL:-512}"
D_FF="${D_FF:-1344}"               # ≈ (8/3) * d_model，64 对齐
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-16}"
ROPE_THETA="${ROPE_THETA:-10000}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-5000}"     # 32*5000*256≈40,960,000 tokens（低资源建议）
WARMUP_ITERS="${WARMUP_ITERS:-500}"
COSINE_CYCLE_ITERS="${COSINE_CYCLE_ITERS:-${MAX_STEPS}}"

MAX_LR="${MAX_LR:-1e-4}"           # 推荐学习率（来源于前面 sweep 的结论）
MIN_LR="${MIN_LR:-3e-5}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
EPS="${EPS:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

LOG_PATH="${LOG_PATH:-${ART_DIR}/exp_log.jsonl}"
CKPT_PATH="${CKPT_PATH:-${ART_DIR}/lm.ckpt}"

# 可选禁用 RMSNorm（NO_RMSNORM=1 时传递 --no_rmsnorm）
NO_RMSNORM="${NO_RMSNORM:-0}"
EXTRA_ARGS=""
if [[ "${NO_RMSNORM}" == "1" ]]; then
  EXTRA_ARGS="--no_rmsnorm"
fi
NO_POS_EMB="${NO_POS_EMB:-0}"
if [[ "${NO_POS_EMB}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --no_pos_emb"
fi
NORM_STYLE="${NORM_STYLE:-pre}"
FFN_STYLE="${FFN_STYLE:-swiglu}"
FFN_MATCH_PARAMS="${FFN_MATCH_PARAMS:-0}"

# -----------------------------------------------------------------------------
# 学习率扫参参数（可按需修改）
# -----------------------------------------------------------------------------
LR_LIST="${LR_LIST:-1e-4,1.5e-4,2e-4,2.5e-4,3e-4}"    # 先粗扫，后精扫
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_ITERS="${EVAL_ITERS:-10}"

# MPS 后向优化：在 lr_sweep 中可开启 aot_eager；训练默认关闭（更兼容）
SWEEP_COMPILE_BACKEND="${SWEEP_COMPILE_BACKEND:-aot_eager}"

# =============================================================================
# 运行入口
# =============================================================================
case "${MODE}" in
  train)
    echo "[Train] 输出目录：${ART_DIR}"
    echo "[Train] 日志：${LOG_PATH}"
    echo "[Train] 检查点：${CKPT_PATH}"
    uv run cs336_basics/train_lm.py \
      --train_tokens "${TRAIN_TOKENS}" \
      --valid_tokens "${VALID_TOKENS}" \
      --tokens_dtype "${TOKENS_DTYPE}" \
      --device "${DEVICE}" \
      --vocab_size "${VOCAB_SIZE}" \
      --context_length "${CONTEXT_LENGTH}" \
      --d_model "${D_MODEL}" \
      --num_layers "${NUM_LAYERS}" \
      --num_heads "${NUM_HEADS}" \
      --d_ff "${D_FF}" \
      --rope_theta "${ROPE_THETA}" \
      --batch_size "${BATCH_SIZE}" \
      --max_steps "${MAX_STEPS}" \
      --warmup_iters "${WARMUP_ITERS}" \
      --cosine_cycle_iters "${COSINE_CYCLE_ITERS}" \
      --max_lr "${MAX_LR}" \
      --min_lr "${MIN_LR}" \
      --ffn_style "${FFN_STYLE}" \
      $( [[ "${FFN_MATCH_PARAMS}" == "1" ]] && echo --ffn_match_params ) \
      --beta1 "${BETA1}" \
      --beta2 "${BETA2}" \
      --eps "${EPS}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --grad_clip_norm "${GRAD_CLIP_NORM}" \
      --log_path "${LOG_PATH}" \
      --checkpoint_path "${CKPT_PATH}" \
      --eval_every "${EVAL_EVERY}" \
      --eval_iters "${EVAL_ITERS}" \
      --save_every "$((MAX_STEPS/5))" \
      --norm_style "${NORM_STYLE}" \
      ${EXTRA_ARGS}
    echo "[Train] 完成。日志见 ${LOG_PATH}，检查点见 ${CKPT_PATH}"
    ;;

  sweep)
    echo "[Sweep] 输出基目录：${ART_DIR}"
    uv run cs336_basics/lr_sweep.py \
      --train_tokens "${TRAIN_TOKENS}" \
      --valid_tokens "${VALID_TOKENS}" \
      --tokens_dtype "${TOKENS_DTYPE}" \
      --device "${DEVICE}" \
      --vocab_size "${VOCAB_SIZE}" \
      --context_length "${CONTEXT_LENGTH}" \
      --d_model "${D_MODEL}" \
      --num_layers "${NUM_LAYERS}" \
      --num_heads "${NUM_HEADS}" \
      --d_ff "${D_FF}" \
      --rope_theta "${ROPE_THETA}" \
      --batch_size "${BATCH_SIZE}" \
      --max_steps "${MAX_STEPS}" \
      --warmup_iters "${WARMUP_ITERS}" \
      --cosine_cycle_iters "${COSINE_CYCLE_ITERS}" \
      --min_lr "${MIN_LR}" \
      --lr_list "${LR_LIST}" \
      --grad_clip_norm "${GRAD_CLIP_NORM}" \
      --beta1 "${BETA1}" \
      --beta2 "${BETA2}" \
      --eps "${EPS}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --eval_every "${EVAL_EVERY}" \
      --eval_iters "${EVAL_ITERS}" \
      --out_dir "${ART_DIR}" \
      --seed 42 \
      --compile_backend "${SWEEP_COMPILE_BACKEND}"
    echo "[Sweep] 完成。请查看 ${ART_DIR}/<timestamp>/lr_*/log.jsonl 与 summary.json"
    ;;

  *)
    echo "未知模式：${MODE}（支持 train 或 sweep）"
    exit 2
    ;;
esac



# 训练
# MAX_STEPS=5000 WARMUP_ITERS=2000 COSINE_CYCLE_ITERS=5000 MAX_LR=6e-4 MIN_LR=3e-4 WEIGHT_DECAY=0.05 VOCAB_SIZE=10000 bash run_training.sh train artifacts/tinystories_tokens/train.npy artifacts/tinystories_tokens/valid.npy mps
