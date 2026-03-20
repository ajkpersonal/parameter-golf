#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/artifacts"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-dense_calib_${RUN_STAMP}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OUTER_TIMEOUT="${OUTER_TIMEOUT:-30m}"
WRAPPER_LOG="$ROOT_DIR/logs/${RUN_ID}.runner.txt"
MODEL_OUT="${MODEL_OUT:-$ROOT_DIR/artifacts/${RUN_ID}.pt}"
QUANT_OUT="${QUANT_OUT:-$ROOT_DIR/artifacts/${RUN_ID}.int8.ptz}"

COMMON_ENV=(
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
  "DATA_PATH=${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
  "TOKENIZER_PATH=${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
  "VOCAB_SIZE=${VOCAB_SIZE:-1024}"
  "NUM_LAYERS=${NUM_LAYERS:-9}"
  "MODEL_DIM=${MODEL_DIM:-512}"
  "NUM_HEADS=${NUM_HEADS:-8}"
  "NUM_KV_HEADS=${NUM_KV_HEADS:-4}"
  "MLP_MULT=${MLP_MULT:-2}"
  "TIE_EMBEDDINGS=${TIE_EMBEDDINGS:-1}"
  "TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-1024}"
  "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-524288}"
  "WARMUP_STEPS=${WARMUP_STEPS:-8}"
  "WARMDOWN_ITERS=${WARMDOWN_ITERS:-3000}"
  "VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}"
  "TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-50}"
  "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-720}"
  "TIED_EMBED_LR=${TIED_EMBED_LR:-0.04}"
  "MATRIX_LR=${MATRIX_LR:-0.032}"
  "SCALAR_LR=${SCALAR_LR:-0.032}"
  "MUON_MOMENTUM=${MUON_MOMENTUM:-0.99}"
  "MUON_MOMENTUM_WARMUP_START=${MUON_MOMENTUM_WARMUP_START:-0.92}"
  "MUON_MOMENTUM_WARMUP_STEPS=${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
  "GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}"
  "COMPRESSION_AWARE_MODE="
  "SEED=${SEED:-1337}"
  "RUN_ID=${RUN_ID}"
)

echo "Starting dense calibration run: $RUN_ID"
echo "Wrapper log: $WRAPPER_LOG"

rm -f "$WRAPPER_LOG"

timeout "$OUTER_TIMEOUT" \
  env "${COMMON_ENV[@]}" \
  python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py \
  > "$WRAPPER_LOG" 2>&1

cp "$ROOT_DIR/final_model.pt" "$MODEL_OUT"
cp "$ROOT_DIR/final_model.int8.ptz" "$QUANT_OUT"

echo "Saved checkpoint: $MODEL_OUT"
echo "Saved quant artifact: $QUANT_OUT"
echo "Trainer log: $ROOT_DIR/logs/${RUN_ID}.txt"
