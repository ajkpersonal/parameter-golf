#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RUN_ID="${RUN_ID:-dense_integrated_$(date -u +%Y%m%d_%H%M%S)}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-3}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.03}"
export MATRIX_LR="${MATRIX_LR:-0.02}"
export SCALAR_LR="${SCALAR_LR:-0.02}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.7}"
export LOWBIT_BITS="${LOWBIT_BITS:-6}"
export LOWBIT_NAME_PATTERNS="${LOWBIT_NAME_PATTERNS:-.mlp.,.attn.c_q.,.attn.c_v.,.attn.proj.}"
export INT8_KEEP_FLOAT_NAME_PATTERNS="${INT8_KEEP_FLOAT_NAME_PATTERNS:-tok_emb.weight,blocks.7.attn.c_k.weight,blocks.8.attn.c_k.weight}"
export INT8_GROUP_OVERRIDES="${INT8_GROUP_OVERRIDES:-.attn.c_k.:64}"
export SERIAL_COMPRESSOR="${SERIAL_COMPRESSOR:-zstd}"

exec "$ROOT_DIR/scripts/run_dense_calibration.sh"
