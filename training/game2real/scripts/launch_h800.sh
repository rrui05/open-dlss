#!/usr/bin/env bash
# Default: train the full NR 310.8.0 graph from scratch using continuous BF16 math.
# This does not load NVIDIA NR weights or reproduce FP8 deployment bit patterns.
# GAME2REAL_CONFIG selects another explicit experiment; trailing arguments go to train.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${NPROC_PER_NODE:=1}"
: "${OMP_NUM_THREADS:=8}"
: "${GAME2REAL_CONFIG:=configs/quality_h800.yaml}"
export OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
  -m gta2real.train --config "$GAME2REAL_CONFIG" "$@"
