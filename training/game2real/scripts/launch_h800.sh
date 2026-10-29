#!/usr/bin/env bash
# Run from this folder; all remaining arguments are forwarded to train.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${NPROC_PER_NODE:=1}"
: "${OMP_NUM_THREADS:=8}"
export OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
  -m gta2real.train --config configs/quality_h800.yaml "$@"
