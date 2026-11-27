#!/bin/bash
# Phase 0 go/no-go for the flagship (Qwen3.8-27B) on the school cluster.
# Adjust the #SBATCH lines to your cluster (partition, account, GPU type) before submitting:
#   sbatch scripts/cluster/go_no_go_27b.sbatch
#SBATCH --job-name=faithqs-go-no-go-27b
#SBATCH --gres=gpu:1                 # one 80 GB GPU (A100/H100)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=reports/go_no_go/slurm-%j.out

set -euo pipefail
export FAITHQS_SCRATCH="${FAITHQS_SCRATCH:-${SCRATCH:-$HOME/scratch}}"
export HF_HOME="${HF_HOME:-$FAITHQS_SCRATCH/hf}"
mkdir -p reports/go_no_go

uv sync --extra train
nvidia-smi --query-gpu=name,memory.total --format=csv

# 1) Load + generate (bf16, text-only) : does the 27B fit for inference?
uv run python scripts/go_no_go/smoke_generate.py --env cluster --model qwen38_27b || true

# 2) GRPO without vLLM, bf16 weights + LoRA (HF generation, slow but simplest)
uv run python scripts/go_no_go/grpo_smoke.py --env cluster --model qwen38_27b --steps 3 \
  --out reports/go_no_go/grpo_qwen38_27b_cluster_bf16.json || true

# 3) Fallback: 4-bit base weights (QLoRA). Needs bitsandbytes: uv pip install bitsandbytes
# uv run python scripts/go_no_go/grpo_smoke.py --env cluster --model qwen38_27b --steps 3 \
#   --load-in-4bit --out reports/go_no_go/grpo_qwen38_27b_cluster_4bit.json
