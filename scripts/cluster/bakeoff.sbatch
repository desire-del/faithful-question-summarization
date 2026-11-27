#!/bin/bash
# Flagship bake-off (end of phase 1): 3 candidates in few-shot on dev, then the protocol decision.
# Prerequisites: go_no_go_27b.sbatch passed; Hub access accepted for google/medgemma-27b-text-it
# (gated: run `huggingface-cli login` once on the cluster). Adjust #SBATCH lines, then:
#   sbatch scripts/cluster/bakeoff.sbatch
#SBATCH --job-name=faithqs-bakeoff
#SBATCH --gres=gpu:1                 # one 80 GB GPU (bf16 31B fits for inference)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=reports/bakeoff/slurm-%j.out

set -euo pipefail
export FAITHQS_SCRATCH="${FAITHQS_SCRATCH:-${SCRATCH:-$HOME/scratch}}"
export HF_HOME="${HF_HOME:-$FAITHQS_SCRATCH/hf}"
mkdir -p reports/bakeoff

uv sync --extra train --extra eval
uv run faithqs data fetch
for model in qwen38_27b gemma4_31b medgemma_27b; do
  uv run python scripts/bakeoff.py generate --env cluster --model "$model"
done
uv run python scripts/bakeoff.py decide --env cluster
