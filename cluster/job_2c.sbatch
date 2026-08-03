#!/usr/bin/env bash
#SBATCH --job-name=ridae_2c
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/2c_prm.out
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
set -euo pipefail
cd "$RIDAE_ROOT"
mkdir -p logs
source activate ridae
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# GPU pass: score once, dump per-step scores. All statistics are re-derivable on CPU later.
python experiments/prm_external.py score --out experiments/results_prm/scores.json --device auto
# Run the sanity gate immediately so a bad scoring format fails HERE, not in the paper.
python experiments/prm_external.py analyze --scores experiments/results_prm/scores.json
