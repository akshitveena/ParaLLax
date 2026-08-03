#!/usr/bin/env bash
#SBATCH --job-name=ridae_2a
#SBATCH --array=0-4                  # one task per seed -> all 5 seeds run in PARALLEL
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/2a_seed%a.out
# ---- EDIT for your cluster: partition/account/module names ----
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
set -euo pipefail
cd "$RIDAE_ROOT"
mkdir -p logs
source activate ridae
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=$SLURM_ARRAY_TASK_ID
CK="checkpoints/e2e_multiseed/seed${S}"

echo "== 2a seed $S on $(nvidia-smi --query-gpu=name --format=csv,noheader) =="
python main/train_sdae_e2e.py --cache data/step_cache.pt --ckpt_dir "$CK" \
    --heads prm_chain --seed "$S" --epochs 12 --device cuda

python experiments/eval_e2e.py --ckpt "$CK/sdae_e2e_best.pt" --seed "$S" \
    --tag F_e2e --device cuda --out experiments/results_e2e
