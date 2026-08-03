#!/usr/bin/env bash
#SBATCH --job-name=ridae_2b
#SBATCH --array=0-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/2b_seed%a.out
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
set -euo pipefail
cd "$RIDAE_ROOT"
mkdir -p logs
source activate ridae
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=$SLURM_ARRAY_TASK_ID
CK="checkpoints/e2e_recononly/seed${S}"

# Variant R-e2e: recon-only. --heads none also switches model selection to val L_denoise,
# because with no chain loss, selecting on chain_f1 would be selecting on pure noise.
echo "== 2b seed $S (recon-only) =="
python main/train_sdae_e2e.py --cache data/step_cache.pt --ckpt_dir "$CK" \
    --heads none --seed "$S" --epochs 12 --device cuda

python experiments/eval_e2e.py --ckpt "$CK/sdae_e2e_best.pt" --seed "$S" \
    --tag R_e2e --device cuda --out experiments/results_e2e
