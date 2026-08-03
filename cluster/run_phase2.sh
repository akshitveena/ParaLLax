#!/usr/bin/env bash
# run_phase2.sh — 2a + 2b + 2c on a single A100, sequentially. No SLURM needed.
# Resumable: any seed whose result JSON already exists is skipped, so a dropped
# session or a killed job restarts where it stopped instead of at seed 0.
set -euo pipefail
RIDAE_ROOT="${RIDAE_ROOT:-$PWD}"; cd "$RIDAE_ROOT"
export HF_HOME="${HF_HOME:-$RIDAE_ROOT/.hf}"

# Auto-activate the venv if it exists and isn't active. Running these experiments under the
# system python is a silent failure mode: it may lack sentence-transformers, or worse, import
# a different torch than the one setup_venv.sh verified against the GPU.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$RIDAE_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$RIDAE_ROOT/.venv/bin/activate"
  echo "[env] activated $RIDAE_ROOT/.venv"
fi
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "!! torch cannot see the GPU — run cluster/setup_venv.sh first"; exit 1; }
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} TOKENIZERS_PARALLELISM=false
SEEDS="${SEEDS:-0 1 2 3 4}"
mkdir -p logs experiments/results_e2e

run_variant () {  # $1=tag  $2=heads  $3=ckpt_root
  for S in $SEEDS; do
    J="experiments/results_e2e/$1_seed${S}.json"
    if [ -f "$J" ]; then echo "== $1 seed $S already done, skipping =="; continue; fi
    echo "== $1 seed $S =="
    python main/train_sdae_e2e.py --cache data/step_cache.pt --ckpt_dir "$3/seed${S}" \
        --heads "$2" --seed "$S" --epochs 12 --device cuda 2>&1 | tee "logs/$1_seed${S}.log"
    python experiments/eval_e2e.py --ckpt "$3/seed${S}/sdae_e2e_best.pt" --seed "$S" \
        --tag "$1" --device cuda --out experiments/results_e2e
  done
}

echo "########## Phase 2a — e2e full (denoise + PRM + chain) ##########"
run_variant F_e2e prm_chain checkpoints/e2e_multiseed

echo "########## Phase 2b — recon-only (selection switches to val L_denoise) ##########"
run_variant R_e2e none checkpoints/e2e_recononly

echo "########## Phase 2a/2b summary ##########"
python experiments/eval_e2e.py --aggregate --out experiments/results_e2e

echo "########## Phase 2c — external PRM ##########"
if [ ! -f experiments/results_prm/scores.json ]; then
  python experiments/prm_external.py score --out experiments/results_prm/scores.json \
      --device auto --dtype bf16
fi
python experiments/prm_external.py analyze --scores experiments/results_prm/scores.json

echo
echo "ALL DONE. Pull back: experiments/results_e2e/ and experiments/results_prm/"
