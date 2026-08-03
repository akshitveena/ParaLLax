#!/usr/bin/env bash
# setup_a100.sh — one-shot setup on ANY box with an A100 (interactive node, rented instance,
# or a cluster login node). No SLURM required. Run once, then cluster/run_phase2.sh.
set -euo pipefail

RIDAE_ROOT="${RIDAE_ROOT:-$PWD}"
export HF_HOME="${HF_HOME:-$RIDAE_ROOT/.hf}"
cd "$RIDAE_ROOT"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "no nvidia-smi — is this actually a GPU box?"; exit 1; }

echo "== env =="
if command -v conda >/dev/null 2>&1; then
  conda env list | grep -q '^ridae ' || conda create -y -n ridae python=3.11
  # shellcheck disable=SC1091
  source activate ridae
else
  python3 -m venv .venv && source .venv/bin/activate
fi
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers sentence-transformers scikit-learn numpy accelerate safetensors

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible to torch — check driver/toolkit match"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB | bf16={torch.cuda.is_bf16_supported()}")
PY

echo "== data =="
python - <<'PY'
import os, torch, pathlib
root = pathlib.Path(os.environ.get("RIDAE_ROOT", "."))
recs = torch.load(root / "data/step_cache.pt", weights_only=False)
cands = (root / "data/processed_pb/candidates.jsonl").read_text().splitlines()
assert "steps_text" in recs[0], "cache lacks steps_text — e2e training cannot run"
print(f"OK: {len(recs)} records, {len(cands)} candidates, "
      f"{sum(r['chain']=='B' for r in recs)}B / {sum(r['chain']=='A' for r in recs)}A")
PY

echo "== staging model weights into HF_HOME=$HF_HOME =="
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --quiet
echo "  (2c only) staging Math-Shepherd 7B — this is the slow one, ~14-29 GB"
huggingface-cli download peiyi9979/math-shepherd-mistral-7b-prm --quiet

echo
echo "setup complete. Next:"
echo "  bash cluster/run_phase2.sh          # 2a + 2b + 2c sequentially on one GPU"
echo "  # or, on a SLURM cluster, use the array jobs instead:"
echo "  sbatch cluster/job_2a.sbatch && sbatch cluster/job_2b.sbatch && sbatch cluster/job_2c.sbatch"
