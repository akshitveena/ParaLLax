#!/usr/bin/env bash
# setup_venv.sh — create a venv on the GPU box and verify everything before you burn GPU time.
#
# The one decision that matters here: whether to reuse the system torch. GPU containers ship a
# torch built against the exact driver they run. Installing a fresh wheel into an isolated venv
# is how you end up with torch.cuda.is_available() == False on a perfectly good A100. So: if a
# CUDA-capable torch already exists, we inherit it via --system-site-packages and never touch
# it. Only if there isn't one do we install torch ourselves.
#
#   bash cluster/setup_venv.sh && source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || { echo "!! no nvidia-smi — this box has no usable GPU"; exit 1; }

echo
echo "== probing the system python for a CUDA-capable torch =="
SYS_TORCH=0
if python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  SYS_TORCH=1
  python3 -c "import torch; print(f'  found torch {torch.__version__} (cuda {torch.version.cuda}) — will REUSE it')"
else
  echo "  none usable — the venv will install its own torch"
fi

echo
echo "== creating .venv =="
if [ "$SYS_TORCH" = "1" ]; then
  python3 -m venv .venv --system-site-packages
else
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip setuptools wheel

if [ "$SYS_TORCH" = "0" ]; then
  echo "== installing torch (linux wheels bundle CUDA 12) =="
  pip install -q torch
fi

echo "== installing project dependencies =="
pip install -q -r cluster/requirements-gpu.txt

echo
echo "== registering a JupyterLab kernel =="
python -m ipykernel install --user --name ridae --display-name "RiDAE (venv)" >/dev/null
echo "  kernel 'RiDAE (venv)' registered — pick it from the notebook kernel menu"

echo
echo "== VERIFY: gpu =="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU — do NOT start training"
p = torch.cuda.get_device_properties(0)
print(f"  {torch.cuda.get_device_name(0)} | {p.total_memory/1e9:.0f} GB | "
      f"torch {torch.__version__} | bf16={torch.cuda.is_bf16_supported()}")
x = torch.randn(2048, 2048, device="cuda"); (x @ x).sum().item()
print("  matmul on device: OK")
PY

echo "== VERIFY: corpus (came through git, not rsync) =="
python - <<'PY'
import torch, pathlib
recs = torch.load("data/step_cache.pt", weights_only=False)
cands = pathlib.Path("data/processed_pb/candidates.jsonl").read_text().splitlines()
assert "steps_text" in recs[0], "cache lacks steps_text — e2e training cannot run"
print(f"  {len(recs)} records | {len(cands)} candidates | "
      f"{sum(r['chain']=='B' for r in recs)}B / {sum(r['chain']=='A' for r in recs)}A")
PY

echo "== VERIFY: imports the experiments actually use =="
python - <<'PY'
import sys; sys.path[:0] = ["main", "experiments"]
import sdae_prm, ridae, train_sdae_e2e, damage          # noqa: F401
print("  project modules import cleanly")
PY

echo
echo "setup OK. Next:"
echo "  source .venv/bin/activate"
echo "  huggingface-cli download sentence-transformers/all-MiniLM-L6-v2"
echo "  huggingface-cli download peiyi9979/math-shepherd-mistral-7b-prm   # 2c only, ~14-29 GB"
echo "  tmux new -s ridae            # so a closed browser tab doesn't kill a 3-hour run"
echo "  bash cluster/run_phase2.sh"
