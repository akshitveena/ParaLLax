#!/usr/bin/env bash
# setup_venv.sh — build a working GPU venv, torch-first and version-pinned.
#
# Three failures taught this script its shape:
#   1. --system-site-packages + fresh transformers -> new transformers imported the OLD system
#      torch:  ImportError: cannot import name 'TransformGetItemToIndex'.
#   2. Fresh latest torch -> "NVIDIA driver too old (found 12060)". The box's driver caps at
#      CUDA 12.6, so the wheel must be built for <= that. We now DETECT the driver and pick the
#      matching PyTorch index instead of guessing.
#   3. venv torch + leaked system torchvision -> "operator torchvision::nms does not exist".
#      Fixed twice over: an isolated venv (no system leakage) and we never install torchvision
#      at all -- this is a text-only project, and transformers skips it when it is absent.
#
# Order matters: install torch, verify CUDA, FREEZE it into a constraints file, then install
# everything else with -c so pip can never silently move torch underneath us.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU / driver =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || { echo "!! no nvidia-smi — this box has no usable GPU"; exit 1; }

CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
echo "  driver supports CUDA ${CUDA_VER:-unknown}"
case "$CUDA_VER" in
  13.*|12.8|12.9)   TAG=cu128 ;;
  12.6|12.7)        TAG=cu126 ;;
  12.4|12.5)        TAG=cu124 ;;
  12.1|12.2|12.3)   TAG=cu121 ;;
  11.*)             TAG=cu118 ;;
  *) echo "  !! unrecognised CUDA '$CUDA_VER' — defaulting to cu126"; TAG=cu126 ;;
esac
echo "  -> installing torch wheels built for $TAG"

echo
echo "== clean isolated venv (no system site-packages -> no torchvision leakage) =="
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "  !! deactivate your current venv first, then re-run"; exit 1
fi
rm -rf .venv
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip setuptools wheel

echo "== step 1/3: torch ONLY, matched to the driver (no torchvision on purpose) =="
pip install -q torch --index-url "https://download.pytorch.org/whl/$TAG"

echo "== step 2/3: verify CUDA before anything else is installed =="
python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print(f"  !! torch {torch.__version__} cannot init CUDA on this driver.")
    print("     Re-run after editing TAG to the next LOWER cuXXX in this script.")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"  OK  torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
      f"| {p.total_memory/1e9:.0f} GB | bf16={torch.cuda.is_bf16_supported()}")
x = torch.randn(4096, 4096, device="cuda"); (x @ x).sum().item()
print("  OK  real matmul on device")
PY

echo "== step 3/3: freeze torch, then install the rest against that pin =="
pip freeze | grep -iE '^torch==' > cluster/constraints.txt
echo "  pinned: $(cat cluster/constraints.txt)"
pip install -q -c cluster/constraints.txt -r cluster/requirements-gpu.txt

echo
echo "== VERIFY: pip did not move torch =="
python - <<'PY'
import torch, sys
assert torch.cuda.is_available(), "!! torch lost CUDA — a dependency moved it despite the pin"
print(f"  torch {torch.__version__} still CUDA-capable")
PY

echo "== VERIFY: the transformers <-> torch import chain (this is what broke before) =="
python - <<'PY'
import torch, transformers, sentence_transformers as st
print(f"  torch {torch.__version__} | transformers {transformers.__version__} | ST {st.__version__}")
from transformers import AutoModel, PreTrainedModel          # noqa: F401
from sentence_transformers import SentenceTransformer        # noqa: F401
print("  OK  imports clean")
PY

echo "== VERIFY: corpus =="
python - <<'PY'
import torch, pathlib
recs = torch.load("data/step_cache.pt", weights_only=False)
cands = pathlib.Path("data/processed_pb/candidates.jsonl").read_text().splitlines()
assert "steps_text" in recs[0], "cache lacks steps_text — e2e training cannot run"
print(f"  {len(recs)} records | {len(cands)} candidates | "
      f"{sum(r['chain']=='B' for r in recs)}B / {sum(r['chain']=='A' for r in recs)}A")
PY

echo "== VERIFY: project modules =="
python - <<'PY'
import sys; sys.path[:0] = ["main", "experiments"]
import sdae_prm, ridae, train_sdae_e2e, damage              # noqa: F401
print("  OK  project modules import")
PY

python -m ipykernel install --user --name ridae --display-name "RiDAE (venv)" >/dev/null 2>&1 \
  && echo "  JupyterLab kernel 'RiDAE (venv)' registered"

echo
echo "ALL CHECKS PASSED. Next:"
echo "  source .venv/bin/activate"
echo "  hf download sentence-transformers/all-MiniLM-L6-v2"
echo "  apt-get install -y tmux && tmux new -s ridae     # or use nohup"
echo "  bash cluster/run_phase2.sh"
