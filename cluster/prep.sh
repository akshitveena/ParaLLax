#!/usr/bin/env bash
# prep.sh — run ONCE on the LOGIN node (which has internet). Compute nodes usually do not,
# so every model weight must be staged to shared storage here and read offline in the jobs.
set -euo pipefail

: "${RIDAE_ROOT:?set RIDAE_ROOT to the repo path on the cluster}"
: "${HF_HOME:?set HF_HOME to a SHARED-STORAGE cache dir (not \$HOME if it has a small quota)}"

echo "== 1/3 conda env =="
if ! conda env list | grep -q '^ridae '; then
  conda create -y -n ridae python=3.11
fi
# shellcheck disable=SC1091
source activate ridae
pip install -q torch transformers sentence-transformers scikit-learn numpy accelerate

echo "== 2/3 staging model weights into HF_HOME=$HF_HOME =="
mkdir -p "$HF_HOME"
# encoder for the e2e training (small)
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --quiet
# external PRM for 2c (~14-28 GB — this is the long one)
huggingface-cli download peiyi9979/math-shepherd-mistral-7b-prm --quiet

echo "== 3/3 verifying the corpus shipped correctly =="
python - <<'PY'
import os, torch, json, pathlib
root = pathlib.Path(os.environ["RIDAE_ROOT"])
recs = torch.load(root / "data/step_cache.pt", weights_only=False)
cands = (root / "data/processed_pb/candidates.jsonl").read_text().splitlines()
assert "steps_text" in recs[0], "cache lacks steps_text — e2e training cannot run"
print(f"OK: {len(recs)} cached records, {len(cands)} candidates, "
      f"chains={sum(r['chain']=='B' for r in recs)}B/{sum(r['chain']=='A' for r in recs)}A")
PY

echo
echo "prep complete. Now submit:"
echo "  sbatch cluster/job_2a.sbatch     # 5-seed e2e (array) -> 2a"
echo "  sbatch cluster/job_2b.sbatch     # 5-seed recon-only e2e (array) -> 2b"
echo "  sbatch cluster/job_2c.sbatch     # external PRM scoring -> 2c"
