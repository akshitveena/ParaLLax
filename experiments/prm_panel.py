"""
prm_panel.py — E1: a PANEL of open PRMs under our identical confound protocol.

Upgrades 2c's single-model result ("Math-Shepherd is confound-inflated") into a claim about the
FIELD ("PRM-based Type-B evaluation is confound-inflated as a class"). Different groups, different
base models, different training data — four Qwen variants would be one data point, not four.

THE HARD PART is that every PRM scores steps differently. A wrong format silently yields noise, so
each model has its own adapter AND every adapter must clear the same SANITY GATE: its per-step
scores must predict ProcessBench HUMAN step labels above chance (AUC > 0.55). Below that, the
number is not reported — it is flagged as "format unvalidated", never quietly included.

Adapters (verify HF IDs before running; they move):
  shepherd  peiyi9979/math-shepherd-mistral-7b-prm   step tag 'ки', softmax over +/- token logits
  qwen      Qwen/Qwen2.5-Math-PRM-7B                  '<extra_0>' separator, 2-class reward head
  rlhflow   RLHFlow/Llama3.1-8B-PRM-Deepseek-Data     chat turns, P('+') vs P('-') per step
  skywork   Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B   custom repo code (highest format risk)

PRE-COMMITMENT (the paper's own thesis demands it): every model run is reported, before the
numbers are seen. A panel where 3 collapse and 1 does not is MORE credible than 3 that all
collapse — and dropping a non-collapsing model would be the exact failure mode this paper accuses
the field of. So `panel` tabulates every scores_*.json present, deflating or not.

CAVEAT logged in-table: ProcessBench is Qwen's benchmark, so `qwen` may have seen adjacent data —
its RAW number may be inflated for reasons unrelated to confounds. Reported with that flag.

Two passes (never fuse — you WILL tweak the protocol and re-running 7B forwards is the cost):
    python experiments/prm_panel.py score --model qwen     --out experiments/results_prm   # GPU
    python experiments/prm_panel.py score --model rlhflow  --out experiments/results_prm   # GPU
    python experiments/prm_panel.py panel  --dir experiments/results_prm                    # CPU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))

REGISTRY = {
    "shepherd": dict(hf="peiyi9979/math-shepherd-mistral-7b-prm", kind="shepherd", qwen_caveat=False),
    "qwen":     dict(hf="Qwen/Qwen2.5-Math-PRM-7B",               kind="qwen",     qwen_caveat=True),
    "rlhflow":  dict(hf="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",  kind="rlhflow",  qwen_caveat=False),
    "skywork":  dict(hf="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B", kind="skywork", qwen_caveat=True),
}


# --------------------------------------------------------------------------- adapters
class ShepherdAdapter:
    """peiyi9979: append step tag 'ки'; P(correct)=softmax over '+'/'-' logits at each tag."""
    def __init__(self, hf, dtype, device):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            self.tok = AutoTokenizer.from_pretrained(hf)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(hf, use_fast=False)
        self.cand = self.tok.encode("+ -")[1:]                      # [id('+'), id('-')]
        self.tag = self.tok.encode("ки")[-1]
        self.model = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype, device_map=device).eval()
        self.max_len = 2048

    @torch.no_grad()
    def score(self, problem, steps):
        text = problem + "".join(f" {s} ки\n" for s in steps)
        ids = self.tok.encode(text, return_tensors="pt", truncation=True,
                              max_length=self.max_len).to(self.model.device)
        logits = self.model(ids).logits[:, :, self.cand]
        probs = logits.softmax(-1)[:, :, 0]
        return probs[ids == self.tag].float().cpu().tolist()


class QwenAdapter:
    """Qwen2.5-Math-PRM: '<extra_0>' after each step; 2-class reward head, P(correct)=class-1."""
    def __init__(self, hf, dtype, device):
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        self.tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
        # Qwen's released modeling_qwen2_rm.py predates transformers 5.x and reads
        # config.pad_token_id, which the new config no longer auto-exposes. Inject it.
        cfg = AutoConfig.from_pretrained(hf, trust_remote_code=True)
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = (self.tok.pad_token_id if self.tok.pad_token_id is not None
                                else self.tok.eos_token_id)
        self.model = AutoModel.from_pretrained(hf, config=cfg, dtype=dtype, device_map=device,
                                               trust_remote_code=True).eval()
        self.sep = self.tok.encode("<extra_0>")[0]
        self.sys = "Please reason step by step, and put your final answer within \\boxed{}."
        self.max_len = 4096

    @torch.no_grad()
    def score(self, problem, steps):
        msgs = [{"role": "system", "content": self.sys},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}]
        text = self.tok.apply_chat_template(msgs, tokenize=False)
        ids = self.tok.encode(text, return_tensors="pt", truncation=True,
                              max_length=self.max_len).to(self.model.device)
        # use_cache=False avoids the KV-cache path, whose DynamicCache.from_legacy_cache call
        # was removed in transformers 5.x (this is a scoring forward pass — no cache needed).
        out = self.model(input_ids=ids, use_cache=False)
        logits = out[0] if isinstance(out, (tuple, list)) else out.logits  # (B, L, 2)
        mask = (ids == self.sep)
        probs = F.softmax(logits, dim=-1)[..., 1]                   # P(correct)
        return probs[mask].float().cpu().tolist()


class RLHFlowAdapter:
    """RLHFlow Llama PRM: per step, assistant answers '+'/'-'; P(correct)=P('+') at that turn."""
    def __init__(self, hf, dtype, device):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.plus = self.tok.encode("+")[-1]
        self.minus = self.tok.encode("-")[-1]
        self.model = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype, device_map=device).eval()
        self.max_len = 4096

    @torch.no_grad()
    def score(self, problem, steps):
        # One forward PER step, reading P('+') at the generation position. Unambiguous: always
        # returns exactly len(steps) scores, so nothing is skipped. The earlier single-forward
        # "count '+' tokens" version mis-counted whenever '+' appeared in the math itself, which
        # dropped ~27% of solutions NON-randomly (arithmetic-heavy -> longer -> our top confound).
        out = []
        for i in range(len(steps)):
            msgs = []
            for j in range(i + 1):
                msgs.append({"role": "user",
                             "content": (problem + " " + steps[j]) if j == 0 else steps[j]})
                if j < i:
                    msgs.append({"role": "assistant", "content": "+"})
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=self.max_len).input_ids.to(self.model.device)
            two = self.model(ids).logits[0, -1, [self.plus, self.minus]].softmax(-1)
            out.append(float(two[0]))
        return out


class SkyworkAdapter(QwenAdapter):
    """Skywork-o1-Open-PRM is Qwen-2.5 based and follows the same <extra_0> reward-head scheme;
    if its released code diverges, the sanity gate will catch it and the model is flagged, not
    silently trusted."""
    pass


ADAPTERS = {"shepherd": ShepherdAdapter, "qwen": QwenAdapter,
            "rlhflow": RLHFlowAdapter, "skywork": SkyworkAdapter}


# --------------------------------------------------------------------------- score pass (GPU)
def cmd_score(args):
    spec = REGISTRY[args.model]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    t0 = time.time()
    recs = torch.load(args.cache, weights_only=False)
    probs = {json.loads(l)["record_id"]: json.loads(l)["problem"]
             for l in Path(args.data_dir, "candidates.jsonl").read_text().splitlines() if l.strip()}
    recs = recs[:args.limit] if args.limit else recs

    print(f"[E1] loading {spec['hf']} ({spec['kind']} adapter)", flush=True)
    adapter = ADAPTERS[spec["kind"]](spec["hf"], dtype, "auto")

    out, skipped = [], 0
    for i, r in enumerate(recs):
        try:
            ss = adapter.score(probs.get(r["id"], ""), r["steps_text"])
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  [warn] {r['id']}: scorer error {type(e).__name__}: {e}", flush=True)
            continue
        if len(ss) != len(r["steps_text"]):
            skipped += 1
            if skipped <= 3:
                print(f"  [warn] {r['id']}: {len(ss)} scores vs {len(r['steps_text'])} steps — skipped",
                      flush=True)
            continue
        out.append({"id": r["id"], "chain": r["chain"], "split": r["split"],
                    "step_scores": ss, "step_labels": r["step_labels"].tolist()})
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(recs)} ({(time.time()-t0)/60:.1f}m)", flush=True)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"scores_{args.model}.json"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    p.write_text(json.dumps({"model": spec["hf"], "key": args.model, "gpu": gpu,
                             "qwen_caveat": spec["qwen_caveat"],
                             "minutes": round((time.time() - t0) / 60, 2),
                             "n": len(out), "skipped": skipped, "rows": out}, indent=2))
    print(f"[E1] {args.model}: wrote {len(out)} scored ({skipped} skipped) -> {p}")
    print(f"[E1] {(time.time()-t0)/60:.1f} min on {gpu}. Now: python experiments/prm_panel.py panel")


# --------------------------------------------------------------------------- panel table (CPU)
def analyze_one(blob, cache, data_dir, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from multiseed_ablation import build_confounds
    rows = blob["rows"]
    ids = [r["id"] for r in rows]; y = np.array([r["chain"] for r in rows])

    fs, fl = [], []
    for r in rows:
        for s, l in zip(r["step_scores"], r["step_labels"]):
            if l >= 0:
                fs.append(1.0 - s); fl.append(int(l))
    gate = roc_auc_score(fl, fs) if len(set(fl)) > 1 else float("nan")

    full = torch.load(cache, weights_only=False)
    pos = {i: k for k, i in enumerate(ids)}
    sub = sorted([r for r in full if r["id"] in pos], key=lambda r: pos[r["id"]])
    C = build_confounds(sub, data_dir)
    rng = np.random.RandomState(seed); idx = np.arange(len(rows)); rng.shuffle(idx)
    cut = int(0.8 * len(rows)); tri, vai = idx[:cut], idx[cut:]
    yb = (y[vai] == "B").astype(int)

    res = {"gate": gate, "n": len(rows), "skipped": blob.get("skipped", 0),
           "qwen_caveat": blob.get("qwen_caveat", False)}
    for name, agg in [("min", np.array([1 - min(r["step_scores"]) for r in rows])),
                      ("mean", np.array([1 - float(np.mean(r["step_scores"])) for r in rows]))]:
        def fit(v):
            clf = LogisticRegression(max_iter=2000).fit(v[tri].reshape(-1, 1), y[tri])
            return (f1_score(y[vai], clf.predict(v[vai].reshape(-1, 1)), pos_label="B"),
                    roc_auc_score(yb, v[vai]))
        raw_f1, raw_auc = fit(agg)
        beta, *_ = np.linalg.lstsq(C[tri], agg[tri], rcond=None)
        ctl_f1, ctl_auc = fit(agg - C @ beta)
        res[name] = dict(raw_f1=raw_f1, ctl_f1=ctl_f1, raw_auc=raw_auc, ctl_auc=ctl_auc)
    return res


def cmd_panel(args):
    files = sorted(Path(args.dir).glob("scores_*.json")) + \
            ([Path(args.dir, "scores.json")] if Path(args.dir, "scores.json").exists() else [])
    if not files:
        print(f"no scores_*.json in {args.dir}"); return
    print("=" * 96)
    print("  E1 — PRM PANEL under the identical confound protocol (min-aggregation)")
    print("=" * 96)
    print(f"  {'model':<34}{'gate':>6}{'raw f1':>9}{'ctl f1':>9}{'Δf1':>8}{'raw AUC':>9}{'ctl AUC':>9}  note")
    print("  " + "-" * 92)
    for f in files:
        blob = json.loads(f.read_text())
        key = blob.get("key") or ("shepherd" if f.name == "scores.json"
                                   else f.stem.replace("scores_", ""))
        n = len(blob.get("rows", []))
        if n < 20:                                    # adapter produced (almost) nothing
            print(f"  {key:<34}{'—':>6}{'—':>9}{'—':>9}{'—':>8}{'—':>9}{'—':>9}  "
                  f"ADAPTER FAILED — {n}/{blob.get('n',0)+blob.get('skipped',0)} scored, format wrong")
            continue
        r = analyze_one(blob, args.cache, args.data_dir)
        gate_ok = r["gate"] > 0.55
        note = ("GATE FAIL — format unvalidated, DO NOT REPORT" if not gate_ok
                else ("Qwen-family: ProcessBench contamination possible" if r["qwen_caveat"] else ""))
        m = r["min"]
        print(f"  {key:<34}{r['gate']:>6.3f}{m['raw_f1']:>9.3f}{m['ctl_f1']:>9.3f}"
              f"{m['ctl_f1']-m['raw_f1']:>+8.3f}{m['raw_auc']:>9.3f}{m['ctl_auc']:>9.3f}  {note}")
    print("=" * 96)
    print("  gate = step-score AUC vs HUMAN step labels; must exceed 0.55 or the number is noise.")
    print("  Δf1 = ctl − raw; large negative = confound-inflated. Reporting ALL models run (pre-committed).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--model", required=True, choices=list(REGISTRY))
    s.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    s.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    s.add_argument("--out", default=str(HERE / "results_prm"))
    s.add_argument("--dtype", default="bf16"); s.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("panel"); p.set_defaults(fn=cmd_panel)
    p.add_argument("--dir", default=str(HERE / "results_prm"))
    p.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    p.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
