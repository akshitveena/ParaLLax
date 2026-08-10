"""
qwen_diag.py — pinpoint the Qwen2.5-Math-PRM scoring format on transformers 5.x.

The panel adapter gave gate 0.559 / ctl f1 0.000, and load warned 'lm_head.weight UNEXPECTED'.
That signals the model is NOT returning its 2-class reward logits where we read them. This dumps
the actual output structure on ONE solution so we can see: what does the model return, what shape,
and does reading class-0 vs class-1 at <extra_0> positions align with human step labels.

    python experiments/qwen_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "main"))
HF = "Qwen/Qwen2.5-Math-PRM-7B"


def main():
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import json
    tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(HF, trust_remote_code=True)
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModel.from_pretrained(HF, config=cfg, dtype=torch.bfloat16,
                                      device_map="auto", trust_remote_code=True).eval()
    sep = tok.encode("<extra_0>")[0]
    print(f"<extra_0> id = {sep}")

    recs = torch.load(ROOT / "data/step_cache.pt", weights_only=False)
    probs = {json.loads(l)["record_id"]: json.loads(l)["problem"]
             for l in Path(ROOT, "data/processed_pb/candidates.jsonl").read_text().splitlines() if l.strip()}
    # grab a record that HAS a labelled error so we can check alignment
    r = next(x for x in recs if 1 in list(x["step_labels"]))
    steps = r["steps_text"]; labels = list(r["step_labels"])
    msgs = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": probs.get(r["id"], "")},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    print(f"output type: {type(out)}")
    if hasattr(out, "keys"):
        print(f"output keys: {list(out.keys())}")
    o0 = out[0] if isinstance(out, (tuple, list)) else getattr(out, "logits", out[0])
    print(f"out[0] shape: {tuple(o0.shape)}   (vocab size = {getattr(cfg,'vocab_size','?')})")
    mask = (ids == sep)
    print(f"n <extra_0> positions: {int(mask.sum())}   n steps: {len(steps)}   labels: {labels}")

    probm = F.softmax(o0, dim=-1)
    for cls in (0, 1):
        if o0.shape[-1] >= 2:
            sc = probm[..., cls][mask].float().cpu().tolist()
            print(f"  class-{cls} P at steps: {[round(x,3) for x in sc]}")
    print("\nRead: the class whose (1 - P) tracks the labels (1=error) is the CORRECT-prob class.")
    print("If out[0] last-dim is vocab (~150k), the reward head is elsewhere — inspect keys above.")


if __name__ == "__main__":
    main()
