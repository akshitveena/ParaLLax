"""
train_decoder.py — Stage 0 of Goal B: can a decoder generate reasoning TEXT from OUR
step-code representation?

Freezes the e2e SDAE, takes its step-code sequence (the representation you captured),
projects it into a pretrained T5-small decoder, and trains T5 to RECONSTRUCT the reasoning
text conditioned on those step-codes. Feasibility GATE: coherent reconstruction -> generation
(Stage 1) is viable; mush -> the latent is too lossy to generate from (honest stop).

Prints sample (original vs reconstructed) each epoch so you can SEE it working.

    python main/train_decoder.py --cache data/step_cache.pt \
        --sdae checkpoints/checkpoints_sdae_e2e/sdae_e2e_best.pt --epochs 6 --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdae_prm import StepSDAE_PRM
from ridae import RiDAE
from train_sdae_e2e import split_recs, encode_batch


class Decoder(nn.Module):
    """Your step-codes -> projection -> T5 decoder -> reasoning text."""
    def __init__(self, code_dim=256, t5_name="t5-small"):
        super().__init__()
        from transformers import T5ForConditionalGeneration, T5TokenizerFast
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.tok = T5TokenizerFast.from_pretrained(t5_name)
        self.proj = nn.Linear(code_dim, self.t5.config.d_model)

    def _eo(self, codes):
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=self.proj(codes))

    def forward(self, codes, mask, labels):
        return self.t5(encoder_outputs=self._eo(codes), attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def generate(self, codes, mask, max_length=200):
        ids = self.t5.generate(encoder_outputs=self._eo(codes), attention_mask=mask,
                               max_length=max_length, num_beams=3)
        return self.tok.batch_decode(ids, skip_special_tokens=True)


def step_codes(enc, sdae, recs, dev, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(recs), bs):
            X, t, pad, SL, ch = encode_batch(enc, recs[i:i + bs], dev, rng=None)
            h = sdae.encode(X, pad)                        # (B,T,256) contextualized step-codes
            for b in range(h.size(0)):
                n = int((~pad[b]).sum().item())
                out.append(h[b, :n].cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--sdae", default="checkpoints/checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--ckpt_dir", default="checkpoints/decoder")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_tok", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    recs = torch.load(args.cache, weights_only=False)
    tr, va = split_recs(recs)
    ck = torch.load(args.sdae, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(dev); sdae.load_state_dict(ck["sdae"]); sdae.eval()
    for p in list(enc.st.parameters()) + list(sdae.parameters()):
        p.requires_grad_(False)

    print("[dec] computing frozen step-codes ...", flush=True)
    tr_codes = step_codes(enc, sdae, tr, dev)
    va_codes = step_codes(enc, sdae, va, dev)
    tr_text = ["\n".join(r["steps_text"]) for r in tr]
    va_text = ["\n".join(r["steps_text"]) for r in va]

    model = Decoder().to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    print(f"[dec] train={len(tr)} val={len(va)} | T5-small decoder on your step-codes", flush=True)

    def batch(codes, texts, idx):
        cs = [codes[i] for i in idx]
        T = max(c.size(0) for c in cs)
        C = torch.zeros(len(cs), T, cs[0].size(1))
        mask = torch.zeros(len(cs), T, dtype=torch.long)
        for j, c in enumerate(cs):
            C[j, :c.size(0)] = c; mask[j, :c.size(0)] = 1
        lab = model.tok([texts[i] for i in idx], max_length=args.max_tok, truncation=True,
                        padding=True, return_tensors="pt").input_ids
        lab[lab == model.tok.pad_token_id] = -100
        return C.to(dev), mask.to(dev), lab.to(dev)

    for ep in range(args.epochs):
        model.train()
        order = np.random.permutation(len(tr)); tot = 0.0; nb = 0
        t0 = time.time()
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            C, m, lab = batch(tr_codes, tr_text, idx)
            loss = model(C, m, lab)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        model.eval()
        C, m, _ = batch(va_codes, va_text, [0, 1])
        gen = model.generate(C, m, args.max_tok)
        print(f"[dec] ep{ep} loss={tot/nb:.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
        for k in range(2):
            print(f"   ORIG: {va_text[k][:160]!r}")
            print(f"   GEN : {gen[k][:160]!r}")
        torch.save({"proj": model.proj.state_dict(), "t5": model.t5.state_dict()},
                   Path(args.ckpt_dir) / "decoder_best.pt")

    print("[dec] done. Coherent GEN -> Stage 1 (sampling/interpolation) is viable.")


if __name__ == "__main__":
    main()
