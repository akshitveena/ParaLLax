# Draft: Factored architecture in the main Method (conditional on E2)

> **Do not paste until `e2_factored_architecture.py` has run.** Every `<PLACEHOLDER>` is a real number
> from that run. Which of the three variants below you keep depends on the E2 decision rule (PLAN.md §E2).
> This replaces / extends **§2 Architecture** and demotes current Appendix N to motivation.

---

## Variant 1 — use ONLY if E2 shows content ≈0.996 AND controlled f1B ≥ 0.591

**§2.x Factored detection: a content-preserving encoder with a non-competing validity readout.**

The shared-bottleneck design of Figure 4 forces one latent to serve reconstruction, per-step validity,
and chain validity at once. Appendix N shows the cost is not incidental but a gradient-competition
collapse: with the two supervised heads present, retrieval recall of the source candidate falls from
0.996 to 0.068 — the latent evicts the content it was also asked to preserve. Because a content-preserving
latent already separates Type A from Type B without supervision (AUC 0.693 vs. the supervised 0.673),
we factor the two objectives apart. A content-preserving encoder is trained with the reconstruction
objective alone; a lightweight validity readout consumes its step-codes through a pathway carrying **no
gradient into the reconstruction bottleneck**. The readout therefore cannot evict content, and the
encoder cannot be reshaped by the validity loss.

The factored model preserves content (Recall@1 `<PH_recall>` vs. 0.996 recon-only) **and** matches the
shared model's controlled signal (f1B `<PH_f1b>` vs. 0.591 end-to-end), resolving the trade-off Appendix N
identified rather than merely characterizing it. [Regime = `<PH_regime>`.]

---

## Variant 2 — use if content is preserved but validity lands ~0.69 (below e2e 0.741)

**§2.x A content-preserving variant, with a stated validity cost.**

… [motivation identical to Variant 1] … Factoring the objectives preserves content almost perfectly
(Recall@1 `<PH_recall>`) but recovers only `<PH_auc>` validity AUC, below the `0.741` the shared
end-to-end model buys by reshaping the encoder itself. This is the trade-off Appendix N predicted: a
fixed content-preserving latent does not, on its own, supply all the separability that full plasticity
creates. We present the factored model as the design of choice **when content recoverability matters**
(mining, generation, faithfulness flagging), and the shared model when peak validity discrimination is
the only goal. The two are not strictly ordered.

---

## Variant 3 — use if E2 fails both targets

**Keep Appendix N as-is.** Do not move the factored design into the main text. Add one sentence to
Appendix N's closing paragraph:

> We tested the factored design directly (Appendix `<PH_ref>`): a non-competing readout on a
> content-preserving encoder retains content (Recall@1 `<PH_recall>`) but reaches only f1B `<PH_f1b>`,
> confirming that the separability the shared model attains is created by reshaping the encoder and is
> not recoverable from a fixed latent. The trade-off is therefore intrinsic to this class of
> representation, not an artifact of joint training.

---

### Abstract / intro hook (only Variants 1–2)
Add to contributions: *"and we show the content/validity trade-off it induces is resolvable by
factoring the objectives — a content-preserving encoder with a non-competing validity readout recovers
[the controlled signal / most of it] while keeping Recall@1 `<PH_recall>`."*

### Figure
Extend Figure 4: a second panel showing the factored graph — encoder → {reconstruction decoder} and
encoder →(stop-grad / separate adapter)→ validity readout, with the gradient block drawn explicitly.

### Numbers table to fill from `e2_results.json`
| regime | Recall@1 (content) | controlled f1B | validity AUC |
|---|---|---|---|
| (A) frozen recon-only + MLP readout | `<PH>` | `<PH>` | `<PH>` |
| (B) recon encoder + isolated adapter | `<PH>` | `<PH>` | `<PH>` |
| (C) shared e2e (Variant F-e2e baseline) | ~0.007 | 0.591 | 0.741 |
