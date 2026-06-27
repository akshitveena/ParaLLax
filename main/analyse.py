"""
analyse.py — post-training analysis + the Roadmap's eight figures.

Figures produced (saved to outputs/):
  Fig 1  type-B rate vs dataset difficulty (bar)          fig1_typeb_by_dataset.png
  Fig 2  baseline UMAP (made by experiments/baseline_umap.py)
  Fig 3  RiDAE z-UMAP coloured by type                    ridae_umap_by_type.png
  Fig 4  RiDAE z-UMAP coloured by subject                 ridae_umap_by_subject.png
  Fig 5  64 z-dimension probe accuracies (bar)            ridae_dimension_probe.png
  Fig 6  interpolation trajectory z_B -> z_A (20 pairs)   fig6_interpolation_trajectory.png
  Fig 7  ||z_thinking - z_response|| on ET candidates     fig7_thinking_response_gap.png
  Fig 8  baseline -> v1 -> v2 -> v3 probe ablation        fig8_ablation.png

Fig 7 is the headline contribution: the first *geometric* measure of LLM
reasoning unfaithfulness, computed directly in the trained latent space.

Usage:
    python main/analyse.py --checkpoint checkpoints/ridae_best.pt --version v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_pipeline import load_candidates, load_contrastive_pairs
from ridae import RiDAE

TYPE_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#7f7f7f", "D": "#bcbcbc"}
# ProcessBench reference Type-B rates (schema Section 1.2) for the Fig-1 overlay.
PROCESSBENCH_REF = {"GSM8K": 0.04, "MATH": 0.19, "OlympiadBench": 0.32, "OmniMath": 0.52}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
def encode_corpus(model: RiDAE, candidates):
    texts = [c.full_text for c in candidates]
    labels = np.array([c.candidate_type for c in candidates])
    return model.encode(texts), labels


def run_umap(embeddings, n_neighbors=15, min_dist=0.1):
    import umap
    reducer = umap.UMAP(n_neighbors=min(n_neighbors, max(2, len(embeddings) - 1)),
                        min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_umap(embeddings_2d, labels, title, output_path, color_map=None):
    plt = _plt()
    color_map = color_map or TYPE_COLORS
    fig, ax = plt.subplots(figsize=(8, 7))
    for lab in sorted(set(labels)):
        m = labels == lab
        ax.scatter(embeddings_2d[m, 0], embeddings_2d[m, 1], s=14, alpha=0.7,
                   label=str(lab), c=color_map.get(str(lab)) if color_map else None)
    ax.set_title(title); ax.legend(title="label", markerscale=1.5)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"[analyse] saved {output_path}")


def _ab_mask(labels):
    return np.isin(labels, ["A", "B"])


def linear_probe(embeddings, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    m = _ab_mask(labels)
    X, y = embeddings[m], labels[m]
    if len(set(y)) < 2 or len(y) < 8:
        return {"error": "not enough A/B samples", "n": int(len(y))}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {"accuracy": float(accuracy_score(yte, pred)),
            "f1_B": float(f1_score(yte, pred, pos_label="B")),
            "f1_macro": float(f1_score(yte, pred, average="macro")),
            "n_test": int(len(yte))}


def dimension_probe(embeddings, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    m = _ab_mask(labels)
    X, y = embeddings[m], labels[m]
    if len(set(y)) < 2 or len(y) < 8:
        return []
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    accs = []
    for d in range(X.shape[1]):
        clf = LogisticRegression(max_iter=1000).fit(Xtr[:, [d]], ytr)
        accs.append(float(accuracy_score(yte, clf.predict(Xte[:, [d]]))))
    return accs


def plot_dimension_probe(accs, output_path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(accs)), accs, color="#5B4FD4")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xlabel("z dimension"); ax.set_ylabel("A-vs-B accuracy")
    ax.set_title("Fig 5 — per-dimension type-probe accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"[analyse] saved {output_path}")


# --------------------------------------------------------------------------- #
# Figure 1 — Type-B rate vs dataset difficulty
# --------------------------------------------------------------------------- #
def plot_type_b_by_dataset(candidates, output_path):
    from collections import defaultdict
    plt = _plt()
    correct = defaultdict(int); typeb = defaultdict(int)
    for c in candidates:
        if c.answer_correct:
            correct[c.dataset] += 1
            if c.candidate_type == "B":
                typeb[c.dataset] += 1
    datasets = sorted(correct, key=lambda d: PROCESSBENCH_REF.get(d, 0))
    measured = [typeb[d] / correct[d] if correct[d] else 0 for d in datasets]
    ref = [PROCESSBENCH_REF.get(d) for d in datasets]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(datasets)); w = 0.38
    ax.bar(x - w/2, measured, w, label="measured (this corpus)", color="#ff7f0e")
    if any(r is not None for r in ref):
        ax.bar(x + w/2, [r or 0 for r in ref], w, label="ProcessBench ref", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("Type-B rate (of correct answers)")
    ax.set_title("Fig 1 — Type-B rate vs dataset difficulty"); ax.legend()
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"[analyse] saved {output_path}")
    return {d: measured[i] for i, d in enumerate(datasets)}


# --------------------------------------------------------------------------- #
# Figure 7 — ||z_thinking - z_response|| on extended-thinking candidates
# --------------------------------------------------------------------------- #
def _gap_stats(model, cands):
    """||z_thinking - z_response|| per candidate + separability AUC."""
    z_think = model.encode([c.thinking_text for c in cands])
    z_resp = model.encode([c.response_text for c in cands])
    gaps = np.linalg.norm(z_think - z_resp, axis=1)
    is_b = np.array([c.candidate_type == "B" for c in cands])
    out = {"n": int(len(cands)), "mean_gap_typeB": float(gaps[is_b].mean()) if is_b.any() else None,
           "mean_gap_nonB": float(gaps[~is_b].mean()) if (~is_b).any() else None}
    if is_b.any() and (~is_b).any():
        try:
            from sklearn.metrics import roc_auc_score
            out["gap_auc_predicting_B"] = float(roc_auc_score(is_b, gaps))
        except Exception:
            pass
    return gaps, is_b, out


def thinking_response_gap(model, candidates, output_path):
    """Fig 7 — the headline geometric unfaithfulness measure.

    STRONGEST claim only on thinking_source == 'claude_api_block' (architecturally
    separate thinking). QwQ 'inline_think_tags' supports a WEAKER claim (approach
    divergence is textual, not architecturally proven) and is reported separately.
    """
    have = lambda c: c.thinking_text and c.response_text
    # Honesty ladder by claim-strength. gpt_oss_cot_block is cleanly API-separated but
    # still SINGLE-PASS, so it sits in the parsed_reasoning tier — NOT the Claude tier.
    TIER_SOURCES = {
        "claude_api_block": ["claude_api_block"],
        "parsed_reasoning": ["parsed_reasoning", "gpt_oss_cot_block"],
        "inline_think_tags": ["inline_think_tags"],
    }
    tiers = {tier: [c for c in candidates if c.thinking_source in srcs and have(c)]
             for tier, srcs in TIER_SOURCES.items()}

    result = {}
    plt = _plt()
    # Primary = the strongest tier that has enough data.
    primary, label = None, None
    for t in ("claude_api_block", "parsed_reasoning", "inline_think_tags"):
        if len(tiers[t]) >= 2:
            primary, label = tiers[t], t
            break
    if primary is None:
        print("[analyse] Fig 7 skipped — need >=2 candidates with thinking + response")
        return {"error": "insufficient thinking candidates"}
    if label != "claude_api_block":
        label += " (WEAKER claim — not architecturally separated)"

    gaps, is_b, stats = _gap_stats(model, primary)
    result["primary_source"] = label
    # Report every tier that has data, so all three claims are visible.
    for t in ("claude_api_block", "parsed_reasoning", "inline_think_tags"):
        if len(tiers[t]) >= 2:
            result[t] = _gap_stats(model, tiers[t])[2]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(gaps.min(), gaps.max() + 1e-9, 20)
    if (~is_b).any():
        ax.hist(gaps[~is_b], bins=bins, alpha=0.6, label="A / faithful", color="#1f77b4")
    if is_b.any():
        ax.hist(gaps[is_b], bins=bins, alpha=0.6, label="B (thinking-gap)", color="#ff7f0e")
    ax.set_xlabel("||z_thinking - z_response||  (latent-space gap)")
    ax.set_ylabel("count")
    ax.set_title(f"Fig 7 — thinking-vs-response gap [{label}]")
    ax.legend()
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"[analyse] saved {output_path}")
    print(f"[analyse] Fig 7 [{label}]: mean gap B={stats['mean_gap_typeB']} "
          f"nonB={stats['mean_gap_nonB']} AUC={stats.get('gap_auc_predicting_B')}")
    return result


# --------------------------------------------------------------------------- #
# Figure 6 — interpolation trajectory over many pairs
# --------------------------------------------------------------------------- #
def interpolation_experiment(model, type_a, type_b, corpus_candidates, n_steps=9):
    import torch, torch.nn.functional as F
    z_a = model.encode([type_a.full_text])[0]
    z_b = model.encode([type_b.full_text])[0]
    with torch.no_grad():
        corpus_emb = model._encode_with_grad([c.full_text for c in corpus_candidates]).cpu()
    out = []
    for i in range(n_steps):
        t = round((i + 1) / (n_steps + 1), 3)
        z_t = (1 - t) * z_b + t * z_a
        with torch.no_grad():
            zt = torch.tensor(z_t, dtype=torch.float32, device=model.device).unsqueeze(0)
            recon = F.normalize(model.decoder(zt).cpu(), dim=1)
            sims = (corpus_emb @ recon.t()).squeeze(1)
            j = int(sims.argmax())
        nn_c = corpus_candidates[j]
        out.append({"t": t, "nearest_type": nn_c.candidate_type,
                    "nearest_confidence": nn_c.type_confidence,
                    "nearest_group": nn_c.contrastive_group, "cosine": float(sims[j])})
    return out


def plot_interpolation_trajectory(all_traj, output_path):
    """all_traj: list of per-pair trajectories. Plot fraction-nearest-is-A vs t."""
    plt = _plt()
    if not all_traj:
        return
    ts = [r["t"] for r in all_traj[0]]
    frac_a = []
    for k in range(len(ts)):
        vals = [traj[k]["nearest_type"] == "A" for traj in all_traj]
        frac_a.append(sum(vals) / len(vals))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0] + ts + [1], [0] + frac_a + [1], "-o", color="#5B4FD4")
    ax.set_xlabel("interpolation t   (z_B -> z_A)")
    ax.set_ylabel("fraction of pairs whose nearest neighbour is Type A")
    ax.set_title(f"Fig 6 — interpolation trajectory ({len(all_traj)} pairs)")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"[analyse] saved {output_path}")


# --------------------------------------------------------------------------- #
# Figure 8 — ablation across model versions (accumulates over phases)
# --------------------------------------------------------------------------- #
def log_ablation(output_dir, version, accuracy, f1_b=None):
    path = Path(output_dir) / "ablation_log.json"
    log = json.loads(path.read_text()) if path.exists() else {}
    log[version] = {"accuracy": accuracy, "f1_B": f1_b}
    path.write_text(json.dumps(log, indent=2))
    return log


def plot_ablation(output_dir):
    plt = _plt()
    path = Path(output_dir) / "ablation_log.json"
    if not path.exists():
        return
    log = json.loads(path.read_text())
    order = ["baseline", "v1", "v2", "v3", "current"]
    versions = [v for v in order if v in log] + [v for v in log if v not in order]
    accs = [log[v]["accuracy"] for v in versions]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(versions, accs, color=["#bcbcbc"] + ["#5B4FD4"] * (len(versions) - 1))
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_ylabel("A-vs-B linear-probe accuracy")
    ax.set_title("Fig 8 — probe accuracy across versions"); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout(); fig.savefig(Path(output_dir) / "fig8_ablation.png", dpi=150)
    plt.close(fig)
    print(f"[analyse] saved {Path(output_dir)/'fig8_ablation.png'}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ridae_best.pt")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--version", default="current", help="label for the Fig-8 ablation log")
    ap.add_argument("--max_interp_pairs", type=int, default=20)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = RiDAE.load(args.checkpoint, device=args.device)
    candidates = load_candidates(data_dir / "candidates.jsonl")
    print(f"[analyse] loaded {len(candidates)} candidates")

    emb, labels = encode_corpus(model, candidates)

    # Fig 1
    fig1 = plot_type_b_by_dataset(candidates, out_dir / "fig1_typeb_by_dataset.png")

    # Fig 3 / 4 (UMAP)
    emb2d = run_umap(emb)
    plot_umap(emb2d, labels, "Fig 3 — RiDAE z-space by type", out_dir / "ridae_umap_by_type.png")
    subjects = np.array([c.subject or "unknown" for c in candidates])
    if len(set(subjects)) > 1:
        plot_umap(emb2d, subjects, "Fig 4 — RiDAE z-space by subject",
                  out_dir / "ridae_umap_by_subject.png", color_map={})

    # Fig 5 + linear probe
    lp = linear_probe(emb, labels)
    print(f"[analyse] linear probe: {lp}")
    dp = dimension_probe(emb, labels)
    if dp:
        plot_dimension_probe(dp, out_dir / "ridae_dimension_probe.png")
        top = sorted(range(len(dp)), key=lambda i: dp[i], reverse=True)[:5]
        print(f"[analyse] top type dims: {[(d, round(dp[d],3)) for d in top]}")

    # Fig 7 — the headline gap
    fig7 = thinking_response_gap(model, candidates, out_dir / "fig7_thinking_response_gap.png")

    # Fig 6 — interpolation over up to N pairs
    interp_all = []
    pairs_path = data_dir / "contrastive_pairs.json"
    if pairs_path.exists():
        pairs = load_contrastive_pairs(pairs_path)[:args.max_interp_pairs]
        for p in pairs:
            interp_all.append(interpolation_experiment(model, p.type_a, p.type_b, candidates))
        if interp_all:
            plot_interpolation_trajectory(interp_all, out_dir / "fig6_interpolation_trajectory.png")

    # Fig 8 — ablation log + plot
    if "error" not in lp:
        log_ablation(out_dir, args.version, lp["accuracy"], lp.get("f1_B"))
        plot_ablation(out_dir)

    with (out_dir / "analysis_results.json").open("w") as fh:
        json.dump({"linear_probe": lp, "dimension_probe": dp, "fig1_typeb": fig1,
                   "fig7_gap": fig7, "interpolation": interp_all}, fh, indent=2)
    print(f"[analyse] wrote {out_dir/'analysis_results.json'}")


if __name__ == "__main__":
    main()
