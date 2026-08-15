"""End-to-end validation on a ground-truth synthetic corpus.

Five experiments, each falsifiable in advance:

  E1  Gate detection      — planted corruptions vs clean episodes:
                            precision / recall / F1 / AUROC (industry-standard
                            detection metrics; AUROC over a continuous
                            corruption-severity score).
  E2  Noise inflates VS   — Vendi WITH vs WITHOUT gates: ungated corpora must
                            score HIGHER (corruption masquerades as novelty).
                            This is the core first-principles claim.
  E3  Label collapse      — episodes from ONE behavior family must collapse to
                            a small effective count; a same-size RANDOM mix of
                            families must not. Bandwidth honesty bracket.
  E4  Duplicate recall@k  — planted near-duplicate pairs (same behavior, new
                            room/speed) must appear in the top-k most-similar
                            pairs. Nuisance-quotient effectiveness.
  E5  Curation curve      — greedy max-Vendi keep-list: fraction kept vs
                            diversity retained (the Track-1 deliverable).

Outputs: report/figures/*.png + report/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionvendi.curate import curation_curve, nearest_duplicates
from motionvendi.gates import gate_episode
from motionvendi.kernels import kernel_histogram_stats, median_bandwidth, rbf_kernel, validate_psd
from motionvendi.normalize import episode_to_vector
from motionvendi.synthetic import FAMILIES, make_corpus
from motionvendi.vendi import bootstrap_vendi, eigenvalue_spectrum, vendi_score

OUT = Path(__file__).resolve().parents[1] / "report"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

N_STEPS = 32
RNG = np.random.default_rng(42)


def corruption_severity(ep) -> float:
    """Continuous severity score for AUROC: worst normalized gate evidence."""
    rep = gate_episode(ep["left"], ep["right"])
    worst = 0.0
    for side_ev in rep.evidence.values():
        worst = max(
            worst,
            side_ev["finite"]["nan_frac"] / 0.05,
            side_ev["teleport"]["max_speed"] / 6.0,
            side_ev["frozen"]["frozen_frac"] / 0.9,
            side_ev["quaternion"]["bad_quat_frac"] / 0.05,
            side_ev["rotation_rate"]["max_rot_rate"] / (4 * np.pi),
        )
    return worst


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), no sklearn needed."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def vectorize(corpus) -> tuple[np.ndarray, list]:
    vecs, kept = [], []
    for ep in corpus:
        v = episode_to_vector(ep["left"], ep["right"], ep["head"], n_steps=N_STEPS)
        if np.all(np.isfinite(v)):
            vecs.append(v)
            kept.append(ep)
    return np.asarray(vecs), kept


def main() -> None:
    metrics: dict = {}
    corpus = make_corpus(n_per_family=20, n_duplicate_pairs=12, n_corrupt=24, seed=7)
    is_corrupt = np.array(["corruption" in ep for ep in corpus])
    print(f"corpus: {len(corpus)} episodes, {int(is_corrupt.sum())} corrupted (ground truth)")

    # ---------------------------------------------------------------- E1 gates
    flagged = np.array([not gate_episode(ep["left"], ep["right"]).passed for ep in corpus])
    tp = int((flagged & is_corrupt).sum())
    fp = int((flagged & ~is_corrupt).sum())
    fn = int((~flagged & is_corrupt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    severity = np.array([corruption_severity(ep) for ep in corpus])
    metrics["E1_gate_detection"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc(is_corrupt, severity),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_corrupt": int(is_corrupt.sum()),
        "n_clean": int((~is_corrupt).sum()),
    }
    print(f"E1 gates: P={precision:.3f} R={recall:.3f} F1={f1:.3f} AUROC={metrics['E1_gate_detection']['auroc']:.3f}")

    # ------------------------------------------------- E2 noise inflates Vendi
    X_all, kept_all = vectorize(corpus)  # ungated (only drops non-finite vectors)
    clean_corpus = [ep for ep in corpus if gate_episode(ep["left"], ep["right"]).passed]
    X_gated, kept_gated = vectorize(clean_corpus)
    sigma2 = median_bandwidth(X_gated)  # one bandwidth, fit on gated data, reused
    kb = lambda X: rbf_kernel(X, sigma2=sigma2)
    n_fair = min(len(X_all), len(X_gated), 100)
    vs_ungated = bootstrap_vendi(X_all, kb, sample_size=n_fair, n_boot=40, rng=RNG)
    vs_gated = bootstrap_vendi(X_gated, kb, sample_size=n_fair, n_boot=40, rng=RNG)
    metrics["E2_noise_inflation"] = {
        "vendi_ungated": vs_ungated,
        "vendi_gated": vs_gated,
        "inflation_pct": 100 * (vs_ungated["mean"] - vs_gated["mean"]) / vs_gated["mean"],
        "sample_size": n_fair,
    }
    print(
        f"E2 noise inflation: ungated VS={vs_ungated['mean']:.1f} vs gated VS={vs_gated['mean']:.1f} "
        f"(+{metrics['E2_noise_inflation']['inflation_pct']:.1f}% fake diversity from corruption)"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["gated\n(corruption removed)", "ungated\n(corruption included)"],
        [vs_gated["mean"], vs_ungated["mean"]],
        yerr=[vs_gated["std"], vs_ungated["std"]],
        color=["#2a9d8f", "#e76f51"],
        capsize=6,
    )
    ax.set_ylabel(f"Vendi score (fixed n={n_fair}, 40 bootstraps)")
    ax.set_title("Noise is maximally novel: corruption inflates measured diversity")
    fig.tight_layout()
    fig.savefig(FIG / "e2_noise_inflation.png", dpi=150)

    # ------------------------------------------------------- E3 label collapse
    fam_of = [ep["family"] for ep in kept_gated]
    target_fam = "wipe_circle"
    idx_one = [i for i, f in enumerate(fam_of) if f == target_fam]
    n_grp = len(idx_one)
    idx_rand = RNG.choice(len(X_gated), size=n_grp, replace=False)
    K_one = kb(X_gated[idx_one])
    K_rand = kb(X_gated[idx_rand])
    vs_one, vs_rand = vendi_score(K_one), vendi_score(K_rand)
    # bandwidth honesty bracket: sweep sigma2, both curves must separate
    sweep = {}
    for mult in [0.1, 0.3, 1.0, 3.0, 10.0]:
        kb_m = lambda X: rbf_kernel(X, sigma2=sigma2 * mult)
        sweep[mult] = {
            "one_family": vendi_score(kb_m(X_gated[idx_one])),
            "random_mix": vendi_score(kb_m(X_gated[idx_rand])),
        }
    metrics["E3_label_collapse"] = {
        "family": target_fam,
        "n": n_grp,
        "vendi_one_family": vs_one,
        "vendi_random_mix": vs_rand,
        "collapse_ratio": vs_one / vs_rand,
        "bandwidth_sweep": sweep,
        "psd_ok": validate_psd(K_one)[0] and validate_psd(K_rand)[0],
        "kernel_stats_one_family": kernel_histogram_stats(K_one),
    }
    print(f"E3 collapse: one-family VS={vs_one:.2f} vs random-mix VS={vs_rand:.2f} (n={n_grp} each)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        [f"one family\n({target_fam})", "random mix\n(same n)"],
        [vs_one, vs_rand],
        color=["#264653", "#e9c46a"],
    )
    axes[0].set_ylabel(f"Vendi score (n={n_grp})")
    axes[0].set_title("Label-collapse test: same task collapses, mix doesn't")
    mults = sorted(sweep)
    axes[1].plot(mults, [sweep[m]["one_family"] for m in mults], "o-", label="one family")
    axes[1].plot(mults, [sweep[m]["random_mix"] for m in mults], "s-", label="random mix")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("bandwidth multiplier (x median heuristic)")
    axes[1].set_ylabel("Vendi score")
    axes[1].set_title("Honesty bracket: separation across bandwidths")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "e3_label_collapse.png", dpi=150)

    # -------------------------------------------------- E4 duplicate recall@k
    # Standard retrieval formulation: for each planted duplicate, is its
    # partner among its top-k nearest neighbors? (A global "closest pairs"
    # list is confounded by legitimate unplanned near-dups in the corpus.)
    idx_of = {id(ep): i for i, ep in enumerate(kept_gated)}
    dup_queries = [
        (idx_of[id(ep)], idx_of[id(corpus[ep["is_duplicate_of"]])])
        for ep in kept_gated
        if "is_duplicate_of" in ep and id(corpus[ep["is_duplicate_of"]]) in idx_of
    ]
    K_full = kb(X_gated)
    sim = K_full.copy()
    np.fill_diagonal(sim, -np.inf)
    metrics["E4_duplicate_recall"] = {"n_planted_pairs": len(dup_queries)}
    for k_at in (1, 3, 5):
        hits = sum(
            int(j in np.argsort(sim[i])[::-1][:k_at]) for i, j in dup_queries
        )
        metrics["E4_duplicate_recall"][f"nn_recall@{k_at}"] = (
            hits / len(dup_queries) if dup_queries else float("nan")
        )
    top_pairs = nearest_duplicates(K_full, top_k=5)
    metrics["E4_duplicate_recall"]["top5_most_similar_pairs"] = [
        {
            "i": i,
            "j": j,
            "similarity": s,
            "same_family": kept_gated[i]["family"] == kept_gated[j]["family"],
        }
        for i, j, s in top_pairs
    ]
    print(f"E4 duplicate NN recall: { {k: v for k, v in metrics['E4_duplicate_recall'].items() if 'recall' in k} }")

    # ---------------------------------------------------- E5 curation curve
    cc = curation_curve(K_full, fractions=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    metrics["E5_curation_curve"] = {
        "full_vendi": cc["full_vendi"],
        "curve": cc["curve"],
        "n_episodes": len(K_full),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fr = [c["fraction_kept"] for c in cc["curve"]]
    rt = [c["vendi_retained_frac"] for c in cc["curve"]]
    axes[0].plot(fr, rt, "o-", color="#2a9d8f")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="random-keep baseline")
    axes[0].set_xlabel("fraction of episodes kept (greedy max-Vendi order)")
    axes[0].set_ylabel("fraction of diversity retained")
    axes[0].set_title("Curation curve: small subsets hold most of the diversity")
    axes[0].legend()
    lam = eigenvalue_spectrum(K_full)
    axes[1].semilogy(np.arange(1, min(len(lam), 60) + 1), lam[:60], ".-", color="#264653")
    axes[1].set_xlabel("eigenvalue rank")
    axes[1].set_ylabel("normalized eigenvalue (log)")
    axes[1].set_title(f"Kernel spectrum — VS={cc['full_vendi']:.1f} effective behaviors / {len(K_full)} episodes")
    fig.tight_layout()
    fig.savefig(FIG / "e5_curation_spectrum.png", dpi=150)

    ten_pct = [c for c in cc["curve"] if c["fraction_kept"] == 0.1][0]
    print(
        f"E5 curation: 10% of episodes retain {100 * ten_pct['vendi_retained_frac']:.0f}% of diversity; "
        f"full corpus = {cc['full_vendi']:.1f} effective behaviors ({len(FAMILIES)} planted)"
    )

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nwrote {OUT / 'metrics.json'} and {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
