"""MotionVendi on REAL EgoVerse episodes (pose-only R2 sample).

Same pipeline as the synthetic validation, now producing the real-data
deliverables:

  R1  Prevalence audit  — gate failure rates per lab with reasons (Track 1:
                          how much of the corpus is measurement lies?)
  R2  Real label collapse — episodes sharing the most common task_name vs a
                          random same-size mix (the falsifiable core, on
                          real labels)
  R3  Per-lab diversity — size-fair bootstrap Vendi per lab (who actually
                          contributes behaviors, not hours?)
  R4  Curation          — greedy keep-list, curve, top redundant episodes and
                          nearest-duplicate pairs BY NAME (auditable)

Outputs: report/figures/real_*.png + report/real_metrics.json + keep/drop CSV.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionvendi.curate import curation_curve, nearest_duplicates
from motionvendi.kernels import kernel_histogram_stats, median_bandwidth, rbf_kernel, validate_psd
from motionvendi.vendi import bootstrap_vendi, vendi_score
from motionvendi.zarr_loader import load_folder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/real"
OUT = ROOT / "report"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)


def lab_of(ep: dict) -> str:
    return Path(ep["path"]).parent.name


def main() -> None:
    metrics: dict = {}
    X, kept, dropped = load_folder(DATA, n_steps=32)
    all_eps = len(kept) + len(dropped)
    print(f"loaded {all_eps} episodes: {len(kept)} passed gates, {len(dropped)} dropped")

    # ------------------------------------------------------ R1 prevalence audit
    labs = sorted({lab_of(ep) for ep, _ in dropped} | {lab_of(ep) for ep in kept})
    audit = {}
    for lab in labs:
        n_kept = sum(lab_of(ep) == lab for ep in kept)
        lab_dropped = [(ep, r) for ep, r in dropped if lab_of(ep) == lab]
        reasons = Counter(reason.split(":")[-1] for _, r in lab_dropped for reason in r.reasons)
        audit[lab] = {
            "n": n_kept + len(lab_dropped),
            "n_dropped": len(lab_dropped),
            "drop_rate": len(lab_dropped) / max(1, n_kept + len(lab_dropped)),
            "reasons": dict(reasons),
        }
    metrics["R1_prevalence_audit"] = audit
    for lab, a in audit.items():
        print(f"R1 {lab}: {a['n_dropped']}/{a['n']} dropped ({100*a['drop_rate']:.0f}%) reasons={a['reasons']}")

    # main analysis stratum: episodes with a fully-finite behavior vector
    finite = np.all(np.isfinite(X), axis=1) if len(X) else np.array([], bool)
    Xf = X[finite]
    keptf = [ep for ep, ok in zip(kept, finite) if ok]
    no_head = sum(1 for ep in keptf if not ep["has_head_pose"])
    metrics["stratum"] = {
        "n_vectors": int(len(Xf)),
        "n_missing_hand_excluded": int(len(kept) - len(Xf)),
        "n_first_frame_fallback": no_head,
        "embodiments": dict(Counter(ep["embodiment"] for ep in keptf)),
    }
    print(f"stratum: {len(Xf)} full-vector episodes ({no_head} use first-frame fallback)")

    sigma2 = median_bandwidth(Xf)
    kb = lambda A: rbf_kernel(A, sigma2=sigma2)
    K = kb(Xf)
    psd_ok, min_eig = validate_psd(K)
    metrics["kernel"] = {
        "sigma2_median_heuristic": sigma2,
        "psd_ok": bool(psd_ok),
        "min_eig": min_eig,
        "offdiag_stats": kernel_histogram_stats(K),
    }

    # ------------------------------------------------- R2 real label collapse
    tasks = Counter(ep["task_name"] for ep in keptf if ep["task_name"])
    top_task, top_n = (tasks.most_common(1)[0] if tasks else ("", 0))
    if top_n >= 8:
        idx_task = [i for i, ep in enumerate(keptf) if ep["task_name"] == top_task]
        idx_rand = RNG.choice(len(Xf), size=len(idx_task), replace=False)
        vs_task = vendi_score(kb(Xf[idx_task]))
        vs_rand = vendi_score(kb(Xf[idx_rand]))
        sweep = {}
        for mult in [0.1, 0.3, 1.0, 3.0, 10.0]:
            kbm = lambda A: rbf_kernel(A, sigma2=sigma2 * mult)
            sweep[mult] = {
                "one_task": vendi_score(kbm(Xf[idx_task])),
                "random_mix": vendi_score(kbm(Xf[idx_rand])),
            }
        metrics["R2_real_label_collapse"] = {
            "task": top_task,
            "n": len(idx_task),
            "vendi_one_task": vs_task,
            "vendi_random_mix": vs_rand,
            "collapse_ratio": vs_task / vs_rand if vs_rand else None,
            "bandwidth_sweep": sweep,
        }
        print(f"R2 collapse on '{top_task}' (n={len(idx_task)}): task VS={vs_task:.2f} vs random VS={vs_rand:.2f}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(
            [f"one task\n('{top_task[:24]}')", "random mix\n(same n)"],
            [vs_task, vs_rand],
            color=["#264653", "#e9c46a"],
        )
        axes[0].set_ylabel(f"Vendi score (n={len(idx_task)})")
        axes[0].set_title("REAL DATA label-collapse test")
        mults = sorted(sweep)
        axes[1].plot(mults, [sweep[m]["one_task"] for m in mults], "o-", label="one task")
        axes[1].plot(mults, [sweep[m]["random_mix"] for m in mults], "s-", label="random mix")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("bandwidth multiplier")
        axes[1].set_ylabel("Vendi score")
        axes[1].set_title("Honesty bracket (real data)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(FIG / "real_label_collapse.png", dpi=150)
    else:
        print(f"R2 skipped: top task '{top_task}' has only {top_n} episodes in sample")
        metrics["R2_real_label_collapse"] = {"skipped": True, "top_task": top_task, "n": top_n}

    # ---------------------------------------------------- R3 per-lab diversity
    per_lab = {}
    lab_counts = Counter(lab_of(ep) for ep in keptf)
    fair_n = min(n for n in lab_counts.values() if n >= 10) if lab_counts else 0
    for lab, n in sorted(lab_counts.items()):
        if n < 10:
            per_lab[lab] = {"n": n, "skipped": "too few episodes"}
            continue
        idx = [i for i, ep in enumerate(keptf) if lab_of(ep) == lab]
        per_lab[lab] = {"n": n, **bootstrap_vendi(Xf[idx], kb, sample_size=fair_n, n_boot=40, rng=RNG)}
    metrics["R3_per_lab_diversity"] = {"fair_sample_size": fair_n, "labs": per_lab}
    print("R3 per-lab VS (size-fair):", {l: round(v.get("mean", 0), 1) for l, v in per_lab.items() if "mean" in v})

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ls = [l for l, v in per_lab.items() if "mean" in v]
    ax.bar(
        ls,
        [per_lab[l]["mean"] for l in ls],
        yerr=[per_lab[l]["std"] for l in ls],
        color="#2a9d8f",
        capsize=6,
    )
    ax.set_ylabel(f"Vendi score (fixed n={fair_n}, 40 boots)")
    ax.set_title("REAL DATA: effective behavioral diversity by lab (size-fair)")
    fig.tight_layout()
    fig.savefig(FIG / "real_per_lab_diversity.png", dpi=150)

    # ------------------------------------------------------------- R4 curation
    cc = curation_curve(K, fractions=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    dup_pairs = nearest_duplicates(K, top_k=10)
    metrics["R4_curation"] = {
        "full_vendi": cc["full_vendi"],
        "peak_vendi": cc["peak_vendi"],
        "n": len(K),
        "curve": cc["curve"],
        "top10_duplicate_pairs": [
            {
                "a": keptf[i]["name"], "a_lab": lab_of(keptf[i]), "a_task": keptf[i]["task_name"],
                "b": keptf[j]["name"], "b_lab": lab_of(keptf[j]), "b_task": keptf[j]["task_name"],
                "similarity": s,
            }
            for i, j, s in dup_pairs
        ],
    }
    ten = [c for c in cc["curve"] if c["fraction_kept"] == 0.1][0]
    print(f"R4: full VS={cc['full_vendi']:.1f} of n={len(K)}; 10% keeps {100*ten['vendi_retained_frac']:.0f}% of peak diversity")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    fr = [c["fraction_kept"] for c in cc["curve"]]
    rt = [c["vendi_retained_frac"] for c in cc["curve"]]
    ax.plot(fr, rt, "o-", color="#2a9d8f", label="greedy max-Vendi")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="random keep")
    ax.set_xlabel("fraction of episodes kept")
    ax.set_ylabel("fraction of peak diversity retained")
    ax.set_title(f"REAL DATA curation curve (n={len(K)} episodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "real_curation_curve.png", dpi=150)

    # keep/drop CSV: every episode, verdict, reason or greedy rank
    rank_of = {i: r for r, i in enumerate(cc["greedy_order"])}
    with (OUT / "real_keep_drop.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "lab", "task", "verdict", "detail"])
        for ep, rep in dropped:
            w.writerow([ep["name"], lab_of(ep), ep.get("task_name", ""), "DROP", ";".join(rep.reasons)])
        for i, ep in enumerate(keptf):
            w.writerow([ep["name"], lab_of(ep), ep["task_name"], "KEEP", f"greedy_rank={rank_of.get(i, -1)}"])

    (OUT / "real_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nwrote real_metrics.json, real_keep_drop.csv, {len(list(FIG.glob('real_*.png')))} real figures")


if __name__ == "__main__":
    main()
