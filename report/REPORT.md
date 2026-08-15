# MotionVendi — Validation Report

*EgoVerse Data Optimization & Evaluation Suite hackathon · Tracks 1 (Curation Engine) + 2 (Quantitative Diversity Measurement)*

## 1. Problem statement

Imitation-learning corpora are valued by volume (hours, episodes), but training value is information about the state→action mapping, and information is destroyed by two things volume can't see: **corruption** (tracking failures that inject label noise) and **redundancy** (near-identical demos that add zero marginal signal). In EgoVerse specifically, the usual proxies for both are unavailable: task labels are free-text chaos (27,997 distinct strings; 345 spellings of "washing dishes"), success labels have no negative class, and operator/scene metadata is 11–19% filled.

**Hypothesis:** episode value = quality x novelty, and both are measurable from pose streams alone. Corollary (the claim that unifies curation with diversity measurement): *noise is maximally novel* — an ungated diversity metric rewards corruption, so gating is a precondition of measurement, not a separate pipeline stage.

## 2. Method

**Gates** (auditable, evidence-attached): NaN/dropout fraction, physical hand-speed limit (teleport ⇒ tracker re-fit), frozen-stream detection, quaternion degeneracy, angular-rate limit, minimum length. Every keep/drop ships its per-gate evidence dict.

**Nuisance quotient** (each step deletes one nuisance variable):
| Nuisance | Quotient |
|---|---|
| per-episode SLAM world origin | EE pose in instantaneous head frame; first-frame fallback when head pose absent (Scale) |
| speed profile / time warp | **arc-length** reparameterization to fixed steps (uniform-time resampling provably fails this — §4, E4) |
| rotation chart discontinuities | 6D rotation representation |
| position offset in fallback mode | per-episode centering |
| executor handedness | per-hand blocks; missing hand = NaN block, never zero-filled (zeros fabricate a fake behavior cluster) |

**Metric:** RBF kernel on behavior vectors (median-heuristic bandwidth + full sweep reported), explicit PSD validation, Vendi Score = exp(Shannon entropy of the normalized kernel spectrum) — the effective number of distinct behaviors (Friedman & Dieng, TMLR 2023). Order-q variants reported for robustness. **Size-fairness:** VS is bounded by n and non-monotone under growth, so subsets are ranked only via fixed-size bootstrap resamples with 95% CIs.

**Curation:** greedy max-Vendi selection ⇒ keep-list in marginal-diversity order; its reverse is the most-redundant-first drop ranking; the trace is the curation curve.

## 3. Experimental design

All claims are tested on a synthetic corpus with planted ground truth, because falsifiability requires knowing the right answer in advance: 6 parametric behavior families x 20 episodes (per-episode shape variation — humans vary path shape, not just scale), 12 near-duplicate pairs that differ **only by nuisance** (new world pose, new speed warp, fresh execution noise — same behavior), 24 corruptions (teleport, NaN dropout, frozen tracker, quaternion garbage). Pose rows use the EgoVerse convention `[x,y,z,qw,qx,qy,qz]` (wxyz), pinned by round-trip unit tests.

## 4. Results

**E1 — Gate detection.** 24/24 corruptions flagged, 0/132 clean episodes falsely dropped: **precision 1.00, recall 1.00, F1 1.00, AUROC 1.00** (AUROC over a continuous severity score, Mann-Whitney formulation). Physical-limit gates are near-noiseless detectors by construction — the corruption classes they target violate physics, not statistics.

**E2 — Noise inflates measured diversity (core claim).** Identical kernel, identical bandwidth, fixed sample size n=100, 40 bootstraps: ungated corpus VS = **11.39** [10.68, 12.19]; gated VS = **9.79** [9.49, 10.13]. Corruption manufactured **+16.4%** fake diversity. Non-overlapping CIs. Consequence: any diversity-aware data pipeline that does not gate first is optimizing for tracker failure.

**E3 — Label collapse with two-sided control.** 20 episodes of one behavior family score **VS 3.19**; 20 random episodes across families score **VS 6.67** (collapse ratio 0.48). The separation persists across a 100x bandwidth sweep (0.1x–10x median heuristic), which brackets the two dishonest failure modes: too-wide (everything collapses, metric blind) and too-narrow (nothing collapses, VS→n). PSD verified on every kernel; off-diagonal similarity histogram reported.

**E4 — Duplicate retrieval.** Planted duplicate pairs are retrieved at **NN-recall@1 = 0.75, @5 = 0.83** (n=12, standard per-query retrieval formulation). This experiment caught a real methodological bug during development: with uniform-*time* resampling, recall was **0.00** — time-warped duplicates were invisible. Arc-length reparameterization fixed it, and the invariance is now unit-tested. Residual misses are duplicates whose source has an independently-drawn near-twin in-family — visible in the top-5 most-similar pairs, all of which are same-family.

**E5 — Curation curve.** Greedy max-Vendi keep-list: **10% of episodes retain 67% of peak diversity** (random keep retains ~10%); 30% retains ~89%. Full corpus = 10.2 effective behaviors from 156 episodes (6 planted families + genuine within-family variation). Kernel eigenvalue spectrum published alongside — a score without its spectrum hides concentration.

## 5. Threats to validity, stated plainly

1. **Synthetic-to-real gap.** Gates/quotients are validated where ground truth exists; real EgoVerse rates (teleport prevalence, idle fraction) are unmeasured here. The zarr loader runs the identical code path on real episodes, pose-arrays-only.
2. **Object blindness.** The pose kernel deliberately excludes appearance; distinct objects with identical motion are conflated. Fixed-weight PSD image/semantic kernels are the extension point (never per-pair reweighted — entry-wise weight renormalization voids PSD).
3. **Speed deletion.** Arc-length removes the speed profile entirely. Where tempo is the skill, re-append it as an explicit scalar feature.
4. **Scale.** Eigendecomposition caps at ~10^4; corpus claims are bootstrap estimates with CIs, or need Nyström approximation.
5. **Scale-vendor stratum.** Episodes without head pose get a weaker quotient (first-frame) and must be reported separately, never silently mixed.

## 6. Deliverables mapping

| Track requirement | Artifact |
|---|---|
| Keep/drop recommendations (T1) | gate verdicts w/ evidence + greedy drop-ranking (`curate.redundancy_ranking`) |
| Validation report w/ proxy metrics (T1) | this report; P/R/F1/AUROC, bootstrap CIs, curation curve |
| Non-text diversity score ranking two subsets (T2) | pose-only Vendi w/ size-fair bootstrap comparison |
| Dashboard comparing subsets (T2) | `report/figures/*` (inflation bars, collapse + bandwidth sweep, curve + spectrum) |

## 7. References

- Friedman & Dieng, *The Vendi Score: A Diversity Evaluation Metric for Machine Learning*, TMLR 2023.
- Hejna et al., *DemInf: Demonstration Quality Estimation via Mutual Information*, RSS 2025 (offline quality scoring precedent).
- *RINSE*, 2026 (smoothness-filtered subsets: +16% success at 1/6 data — motivates kinematic gates).
- Khazatsky et al., *DROID*, RSS 2024 (idle-filter precedent in production robot data).
- Abbas et al., *SemDeDup*, 2023 (embedding dedup at scale — the appearance-kernel extension point).
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019 (6D rotations).
