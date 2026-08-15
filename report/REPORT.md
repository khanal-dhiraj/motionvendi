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

## 4b. Real-data results (237 EgoVerse episodes, 4 labs)

Stratified pose-only sample from R2 (`processed_v3/{aria, microagi, mecka/flagship, scale}`, ~60 episodes each, ~400 KB/episode without video).

**R1 — Prevalence audit.** 63/237 episodes (27%) fail gates, and the failure *signature identifies the vendor pipeline*: aria fails on wrist-rotation noise (28 streams), mecka on degenerate quaternions (14) + frozen streams (10) + missing pose arrays (7), scale almost exclusively on frozen streams (7). Aria's rotation-violation rate is median **1.0%/frame vs 0% for mecka/scale** — a 10x noise floor difference that independently corroborates the EgoVerse changelog entry fixing aria EE orientations. Consequence: a single global threshold either over-drops aria or under-audits scale; quality scoring must be stratified by vendor.

**Two data quirks discovered and encoded:** (a) zarr arrays are zero-padded past `total_frames` — untruncated, every tail reads as frozen + degenerate-quaternion corruption; (b) sparse teleports (0.3–0.7% of frames) are *normal* in egocentric tracking (hands exit the FOV) — gates recalibrated from single-event to violation-rate thresholds (2% of frames), after which synthetic detection stays perfect (E1 re-run: P/R/F1/AUROC = 1.0) while real drop rates become plausible (100% of aria dropped → 35%).

**R2 — Real label collapse.** `fold_clothes` (n=11): task VS 6.41 vs random-mix VS 7.10. Weak separation, honestly reported: n is small and clothes-folding is genuinely one of the highest-variety manipulation tasks. A larger sample of a stereotyped task (`wash_dishes`) is the natural follow-up.

**R3 — Per-lab diversity (size-fair, 40 bootstraps at fixed n).** mecka **15.0** > microagi 12.3 > scale 11.0 > aria 9.5 effective behaviors. Hours are not information: labs contribute very differently per episode.

**R4 — Curation.** Full sample = 33.8 effective behaviors from 162 gated episodes; greedy keep-list, top-10 nearest-duplicate pairs (by episode name, auditable), and a per-episode keep/drop CSV with per-gate evidence (`real_keep_drop.csv`).

## 5. Threats to validity, stated plainly

1. **Synthetic-to-real gap.** Gates/quotients are validated where ground truth exists (synthetic); real-data rates above are measured on a 237-episode stratified sample, not the full 439k corpus.
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
