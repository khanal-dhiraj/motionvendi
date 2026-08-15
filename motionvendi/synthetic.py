"""Synthetic EgoVerse-like episodes with KNOWN ground truth.

Why synthetic: the falsifiability of the metric comes from controls where the
right answer is known in advance. We generate distinct behavior families
(reach, wipe-circle, pour, fold, stir ...) as parametric two-hand trajectories
in the EgoVerse pose-row format ([x,y,z,qw,qx,qy,qz], wxyz), then produce:

  * near-duplicates   — same family + same params + small execution noise
  * nuisance variants — same behavior, different world origin / speed / phase
  * corruptions       — teleports, NaN dropouts, frozen tracker, quat garbage

Ground truth labels let us report industry-standard detection metrics
(precision / recall / F1 / AUROC) for the gates, duplicate-recall@k for the
kernel, and label-collapse behavior for the score.
"""

from __future__ import annotations

import numpy as np

from .normalize import matrix_to_quat_wxyz

FAMILIES = ["reach", "wipe_circle", "pour", "fold", "stir", "handoff"]


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _traj(family: str, t: np.ndarray, p: dict) -> np.ndarray:
    """Right-hand xyz path for a behavior family on t in [0,1]. Meters."""
    a = p["amp"]
    if family == "reach":
        return np.stack([0.4 * t, 0.05 * np.sin(np.pi * t) * a, 0.2 * t], axis=1)
    if family == "wipe_circle":
        th = 2 * np.pi * p["reps"] * t
        return np.stack([0.15 * np.cos(th) * a, 0.15 * np.sin(th) * a, 0.02 * np.sin(4 * th)], axis=1)
    if family == "pour":
        return np.stack([0.25 * t, 0.1 * (1 - np.cos(np.pi * t)) * a, 0.15 * np.sin(np.pi * t)], axis=1)
    if family == "fold":
        return np.stack([0.2 * np.abs(np.sin(2 * np.pi * t)) * a, 0.3 * t - 0.15, 0.05 * np.cos(2 * np.pi * t)], axis=1)
    if family == "stir":
        th = 2 * np.pi * p["reps"] * t
        return np.stack([0.06 * np.cos(th) * a, 0.06 * np.sin(th) * a, -0.02 * t], axis=1)
    if family == "handoff":
        return np.stack([0.3 * np.sin(np.pi * t / 2) * a, 0.25 * t - 0.1, 0.1 * t * (1 - t) * 4], axis=1)
    raise ValueError(family)


def make_episode(
    family: str,
    rng: np.random.Generator,
    n_frames: int = 240,
    world_offset: np.ndarray | None = None,
    speed_warp: float = 1.0,
    exec_noise: float = 0.004,
    seed_params: dict | None = None,
) -> dict:
    """One synthetic episode: left/right EE pose rows + head pose rows (T, 7)."""
    p = seed_params or {"amp": rng.uniform(0.8, 1.2), "reps": rng.integers(2, 5)}
    t = np.linspace(0, 1, n_frames) ** speed_warp
    right = _traj(family, t, p)
    left = _traj(family, 1 - t, p) * np.array([-1, 1, 1]) * 0.6  # loosely mirrored support hand
    off = world_offset if world_offset is not None else rng.uniform(-3, 3, 3)
    yaw = rng.uniform(-np.pi, np.pi)
    Rw = _rot_z(yaw)

    def rows_from(path: np.ndarray) -> np.ndarray:
        pts_clean = path @ Rw.T + off
        pts = pts_clean + rng.normal(0, exec_noise, path.shape)
        # wrist orientation: tangent yaw of the NOISELESS path — mm-scale
        # execution noise exceeds per-frame displacement, so a noisy tangent
        # would be a random walk (a physical wrist doesn't jitter like that)
        d = np.gradient(pts_clean, axis=0)
        yaws = np.unwrap(np.arctan2(d[:, 1], d[:, 0] + 1e-9))
        w = 15
        padded = np.pad(yaws, w, mode="edge")  # edge-pad: zero-pad would fake a boundary teleport
        yaws = np.convolve(padded, np.ones(w) / w, mode="same")[w:-w]
        rows = np.zeros((len(pts), 7))
        rows[:, :3] = pts
        for i, ang in enumerate(yaws):
            rows[i, 3:7] = matrix_to_quat_wxyz(_rot_z(float(ang)))
        return rows

    head = np.zeros((n_frames, 7))
    head[:, :3] = off + np.array([0, -0.4, 0.5]) + rng.normal(0, 0.002, (n_frames, 3))
    for i in range(n_frames):
        head[i, 3:7] = matrix_to_quat_wxyz(Rw)
    return {
        "family": family,
        "params": p,
        "left": rows_from(left),
        "right": rows_from(right),
        "head": head,
    }


# ---------------------------------------------------------------- corruptions

def corrupt_teleport(ep: dict, rng: np.random.Generator) -> dict:
    out = {**ep, "right": ep["right"].copy(), "corruption": "teleport"}
    idx = rng.integers(10, len(out["right"]) - 10, size=3)
    out["right"][idx, :3] += rng.uniform(1.0, 3.0, (3, 3)) * rng.choice([-1, 1], (3, 3))
    return out


def corrupt_nan_dropout(ep: dict, rng: np.random.Generator, frac: float = 0.2) -> dict:
    out = {**ep, "right": ep["right"].copy(), "corruption": "nan_dropout"}
    n = len(out["right"])
    start = int(rng.integers(0, int(n * (1 - frac))))
    out["right"][start : start + int(n * frac)] = np.nan
    return out


def corrupt_frozen(ep: dict, rng: np.random.Generator) -> dict:
    out = {**ep, "right": ep["right"].copy(), "corruption": "frozen"}
    n = len(out["right"])
    out["right"][int(n * 0.05) :] = out["right"][int(n * 0.05)]
    return out


def corrupt_quat(ep: dict, rng: np.random.Generator) -> dict:
    out = {**ep, "right": ep["right"].copy(), "corruption": "bad_quat"}
    idx = rng.random(len(out["right"])) < 0.3
    out["right"][idx, 3:7] = rng.normal(0, 5.0, (int(idx.sum()), 4))
    return out


CORRUPTIONS = [corrupt_teleport, corrupt_nan_dropout, corrupt_frozen, corrupt_quat]


def make_corpus(
    n_per_family: int = 20,
    n_duplicate_pairs: int = 12,
    n_corrupt: int = 24,
    n_frames: int = 240,
    seed: int = 7,
) -> list[dict]:
    """Corpus with ground truth: `family`, `is_duplicate_of`, `corruption` keys."""
    rng = np.random.default_rng(seed)
    corpus: list[dict] = []
    for fam in FAMILIES:
        for _ in range(n_per_family):
            corpus.append(make_episode(fam, rng, n_frames=n_frames))
    # near-duplicates: same family/params/offset, tiny execution noise, new speed
    dup_sources = rng.choice(len(corpus), size=n_duplicate_pairs, replace=False)
    for src in dup_sources:
        ep = corpus[int(src)]
        dup = make_episode(
            ep["family"],
            rng,
            n_frames=n_frames,
            world_offset=None,  # different room — behavior is the same
            speed_warp=float(rng.uniform(0.8, 1.25)),
            seed_params=ep["params"],
        )
        dup["is_duplicate_of"] = int(src)
        corpus.append(dup)
    # corruptions applied to fresh clean episodes
    for i in range(n_corrupt):
        fam = FAMILIES[i % len(FAMILIES)]
        ep = make_episode(fam, rng, n_frames=n_frames)
        corpus.append(CORRUPTIONS[i % len(CORRUPTIONS)](ep, rng))
    return corpus
