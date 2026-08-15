"""Quality gates: remove measurement lies BEFORE measuring diversity.

First principles: noise is maximally novel. A glitched tracker produces the
most "diverse" trajectories in the corpus, so any diversity metric applied to
ungated data rewards corruption. Gates are therefore a precondition of the
metric, not a separate curation step.

Every gate returns a per-episode boolean + evidence dict so keep/drop
decisions are auditable (no black box).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .normalize import quat_wxyz_to_matrix


@dataclass
class GateReport:
    """Verdict + evidence for one episode/segment."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


def _positions(rows: np.ndarray) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)[:, :3]


def gate_finite(rows: np.ndarray, max_nan_frac: float = 0.05) -> tuple[bool, dict]:
    """Reject episodes with too many NaN/inf frames (tracking dropout)."""
    rows = np.asarray(rows, dtype=np.float64)
    bad = ~np.all(np.isfinite(rows), axis=1)
    frac = float(bad.mean()) if len(rows) else 1.0
    return frac <= max_nan_frac, {"nan_frac": frac}


def gate_teleport(
    rows: np.ndarray,
    fps: float = 30.0,
    max_speed_m_s: float = 6.0,
    max_violation_frac: float = 0.02,
) -> tuple[bool, dict]:
    """Reject episodes where position jumps beyond a physical hand-speed limit
    are PERVASIVE (violation rate), not merely present.

    Peak human hand speed is ~5-6 m/s (throwing); a jump above that is a
    tracker re-fit, not a movement. Real egocentric data (measured on aria)
    carries 0.3-0.7% sparse glitch frames from hands leaving the FOV — those
    are maskable in training, not grounds to discard a 2-minute episode. An
    episode is corrupt when glitches exceed ~2% of frames.
    """
    pos = _positions(rows)
    if len(pos) < 2:
        return False, {"max_speed": np.inf, "teleport_frac": 1.0}
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    speed = speed[np.isfinite(speed)]
    if not len(speed):
        return False, {"max_speed": np.inf, "teleport_frac": 1.0}
    frac = float((speed > max_speed_m_s).mean())
    return frac <= max_violation_frac, {
        "max_speed": float(speed.max()),
        "teleport_frac": frac,
    }


def gate_frozen(
    rows: np.ndarray, max_frozen_frac: float = 0.9, eps: float = 1e-9
) -> tuple[bool, dict]:
    """Reject episodes where the pose stream is a frozen constant (dead tracker).

    Distinct from being idle: a real idle hand still jitters at mm scale; a
    bit-identical repeated row is a pipeline failure.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if len(rows) < 2:
        return False, {"frozen_frac": 1.0}
    frozen = np.all(np.abs(np.diff(rows, axis=0)) < eps, axis=1)
    frac = float(frozen.mean())
    return frac <= max_frozen_frac, {"frozen_frac": frac}


def gate_quaternion(rows: np.ndarray, tol: float = 0.05) -> tuple[bool, dict]:
    """Reject degenerate quaternions (norm far from 1 → junk orientation data)."""
    q = np.asarray(rows, dtype=np.float64)[:, 3:7]
    norms = np.linalg.norm(q, axis=1)
    finite = np.isfinite(norms)
    if not finite.any():
        return False, {"bad_quat_frac": 1.0}
    bad = np.abs(norms[finite] - 1.0) > tol
    frac = float(bad.mean())
    return frac <= 0.05, {"bad_quat_frac": frac}


def gate_rotation_rate(
    rows: np.ndarray,
    fps: float = 30.0,
    max_rad_s: float = 4.0 * np.pi,
    max_violation_frac: float = 0.02,
) -> tuple[bool, dict]:
    """Reject pervasive implausible wrist angular velocity (orientation
    teleports) — rate-thresholded like gate_teleport, for the same reason."""
    q = np.asarray(rows, dtype=np.float64)[:, 3:7]
    finite = np.all(np.isfinite(q), axis=1)
    if finite.sum() < 2:
        return False, {"max_rot_rate": np.inf, "rot_violation_frac": 1.0}
    R = quat_wxyz_to_matrix(q[finite])
    # relative rotation angle between consecutive frames
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])  # R_t^T R_{t+1}
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rate = np.arccos(cos) * fps
    if not len(rate):
        return False, {"max_rot_rate": np.inf, "rot_violation_frac": 1.0}
    frac = float((rate > max_rad_s).mean())
    return frac <= max_violation_frac, {
        "max_rot_rate": float(rate.max()),
        "rot_violation_frac": frac,
    }


def gate_min_length(rows: np.ndarray, min_frames: int = 30) -> tuple[bool, dict]:
    """Reject segments too short to contain a behavior (< 1 s at 30 fps)."""
    n = len(rows)
    return n >= min_frames, {"n_frames": n}


def run_gates(rows: np.ndarray, fps: float = 30.0) -> GateReport:
    """Run all gates on one pose stream ``(T, 7)``; collect evidence."""
    checks = {
        "min_length": gate_min_length(rows),
        "finite": gate_finite(rows),
        "frozen": gate_frozen(rows),
        "quaternion": gate_quaternion(rows),
        "teleport": gate_teleport(rows, fps=fps),
        "rotation_rate": gate_rotation_rate(rows, fps=fps),
    }
    reasons = [name for name, (ok, _) in checks.items() if not ok]
    evidence = {name: ev for name, (_, ev) in checks.items()}
    return GateReport(passed=not reasons, reasons=reasons, evidence=evidence)


def gate_episode(
    ee_left: np.ndarray | None, ee_right: np.ndarray | None, fps: float = 30.0
) -> GateReport:
    """Gate an episode on every hand stream it actually has.

    A missing hand (single-arm embodiment) is not a failure; a present but
    corrupt hand is.
    """
    reports = {}
    for side, rows in (("left", ee_left), ("right", ee_right)):
        if rows is not None:
            reports[side] = run_gates(rows, fps=fps)
    if not reports:
        return GateReport(passed=False, reasons=["no_pose_streams"])
    reasons = [f"{side}:{r}" for side, rep in reports.items() for r in rep.reasons]
    evidence = {side: rep.evidence for side, rep in reports.items()}
    return GateReport(passed=not reasons, reasons=reasons, evidence=evidence)
