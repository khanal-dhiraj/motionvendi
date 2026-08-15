"""Pose normalization: quotient out nuisance variables so that what remains is behavior.

EgoVerse pose rows are ``[x, y, z, qw, qx, qy, qz]`` — position followed by a
**wxyz** quaternion. Misreading the quaternion convention produces
plausible-looking garbage with no error, so every entry point here goes through
:func:`pose_row_to_matrix`, which is pinned by a round-trip unit test.

Nuisance variables and their quotients:
  * world-frame arbitrariness (per-episode SLAM origin) -> express EE poses in
    the instantaneous head frame (`to_head_frame`), or first-frame wrist frame
    as a fallback for episodes without head pose (Scale vendor).
  * time scale -> arc-length/uniform resampling to a fixed number of steps.
  * rotation parameterization discontinuities -> 6D rotation representation
    (first two columns of R; Zhou et al. 2019).
"""

from __future__ import annotations

import numpy as np

POSE_DIM = 7  # x y z qw qx qy qz


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) in **wxyz** order to rotation matrices.

    Accepts shape (4,) or (T, 4). Returns (3, 3) or (T, 3, 3).
    Quaternions are normalized; zero-norm quaternions yield NaN matrices so
    that downstream validity gates can catch them (never silently identity).
    """
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(norm > 1e-12, q / norm, np.nan)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R[0] if single else R


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quat_wxyz_to_matrix` (single 3x3 matrix), wxyz order."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def pose_row_to_matrix(rows: np.ndarray) -> np.ndarray:
    """``[x,y,z,qw,qx,qy,qz]`` row(s) -> homogeneous 4x4 transform(s)."""
    rows = np.asarray(rows, dtype=np.float64)
    single = rows.ndim == 1
    rows = np.atleast_2d(rows)
    if rows.shape[1] != POSE_DIM:
        raise ValueError(f"expected pose rows of dim {POSE_DIM}, got {rows.shape[1]}")
    T = np.tile(np.eye(4), (rows.shape[0], 1, 1))
    T[:, :3, :3] = quat_wxyz_to_matrix(rows[:, 3:7])
    T[:, :3, 3] = rows[:, :3]
    return T[0] if single else T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert homogeneous transform(s), shape (4,4) or (T,4,4)."""
    T = np.asarray(T, dtype=np.float64)
    single = T.ndim == 2
    T = T.reshape(-1, 4, 4)
    Rt = np.swapaxes(T[:, :3, :3], 1, 2)
    out = np.tile(np.eye(4), (T.shape[0], 1, 1))
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum("tij,tj->ti", Rt, T[:, :3, 3])
    return out[0] if single else out


def to_head_frame(ee_rows: np.ndarray, head_rows: np.ndarray) -> np.ndarray:
    """Express EE poses in the instantaneous head frame.

    T_head<-ee(t) = inv(T_world<-head(t)) @ T_world<-ee(t).
    Kills the per-episode SLAM world origin — the largest nuisance variable.
    Returns (T, 4, 4).
    """
    T_we = pose_row_to_matrix(ee_rows)
    T_wh = pose_row_to_matrix(head_rows)
    return np.einsum("tij,tjk->tik", invert_transform(T_wh), T_we)


def to_first_frame(ee_rows: np.ndarray) -> np.ndarray:
    """Fallback quotient when head pose is absent (Scale vendor): express the
    trajectory relative to its own first valid frame."""
    T = pose_row_to_matrix(ee_rows)
    finite = np.all(np.isfinite(T.reshape(len(T), -1)), axis=1)
    if not finite.any():
        return np.full_like(T, np.nan)
    anchor = invert_transform(T[int(np.argmax(finite))])
    return np.einsum("ij,tjk->tik", anchor, T)


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """(…,3,3) rotation matrices -> continuous 6D representation (first two columns)."""
    R = np.asarray(R, dtype=np.float64)
    return R[..., :3, :2].reshape(*R.shape[:-2], 6)


def transforms_to_features(T: np.ndarray) -> np.ndarray:
    """(T,4,4) transforms -> (T, 9) features: xyz + 6D rotation."""
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[:, :3, 3], rotmat_to_6d(T[:, :3, :3])], axis=1)


def resample_uniform(X: np.ndarray, n_steps: int) -> np.ndarray:
    """Linearly resample a (T, D) sequence to (n_steps, D) on a uniform time grid.

    Deletes duration as a nuisance variable while preserving the *shape* of the
    motion. NaN frames are dropped before interpolation; if fewer than two
    valid frames remain, returns all-NaN so gates can reject the segment.
    """
    X = np.asarray(X, dtype=np.float64)
    valid = np.all(np.isfinite(X), axis=1)
    if valid.sum() < 2:
        return np.full((n_steps, X.shape[1]), np.nan)
    Xv = X[valid]
    src = np.linspace(0.0, 1.0, len(Xv))
    dst = np.linspace(0.0, 1.0, n_steps)
    out = np.empty((n_steps, X.shape[1]))
    for d in range(X.shape[1]):
        out[:, d] = np.interp(dst, src, Xv[:, d])
    return out


def episode_to_vector(
    ee_rows_left: np.ndarray | None,
    ee_rows_right: np.ndarray | None,
    head_rows: np.ndarray | None,
    n_steps: int = 32,
) -> np.ndarray:
    """Full quotient pipeline for one episode/segment -> flat behavior vector.

    Each available hand: head-frame (or first-frame fallback) -> xyz+6D ->
    uniform resample -> flatten. Hands are concatenated; a missing hand is
    encoded as its own NaN block and should be handled by scoring
    per-embodiment (never zero-filled — zeros would fabricate a fake
    'distinct behavior' cluster).
    """
    blocks: list[np.ndarray] = []
    for rows in (ee_rows_left, ee_rows_right):
        if rows is None:
            blocks.append(np.full(n_steps * 9, np.nan))
            continue
        T = to_head_frame(rows, head_rows) if head_rows is not None else to_first_frame(rows)
        feats = transforms_to_features(T)
        # center positions so the quotient is translation-clean even in fallback mode
        pos = feats[:, :3]
        finite = np.all(np.isfinite(pos), axis=1)
        if finite.any():
            feats[:, :3] = pos - np.nanmean(pos[finite], axis=0)
        blocks.append(resample_uniform(feats, n_steps).ravel())
    return np.concatenate(blocks)
