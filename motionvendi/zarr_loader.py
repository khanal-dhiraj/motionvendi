"""Load real EgoVerse episodes (folder of per-episode .zarr stores).

Real processed_v3 layout (verified against R2 2026-08-15): zarr v3 stores with
DOT-named arrays at the root — ``left.obs_ee_pose``, ``right.obs_ee_pose``,
``obs_head_pose``, ``annotations`` — and root ``zarr.json`` attributes carrying
``embodiment``, ``task_name``, ``task_description``, ``fps``, ``total_frames``,
``intrinsics``. Pose rows are [x,y,z,qw,qx,qy,qz] (wxyz quaternion).

We read ONLY pose arrays + attrs (never ``images.front_1``, which is nearly
all the bytes). Some vendors lack ``obs_head_pose``; episode_to_vector falls
back to first-frame normalization for them (a weaker quotient — report those
episodes as a separate stratum, never silently mixed).

Requires the optional `zarr` dependency: pip install 'motionvendi[data]'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gates import GateReport, gate_episode
from .normalize import episode_to_vector

EE_KEYS = {"left": "left.obs_ee_pose", "right": "right.obs_ee_pose"}
HEAD_KEY = "obs_head_pose"


def _read_array(root, key: str) -> np.ndarray | None:
    try:
        return np.asarray(root[key])
    except (KeyError, IndexError):
        return None


def load_episode(path: str | Path) -> dict:
    """One .zarr store -> poses + metadata dict."""
    import zarr  # optional dep

    path = Path(path)
    root = zarr.open(str(path), mode="r")
    attrs = dict(root.attrs)

    # Arrays are zero-padded past total_frames (zarr chunk padding); the tail
    # would otherwise read as a frozen/degenerate-quaternion corruption.
    tf = attrs.get("total_frames")
    def _trunc(a):
        return a[: int(tf)] if a is not None and tf else a

    head = _trunc(_read_array(root, HEAD_KEY))
    ep = {
        "path": str(path),
        "name": path.name,
        "left": _trunc(_read_array(root, EE_KEYS["left"])),
        "right": _trunc(_read_array(root, EE_KEYS["right"])),
        "head": head,
        "has_head_pose": head is not None,
        "embodiment": attrs.get("embodiment", ""),
        "task_name": attrs.get("task_name", ""),
        "task_description": attrs.get("task_description", ""),
        "fps": float(attrs.get("fps", 30.0) or 30.0),
        "total_frames": attrs.get("total_frames"),
    }
    ann = _read_array(root, "annotations")
    if ann is not None:
        try:
            ep["annotations"] = [json.loads(a) for a in ann]
        except (TypeError, ValueError):
            ep["annotations"] = None
    return ep


def iter_episode_dirs(folder: str | Path) -> list[Path]:
    """All *.zarr stores under folder (flat or one level of lab subdirs)."""
    folder = Path(folder)
    direct = sorted(folder.glob("*.zarr"))
    nested = sorted(folder.glob("*/*.zarr"))
    return direct + nested


def load_folder(
    folder: str | Path, n_steps: int = 32
) -> tuple[np.ndarray, list[dict], list[tuple[dict, GateReport]]]:
    """Folder of .zarr episodes -> (behavior matrix X, kept metadata, dropped).

    Returns only gate-passing episodes in X; dropped episodes come back with
    their full GateReport so the keep/drop list is auditable. Episode fps from
    attrs is used for the physical-limit gates.
    """
    kept_vecs, kept_meta, dropped = [], [], []
    for ep_path in iter_episode_dirs(folder):
        ep = load_episode(ep_path)
        report = gate_episode(ep["left"], ep["right"], fps=ep["fps"])
        if not report.passed:
            dropped.append((ep, report))
            continue
        vec = episode_to_vector(ep["left"], ep["right"], ep["head"], n_steps=n_steps)
        kept_vecs.append(vec)
        kept_meta.append(ep)
    X = np.asarray(kept_vecs) if kept_vecs else np.empty((0, 0))
    return X, kept_meta, dropped
