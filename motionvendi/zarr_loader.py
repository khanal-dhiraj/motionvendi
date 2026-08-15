"""Load real EgoVerse episodes (folder of per-episode .zarr stores).

Reads ONLY the pose arrays — `left/right.obs_ee_pose`, `obs_head_pose` — and
skips `images.front_1`, which is nearly all the bytes. Pose rows are
[x,y,z,qw,qx,qy,qz] (wxyz quaternion).

Scale-vendor episodes have no `obs_head_pose`; episode_to_vector falls back to
first-frame normalization for them (a weaker quotient — report them as a
separate stratum, never silently mixed).

Requires the optional `zarr` dependency: pip install 'motionvendi[data]'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gates import GateReport, gate_episode
from .normalize import episode_to_vector

EE_KEYS = {"left": "left/obs_ee_pose", "right": "right/obs_ee_pose"}
HEAD_KEY = "obs_head_pose"


def _read_array(root, key: str) -> np.ndarray | None:
    try:
        node = root
        for part in key.split("/"):
            node = node[part]
        return np.asarray(node)
    except (KeyError, IndexError):
        return None


def load_episode(path: str | Path) -> dict:
    """One .zarr store -> {left, right, head, embodiment, annotations}."""
    import zarr  # optional dep

    path = Path(path)
    root = zarr.open(str(path), mode="r")
    attrs = dict(root.attrs)
    ep = {
        "path": str(path),
        "left": _read_array(root, EE_KEYS["left"]),
        "right": _read_array(root, EE_KEYS["right"]),
        "head": _read_array(root, HEAD_KEY),
        "embodiment": attrs.get("embodiment", ""),
        "has_head_pose": _read_array(root, HEAD_KEY) is not None,
    }
    ann = _read_array(root, "annotations")
    if ann is not None:
        try:
            ep["annotations"] = [json.loads(a) for a in ann]
        except (TypeError, ValueError):
            ep["annotations"] = None
    return ep


def load_folder(
    folder: str | Path, n_steps: int = 32, fps: float = 30.0
) -> tuple[np.ndarray, list[dict], list[tuple[str, GateReport]]]:
    """Folder of .zarr episodes -> (behavior matrix X, kept metadata, dropped).

    Returns only gate-passing episodes in X; dropped episodes come back with
    their full GateReport so the keep/drop list is auditable.
    """
    folder = Path(folder)
    kept_vecs, kept_meta, dropped = [], [], []
    for ep_path in sorted(folder.glob("*.zarr")):
        ep = load_episode(ep_path)
        report = gate_episode(ep["left"], ep["right"], fps=fps)
        if not report.passed:
            dropped.append((str(ep_path), report))
            continue
        vec = episode_to_vector(ep["left"], ep["right"], ep["head"], n_steps=n_steps)
        kept_vecs.append(vec)
        kept_meta.append(ep)
    X = np.asarray(kept_vecs) if kept_vecs else np.empty((0, 0))
    return X, kept_meta, dropped
