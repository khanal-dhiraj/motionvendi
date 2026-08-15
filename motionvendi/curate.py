"""Curation = keep the smallest subset that preserves effective diversity.

Greedy max-Vendi selection: repeatedly add the episode that most increases the
Vendi score of the kept set. Because the marginal gain of an episode shrinks
as similar episodes are added (diminishing returns), greedy selection yields
the classic near-optimal curation curve: "X% of the data holds Y% of the
behavioral diversity."
"""

from __future__ import annotations

import numpy as np

from .vendi import vendi_score


def greedy_max_vendi(
    K: np.ndarray, budget: int, seed_idx: int | None = None
) -> tuple[list[int], list[float]]:
    """Greedily select up to `budget` rows of kernel K maximizing Vendi.

    Returns (selected indices in pick order, Vendi score after each pick).
    O(budget * n * s^3) — fine for demo-scale n (few thousand).
    """
    K = np.asarray(K, dtype=np.float64)
    n = len(K)
    budget = min(budget, n)
    if seed_idx is None:
        # start from the episode least similar to everything (most informative alone)
        seed_idx = int(np.argmin(K.sum(axis=1)))
    selected = [seed_idx]
    scores = [1.0]  # VS of a single sample is 1 by definition
    remaining = set(range(n)) - {seed_idx}
    while len(selected) < budget and remaining:
        best_gain, best_j, best_score = -np.inf, None, None
        base = selected
        for j in remaining:
            idx = base + [j]
            s = vendi_score(K[np.ix_(idx, idx)])
            if s - scores[-1] > best_gain:
                best_gain, best_j, best_score = s - scores[-1], j, s
        selected.append(best_j)
        scores.append(best_score)
        remaining.discard(best_j)
    return selected, scores


def curation_curve(
    K: np.ndarray, fractions: list[float] | None = None
) -> dict:
    """Vendi retained vs fraction of episodes kept (greedy order).

    The headline curation deliverable: keep-list + 'diversity retained' curve.
    """
    n = len(K)
    fractions = fractions or [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    full = vendi_score(K)
    order, scores = greedy_max_vendi(K, budget=n)
    # Vendi is not monotone in subset size: a well-chosen subset can out-score
    # the redundancy-dragged full corpus. Normalize by the max achieved along
    # the greedy path so "retained" is a fraction in [0, 1].
    peak = max(scores)
    curve = []
    for f in fractions:
        k = max(1, int(round(f * n)))
        curve.append(
            {
                "fraction_kept": f,
                "episodes_kept": k,
                "vendi": scores[min(k, len(scores)) - 1],
                "vendi_retained_frac": scores[min(k, len(scores)) - 1] / peak,
            }
        )
    return {"full_vendi": full, "peak_vendi": peak, "greedy_order": order, "curve": curve}


def redundancy_ranking(K: np.ndarray) -> np.ndarray:
    """Rank episodes most-redundant-first: reverse of greedy pick order.

    The tail of the greedy order contributes the least marginal diversity —
    those are the drop candidates for the keep/drop recommendation.
    """
    order, _ = greedy_max_vendi(K, budget=len(K))
    return np.asarray(order[::-1])


def nearest_duplicates(K: np.ndarray, top_k: int = 10) -> list[tuple[int, int, float]]:
    """Most-similar episode pairs — human-inspectable near-duplicate evidence."""
    Kc = K.copy().astype(np.float64)
    np.fill_diagonal(Kc, -np.inf)
    pairs = []
    seen = set()
    flat = np.argsort(Kc, axis=None)[::-1]
    for f in flat:
        i, j = divmod(int(f), len(K))
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((key[0], key[1], float(K[i, j])))
        if len(pairs) >= top_k:
            break
    return pairs
