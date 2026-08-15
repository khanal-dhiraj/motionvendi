"""Gates must catch every planted corruption type; curation must prefer
diversity. Ground truth comes from the synthetic generator."""

import numpy as np

from motionvendi.curate import greedy_max_vendi, nearest_duplicates, redundancy_ranking
from motionvendi.gates import gate_episode, run_gates
from motionvendi.synthetic import (
    CORRUPTIONS,
    make_corpus,
    make_episode,
)


def _clean_episode(seed=0):
    return make_episode("reach", np.random.default_rng(seed))


def test_clean_episode_passes_all_gates():
    ep = _clean_episode()
    report = gate_episode(ep["left"], ep["right"])
    assert report.passed, report.reasons


def test_every_corruption_type_is_caught():
    rng = np.random.default_rng(1)
    for corrupt in CORRUPTIONS:
        ep = corrupt(_clean_episode(seed=2), rng)
        report = gate_episode(ep["left"], ep["right"])
        assert not report.passed, f"{ep['corruption']} slipped through"


def test_short_segment_rejected():
    ep = _clean_episode()
    report = run_gates(ep["right"][:10])
    assert not report.passed and "min_length" in report.reasons


def test_missing_hand_is_not_a_failure():
    ep = _clean_episode()
    report = gate_episode(None, ep["right"])
    assert report.passed


def test_greedy_max_vendi_picks_diverse_first():
    # 5 near-identical episodes + 1 outlier: the outlier must be picked within
    # the first two selections (it carries all the marginal diversity).
    K = np.full((6, 6), 0.98)
    K[5, :] = K[:, 5] = 0.02
    np.fill_diagonal(K, 1.0)
    order, scores = greedy_max_vendi(K, budget=3)
    assert 5 in order[:2]
    # NOTE: Vendi is NOT monotone under greedy growth — adding a redundant
    # sample can lower the effective count. We only require the first two
    # picks to capture both clusters (score ~2 with outlier + any clone).
    assert scores[1] > 1.9


def test_redundancy_ranking_puts_duplicates_last():
    K = np.eye(4)
    K[0, 1] = K[1, 0] = 0.999  # 0 and 1 are near-duplicates
    ranking = redundancy_ranking(K)  # most-redundant-first
    assert ranking[0] in (0, 1)


def test_nearest_duplicates_finds_planted_pair():
    K = np.eye(5)
    K[2, 4] = K[4, 2] = 0.97
    pairs = nearest_duplicates(K, top_k=1)
    assert pairs[0][:2] == (2, 4)


def test_corpus_ground_truth_labels_present():
    corpus = make_corpus(n_per_family=3, n_duplicate_pairs=2, n_corrupt=4, n_frames=120)
    assert sum("corruption" in ep for ep in corpus) == 4
    assert sum("is_duplicate_of" in ep for ep in corpus) == 2
    assert all("family" in ep for ep in corpus)
