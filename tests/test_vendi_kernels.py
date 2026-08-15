"""Vendi score sanity: the metric must hit its analytic anchor points, and the
kernel machinery must uphold the PSD guarantee the score depends on."""

import numpy as np
import pytest

from motionvendi.kernels import combine_kernels, kernel_histogram_stats, rbf_kernel, validate_psd
from motionvendi.vendi import bootstrap_vendi, eigenvalue_spectrum, vendi_ratio, vendi_score


def test_identical_samples_vendi_is_one():
    K = np.ones((10, 10))
    assert vendi_score(K) == pytest.approx(1.0, abs=1e-9)


def test_orthogonal_samples_vendi_is_n():
    K = np.eye(8)
    assert vendi_score(K) == pytest.approx(8.0, abs=1e-9)


def test_two_clusters_effective_count_two():
    # 6 copies of A + 6 copies of B, A ⟂ B  =>  exactly 2 effective behaviors.
    K = np.block(
        [[np.ones((6, 6)), np.zeros((6, 6))], [np.zeros((6, 6)), np.ones((6, 6))]]
    )
    assert vendi_score(K) == pytest.approx(2.0, abs=1e-9)


def test_vendi_order_q_monotone_on_skewed_spectrum():
    # q<1 emphasizes rare modes => higher score than q>1 on a skewed spectrum.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (20, 4)), rng.normal(8, 0.05, (3, 4))])
    K = rbf_kernel(X)
    assert vendi_score(K, q=0.5) >= vendi_score(K, q=1.0) >= vendi_score(K, q=2.0)


def test_vendi_ratio_bounds():
    rng = np.random.default_rng(1)
    K = rbf_kernel(rng.normal(size=(30, 5)))
    assert 0.0 < vendi_ratio(K) <= 1.0


def test_rbf_kernel_is_psd_unit_diag():
    rng = np.random.default_rng(2)
    K = rbf_kernel(rng.normal(size=(40, 12)))
    ok, min_eig = validate_psd(K)
    assert ok, f"min eig {min_eig}"
    np.testing.assert_allclose(np.diag(K), 1.0)


def test_combine_kernels_rejects_nonconvex_weights():
    K = np.eye(3)
    with pytest.raises(ValueError):
        combine_kernels([K, K], [0.9, 0.5])


def test_combined_kernel_stays_psd():
    rng = np.random.default_rng(3)
    K1 = rbf_kernel(rng.normal(size=(25, 6)))
    K2 = rbf_kernel(rng.normal(size=(25, 6)))
    ok, _ = validate_psd(combine_kernels([K1, K2], [0.6, 0.4]))
    assert ok


def test_spectrum_sums_to_one():
    rng = np.random.default_rng(4)
    lam = eigenvalue_spectrum(rbf_kernel(rng.normal(size=(15, 3))))
    assert lam.sum() == pytest.approx(1.0)
    assert np.all(np.diff(lam) <= 1e-12)  # descending


def test_bootstrap_vendi_fixed_size_comparable():
    rng = np.random.default_rng(5)
    diverse = rng.normal(size=(60, 8))
    redundant = np.tile(rng.normal(size=(3, 8)), (20, 1)) + rng.normal(0, 0.01, (60, 8))
    kb = lambda X: rbf_kernel(X, sigma2=4.0)
    d = bootstrap_vendi(diverse, kb, sample_size=30, n_boot=20)
    r = bootstrap_vendi(redundant, kb, sample_size=30, n_boot=20)
    assert d["mean"] > r["mean"]  # size-fair ranking: diverse > redundant


def test_kernel_histogram_stats_keys():
    K = rbf_kernel(np.random.default_rng(6).normal(size=(10, 2)))
    stats = kernel_histogram_stats(K)
    assert set(stats) == {"mean", "p05", "p50", "p95"}
