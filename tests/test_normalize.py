"""Pins the pose conventions. Highest bug-severity-per-minute tests in the repo:
misreading the wxyz quaternion produces plausible-looking garbage with no error.
"""

import numpy as np
import pytest

from motionvendi.normalize import (
    episode_to_vector,
    matrix_to_quat_wxyz,
    pose_row_to_matrix,
    quat_wxyz_to_matrix,
    resample_uniform,
    rotmat_to_6d,
    to_first_frame,
    to_head_frame,
)


def _random_unit_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_quat_wxyz_identity():
    # w=1 first => identity rotation. If the code read xyzw, [1,0,0,0] would be
    # a 180-degree rotation about x instead — this is THE convention pin.
    R = quat_wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_quat_wxyz_known_rotation():
    # 90 deg about z in wxyz: [cos45, 0, 0, sin45]
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    R = quat_wxyz_to_matrix(q)
    np.testing.assert_allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = _random_unit_quat(rng)
        q = q if q[0] >= 0 else -q  # canonical sign
        q2 = matrix_to_quat_wxyz(quat_wxyz_to_matrix(q))
        np.testing.assert_allclose(q2, q, atol=1e-9)


def test_zero_quat_yields_nan_not_identity():
    R = quat_wxyz_to_matrix(np.zeros(4))
    assert np.isnan(R).all()


def test_pose_row_to_matrix_places_translation():
    row = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    T = pose_row_to_matrix(row)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)


def test_pose_row_rejects_wrong_dim():
    with pytest.raises(ValueError):
        pose_row_to_matrix(np.zeros((5, 6)))


def test_head_frame_kills_world_offset():
    # Same relative motion recorded in two different world frames must produce
    # identical head-frame trajectories — the whole point of the quotient.
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 50)
    rel = np.stack([0.3 * t, 0.1 * np.sin(2 * np.pi * t), 0.05 * t], axis=1)

    def make(world_offset, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        Rw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        qw = matrix_to_quat_wxyz(Rw)
        head = np.zeros((50, 7))
        head[:, :3] = world_offset
        head[:, 3:7] = qw
        ee = np.zeros((50, 7))
        ee[:, :3] = rel @ Rw.T + world_offset
        ee[:, 3:7] = qw
        return ee, head

    ee1, head1 = make(np.array([0.0, 0.0, 0.0]), 0.0)
    ee2, head2 = make(rng.uniform(-5, 5, 3), 2.1)
    T1 = to_head_frame(ee1, head1)
    T2 = to_head_frame(ee2, head2)
    np.testing.assert_allclose(T1, T2, atol=1e-9)


def test_first_frame_fallback_translation_invariant():
    ee = np.zeros((30, 7))
    ee[:, 0] = np.linspace(0, 1, 30)
    ee[:, 3] = 1.0
    shifted = ee.copy()
    shifted[:, :3] += np.array([10.0, -4.0, 2.0])
    np.testing.assert_allclose(to_first_frame(ee), to_first_frame(shifted), atol=1e-9)


def test_rotmat_to_6d_shape():
    R = np.tile(np.eye(3), (7, 1, 1))
    assert rotmat_to_6d(R).shape == (7, 6)


def test_resample_speed_invariance():
    # Same path traversed at two speeds -> same resampled shape (approx).
    t_slow = np.linspace(0, 1, 400)
    t_fast = np.linspace(0, 1, 90)
    path = lambda t: np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1)
    a = resample_uniform(path(t_slow), 32)
    b = resample_uniform(path(t_fast), 32)
    np.testing.assert_allclose(a, b, atol=5e-3)


def test_arclength_resample_time_warp_invariance():
    # Same spatial path traversed with a nonlinear speed profile (time warp)
    # must produce the same arc-length-parameterized vector. Uniform-in-time
    # resampling FAILS this — it's why the quotient is arc length.
    from motionvendi.normalize import resample_arclength

    t_uniform = np.linspace(0, 1, 300)
    t_warped = np.linspace(0, 1, 300) ** 1.7
    path = lambda t: np.stack(
        [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), 0.3 * t], axis=1
    )
    a = resample_arclength(path(t_uniform), 32)
    b = resample_arclength(path(t_warped), 32)
    np.testing.assert_allclose(a, b, atol=2e-2)


def test_arclength_resample_stationary_falls_back():
    from motionvendi.normalize import resample_arclength

    X = np.tile(np.array([1.0, 2.0, 3.0, 0.5]), (50, 1))
    out = resample_arclength(X, 8)
    assert out.shape == (8, 4) and np.all(np.isfinite(out))


def test_resample_too_few_valid_frames_is_nan():
    X = np.full((10, 3), np.nan)
    X[0] = 0
    assert np.isnan(resample_uniform(X, 8)).all()


def test_episode_vector_missing_hand_is_nan_not_zero():
    rng = np.random.default_rng(2)
    ee = np.zeros((60, 7))
    ee[:, 0] = np.linspace(0, 0.5, 60)
    ee[:, 3] = 1.0
    vec = episode_to_vector(None, ee, None, n_steps=16)
    left_block = vec[: 16 * 9]
    right_block = vec[16 * 9 :]
    assert np.isnan(left_block).all()
    assert np.isfinite(right_block).all()
