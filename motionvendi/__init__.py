"""MotionVendi: gated motion-diversity measurement and curation for egocentric
robot-learning data.

Pipeline: gates (remove measurement lies) -> normalize (quotient nuisance) ->
kernel (define 'same behavior') -> Vendi (effective count) -> curate (greedy
max-diversity keep-list).
"""

from .curate import curation_curve, greedy_max_vendi, nearest_duplicates, redundancy_ranking
from .gates import GateReport, gate_episode, run_gates
from .kernels import combine_kernels, kernel_histogram_stats, median_bandwidth, rbf_kernel, validate_psd
from .normalize import episode_to_vector, pose_row_to_matrix, quat_wxyz_to_matrix, resample_uniform, to_head_frame
from .vendi import bootstrap_vendi, eigenvalue_spectrum, vendi_ratio, vendi_score

__all__ = [
    "GateReport",
    "bootstrap_vendi",
    "combine_kernels",
    "curation_curve",
    "eigenvalue_spectrum",
    "episode_to_vector",
    "gate_episode",
    "greedy_max_vendi",
    "kernel_histogram_stats",
    "median_bandwidth",
    "nearest_duplicates",
    "pose_row_to_matrix",
    "quat_wxyz_to_matrix",
    "rbf_kernel",
    "redundancy_ranking",
    "resample_uniform",
    "run_gates",
    "to_head_frame",
    "validate_psd",
    "vendi_ratio",
    "vendi_score",
]
