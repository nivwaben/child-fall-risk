"""COCO-17 skeleton definition shared by every stage.

Pose models (YOLO-pose, RTMPose, ViTPose) all output these 17 joints in this order.
"""
import numpy as np

JOINTS = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
V = len(JOINTS)
J = {name: i for i, name in enumerate(JOINTS)}

EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
    (0, 5), (0, 6),                            # head-shoulders
]

# Left/right pairs, used when mirroring a skeleton.
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FLIP_INDEX = list(range(V))
for a, b in FLIP_PAIRS:
    FLIP_INDEX[a], FLIP_INDEX[b] = b, a

HEAD = [0, 1, 2, 3, 4]
HIPS = [11, 12]
SHOULDERS = [5, 6]


def adjacency(normalize: bool = True) -> np.ndarray:
    """Symmetric adjacency with self-loops, D^-1/2 (A+I) D^-1/2 normalised."""
    A = np.eye(V, dtype=np.float32)
    for i, j in EDGES:
        A[i, j] = A[j, i] = 1.0
    if not normalize:
        return A
    d = A.sum(1)
    d_inv = np.diag(1.0 / np.sqrt(d))
    return (d_inv @ A @ d_inv).astype(np.float32)
