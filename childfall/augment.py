"""Skeleton augmentations. All take and return (3, T, V) normalised arrays.

These double as the ROBUSTNESS TEST conditions: apply one at test time with a
fixed strength and measure how much each model drops.
"""
import numpy as np

from .skeleton import FLIP_INDEX, HEAD, HIPS, SHOULDERS


def rotate(x: np.ndarray, deg: float) -> np.ndarray:
    """In-plane rotation: a crude stand-in for camera roll / tilt."""
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    out = x.copy()
    xy = np.einsum("ij,jtv->itv", R, x[:2])
    out[:2] = np.where(x[2:3] > 0, xy, 0)
    return out


def view_squash(x: np.ndarray, factor: float) -> np.ndarray:
    """Compress the horizontal axis: approximates a camera seeing the body from the side."""
    out = x.copy()
    out[0] *= factor
    return out


def flip(x: np.ndarray) -> np.ndarray:
    out = x[:, :, FLIP_INDEX].copy()
    out[0] *= -1
    return out


def occlude(x: np.ndarray, frac_joints: float, rng: np.random.Generator, contiguous: bool = True) -> np.ndarray:
    """Drop a fraction of joints (set to 0, conf 0).

    contiguous=True drops a whole body region for a block of frames, which is what
    happens when another child walks in front of the camera.
    """
    out = x.copy()
    _, T, V = x.shape
    k = max(1, int(round(frac_joints * V)))
    joints = rng.choice(V, size=k, replace=False)
    if contiguous:
        length = rng.integers(T // 4, T // 2 + 1)
        s = rng.integers(0, T - length + 1)
        out[:, s:s + length, joints] = 0.0
    else:
        mask = rng.random((T, k)) < 0.5
        for i, j in enumerate(joints):
            out[:, mask[:, i], j] = 0.0
    return out


def low_resolution(x: np.ndarray, noise_std: float, conf_drop: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate a small / far-away person: keypoint jitter plus lower confidence."""
    out = x.copy()
    vis = out[2] > 0
    out[:2] += rng.normal(0, noise_std, out[:2].shape).astype(np.float32) * vis
    out[2] = np.where(vis, np.clip(out[2] - conf_drop, 0.01, 1), 0)
    return out


def child_proportions(x: np.ndarray, limb_scale: float = 0.8, head_scale: float = 1.25) -> np.ndarray:
    """Re-shape an adult skeleton toward child proportions.

    Children have a larger head relative to body and shorter limbs. This moves
    head joints away from the shoulder midpoint and pulls limbs toward the torso.
    Used for the adult->child domain-gap experiment (simulated shift, NOT real child data).
    """
    out = x.copy()
    vis = out[2] > 0
    sho = out[:2, :, SHOULDERS].mean(-1, keepdims=True)
    hip = out[:2, :, HIPS].mean(-1, keepdims=True)
    for j in HEAD:
        out[:2, :, j] = sho[..., 0] + (out[:2, :, j] - sho[..., 0]) * head_scale
    # elbows/wrists from shoulders, knees/ankles from hips
    for parent, kids in ((5, (7, 9)), (6, (8, 10)), (11, (13, 15)), (12, (14, 16))):
        for k in kids:
            out[:2, :, k] = out[:2, :, parent] + (out[:2, :, k] - out[:2, :, parent]) * limb_scale
    out[:2] = np.where(vis, out[:2], 0)
    _ = hip
    return out


def random_train_aug(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation used during training."""
    if rng.random() < 0.5:
        x = flip(x)
    x = rotate(x, rng.uniform(-15, 15))
    x = x * np.array([rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1), 1.0], dtype=np.float32)[:, None, None]
    if rng.random() < 0.3:
        x = occlude(x, rng.uniform(0.1, 0.3), rng)
    if rng.random() < 0.3:
        x = low_resolution(x, 0.03, 0.1, rng)
    return x.astype(np.float32)


# Fixed test-time conditions for the robustness experiment.
ROBUSTNESS_CONDITIONS = {
    "clean": lambda x, rng: x,
    "rotate_30": lambda x, rng: rotate(x, 30),
    "side_view_0.5": lambda x, rng: view_squash(x, 0.5),
    "occlude_20pct": lambda x, rng: occlude(x, 0.2, rng),
    "occlude_40pct": lambda x, rng: occlude(x, 0.4, rng),
    "low_res": lambda x, rng: low_resolution(x, 0.08, 0.25, rng),
    "child_shape": lambda x, rng: child_proportions(x),
}
