"""Rule-based baseline (no learning).

Works on the normalised (3, T, V) input, where units are torso lengths and
y grows downward. Thresholds are starting values: tune them on the TRAIN split only.
"""
from dataclasses import dataclass

import numpy as np

from .skeleton import HEAD, HIPS, SHOULDERS

NORMAL, LIGHT, DANGEROUS = 0, 1, 2
CLASSES = ["normal", "light_fall", "dangerous_fall"]


@dataclass
class RuleConfig:
    fps: float = 15.0
    drop_thresh: float = 1.0       # hip must drop at least this many torso lengths
    speed_thresh: float = 2.5      # peak downward hip speed (torso lengths / s)
    horiz_ratio: float = 1.0       # after the fall: torso width / height > this = lying
    still_seconds: float = 3.0     # immobility after the fall that makes it dangerous (matches labeling guide)
    still_motion: float = 0.3      # mean joint speed (torso lengths / s) counted as "still"
    recover_drop: float = 0.5      # hip back within this of start height = got up


def _joint_mean(x, joints):
    xy = x[:2, :, joints]
    vis = x[2, :, joints] > 0          # (len(joints), T)
    vis = vis.T[None]                   # (1, T, len)
    s = (xy * vis).sum(-1)
    n = np.maximum(vis.sum(-1), 1)
    return s / n                        # (2, T)


def features(x: np.ndarray, cfg: RuleConfig = RuleConfig()) -> dict:
    hip = _joint_mean(x, HIPS)
    sho = _joint_mean(x, SHOULDERS)
    head = _joint_mean(x, HEAD)
    hip_y = hip[1]
    start = np.median(hip_y[: max(3, len(hip_y) // 6)])
    v = np.gradient(hip_y) * cfg.fps
    peak_i = int(np.argmax(v))
    drop = hip_y.max() - start
    fall_i = int(np.argmax(hip_y > start + cfg.drop_thresh)) if (hip_y > start + cfg.drop_thresh).any() else peak_i
    torso_vec = sho - hip
    lying = np.abs(torso_vec[0]) / (np.abs(torso_vec[1]) + 1e-3)
    joint_speed = np.linalg.norm(np.diff(x[:2], axis=1), axis=0).mean(-1) * cfg.fps  # (T-1,)
    after = slice(min(fall_i + int(cfg.fps * 0.5), len(joint_speed) - 1), None)
    still_frames = int((joint_speed[after] < cfg.still_motion).sum())
    head_below_hip_at_fall = bool(head[1, min(fall_i + 2, len(hip_y) - 1)] > hip_y[min(fall_i + 2, len(hip_y) - 1)])
    end_drop = hip_y[-int(cfg.fps):].mean() - start
    return dict(drop=drop, peak_speed=v.max(), lying_end=lying[-int(cfg.fps):].mean(),
                still_s=still_frames / cfg.fps, head_first=head_below_hip_at_fall, end_drop=end_drop)


def classify(x: np.ndarray, cfg: RuleConfig = RuleConfig()) -> int:
    f = features(x, cfg)
    fell = f["drop"] > cfg.drop_thresh and f["peak_speed"] > cfg.speed_thresh
    if not fell:
        return NORMAL
    got_up = f["end_drop"] < cfg.recover_drop
    if got_up and not f["head_first"]:
        return LIGHT
    if f["still_s"] >= cfg.still_seconds or f["head_first"]:
        return DANGEROUS
    return LIGHT


def predict_batch(X: np.ndarray, cfg: RuleConfig = RuleConfig()) -> np.ndarray:
    return np.array([classify(x, cfg) for x in X])
