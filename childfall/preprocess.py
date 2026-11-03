"""Turn raw pose output into fixed-size, normalised model input.

Raw pose array: (T, 17, 3) = x, y in pixels, confidence in [0, 1].
Model input:    (3, T_win, 17) = normalised x, y, confidence.
"""
import numpy as np

from .skeleton import HIPS, SHOULDERS

CONF_MIN = 0.3  # joints below this confidence are treated as missing


def interpolate_missing(seq: np.ndarray, conf_min: float = CONF_MIN, max_gap: int = 8) -> np.ndarray:
    """Fill low-confidence joints by linear interpolation over time (gaps <= max_gap frames).

    Longer gaps stay missing (x = y = 0, conf = 0) so the model can see the occlusion.
    """
    seq = seq.copy()
    T, V, _ = seq.shape
    for v in range(V):
        ok = seq[:, v, 2] >= conf_min
        if ok.all() or not ok.any():
            if not ok.any():
                seq[:, v, :] = 0.0
            continue
        idx = np.arange(T)
        good = idx[ok]
        for c in (0, 1):
            seq[~ok, v, c] = np.interp(idx[~ok], good, seq[ok, v, c])
        # distance (in frames) of every missing frame to the nearest good frame
        dist = np.min(np.abs(idx[:, None] - good[None, :]), axis=1)
        too_far = (~ok) & (dist > max_gap // 2)
        seq[too_far, v, :] = 0.0
        filled = (~ok) & ~too_far
        seq[filled, v, 2] = conf_min  # mark as interpolated, low but non-zero
    return seq


def normalise(seq: np.ndarray, ref_frames: int = 15) -> np.ndarray:
    """Centre on the hip midpoint of the first frames and scale by torso length.

    Uses a reference taken from the START of the clip (not per frame) so that a
    fall still shows up as a large downward movement after normalisation.
    Image y grows downward, so a fall is a positive change in y.
    """
    seq = seq.astype(np.float32).copy()
    ref = seq[:ref_frames]
    valid = ref[..., 2] > 0
    hip = _mean_joint(ref, HIPS, valid)
    sho = _mean_joint(ref, SHOULDERS, valid)
    torso = np.linalg.norm(hip - sho)
    if not np.isfinite(torso) or torso < 1e-3:
        # fallback: bounding-box height of all visible joints
        pts = seq[seq[..., 2] > 0][:, :2]
        torso = (pts[:, 1].max() - pts[:, 1].min()) / 3 if len(pts) else 1.0
        hip = pts.mean(0) if len(pts) else np.zeros(2)
    out = seq.copy()
    out[..., :2] = (seq[..., :2] - hip) / max(torso, 1e-3)
    out[seq[..., 2] == 0, :2] = 0.0
    return out


def _mean_joint(seq, joints, valid):
    pts = [seq[:, j, :2][valid[:, j]] for j in joints]
    pts = np.concatenate([p for p in pts if len(p)], 0) if any(len(p) for p in pts) else np.full((1, 2), np.nan)
    return pts.mean(0)


def resample(seq: np.ndarray, fps_in: float, fps_out: float = 15.0) -> np.ndarray:
    """Resample in time to a common frame rate (nearest frame)."""
    if abs(fps_in - fps_out) < 1e-6:
        return seq
    T = seq.shape[0]
    n = max(1, int(round(T * fps_out / fps_in)))
    idx = np.clip(np.round(np.linspace(0, T - 1, n)).astype(int), 0, T - 1)
    return seq[idx]


def fix_length(seq: np.ndarray, T: int, mode: str = "end") -> np.ndarray:
    """Crop or pad (by repeating the last frame) to exactly T frames.

    mode="end" keeps the last T frames, which keeps post-fall immobility in view.
    """
    n = seq.shape[0]
    if n >= T:
        if mode == "center":
            s = (n - T) // 2
        elif mode == "start":
            s = 0
        else:
            s = n - T
        return seq[s:s + T]
    pad = np.repeat(seq[-1:], T - n, axis=0)
    return np.concatenate([seq, pad], 0)


def to_model_input(seq: np.ndarray, fps: float, T: int = 90, fps_out: float = 15.0) -> np.ndarray:
    """Full pipeline: (T_raw, 17, 3) pixels -> (3, T, 17) float32."""
    s = resample(seq, fps, fps_out)
    s = interpolate_missing(s)
    s = normalise(s)
    s = fix_length(s, T)
    return np.transpose(s, (2, 0, 1)).astype(np.float32)


def sliding_windows(seq: np.ndarray, fps: float, T: int = 90, stride: int = 15, fps_out: float = 15.0):
    """Yield (start_time_s, model_input) over a long video for streaming evaluation."""
    s = resample(seq, fps, fps_out)
    for start in range(0, max(1, s.shape[0] - T + 1), stride):
        win = s[start:start + T]
        win = interpolate_missing(win)
        win = normalise(win)
        win = fix_length(win, T)
        yield start / fps_out, np.transpose(win, (2, 0, 1)).astype(np.float32)
