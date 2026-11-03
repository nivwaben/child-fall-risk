"""Generate SYNTHETIC skeleton clips for smoke-testing the pipeline.

These are hand-scripted stick-figure motions, not real data. Use them only to
check that code runs end to end before the real datasets are ready. Never
report results on them.

    python scripts/make_synthetic.py --out data_synth --n 300
"""
import argparse
import csv
from pathlib import Path

import numpy as np

# Standing template in pixels, hip midpoint at (0, 0), y down, torso ~100 px.
TEMPLATE = np.array([
    [0, -170], [-8, -178], [8, -178], [-16, -172], [16, -172],
    [-40, -100], [40, -100], [-45, -50], [45, -50], [-48, -5], [48, -5],
    [-20, 0], [20, 0], [-22, 90], [22, 90], [-22, 180], [22, 180],
], dtype=np.float32)
LEGS = [13, 14, 15, 16]
FLOOR_LYING_HIP = 165.0  # hip y when lying on the floor


def rot(p, deg):
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], np.float32)
    return p @ R.T


def smooth_step(n):
    s = np.linspace(0, 1, n)
    return s * s * (3 - 2 * s)


def render(angle, hip_x, hip_y, leg_bend, rng, wiggle):
    T = len(angle)
    seq = np.zeros((T, 17, 3), np.float32)
    for t in range(T):
        p = TEMPLATE.copy()
        p[LEGS, 1] *= 1 - leg_bend[t]
        p += rng.normal(0, wiggle[t], p.shape)
        p = rot(p, angle[t])
        p[:, 0] += hip_x[t]
        p[:, 1] += hip_y[t]
        seq[t, :, :2] = p
        seq[t, :, 2] = np.clip(rng.normal(0.9, 0.05, 17), 0, 1)
    return seq


def make_clip(kind, rng, fps=30, seconds=8.0):
    T = int(fps * seconds)
    angle = np.zeros(T)
    hip_x = np.cumsum(rng.normal(0, 0.5, T)) + rng.uniform(-50, 50)
    hip_y = np.zeros(T)
    bend = np.zeros(T)
    wiggle = np.full(T, 2.0)
    side = rng.choice([-1, 1])
    t0 = int(fps * rng.uniform(2.5, 3.8))

    if kind == "normal":
        act = rng.choice(["walk", "jump", "squat", "sit", "crawl_roll"])
        if act == "walk":
            hip_x += side * np.linspace(0, rng.uniform(100, 300), T)
            hip_y += 5 * np.sin(np.linspace(0, 16 * np.pi, T))
        elif act == "jump":
            for k in range(rng.integers(1, 4)):
                s = t0 + k * fps
                n = int(0.5 * fps)
                if s + n < T:
                    hip_y[s:s + n] -= 60 * np.sin(np.linspace(0, np.pi, n))
        elif act == "squat":
            n = int(rng.uniform(0.8, 1.5) * fps)
            bend[t0:t0 + n] = 0.45 * smooth_step(n)
            bend[t0 + n:] = 0.45
            hip_y = bend / 0.45 * 80
        elif act == "sit":
            n = int(rng.uniform(1.0, 2.0) * fps)
            prof = np.concatenate([np.zeros(t0), smooth_step(n), np.ones(T - t0 - n)])
            hip_y = prof * 90
            bend = prof * 0.5
            angle = prof * rng.uniform(5, 20) * side
        else:  # slow deliberate roll on a mat: goes down slowly, keeps moving, gets up
            n = int(rng.uniform(1.5, 2.5) * fps)
            down = np.concatenate([np.zeros(t0), smooth_step(n), np.ones(T - t0 - n)])
            up_s = min(T - 1, t0 + n + int(2 * fps))
            down[up_s:] = 1 - smooth_step(T - up_s)
            angle = down * 85 * side
            hip_y = down * FLOOR_LYING_HIP
            wiggle = 2 + 6 * down
        return render(angle, hip_x, hip_y, bend, rng, wiggle)

    # falls
    start_h = 0.0
    head_first = False
    if kind == "dangerous_fall":
        start_h = -rng.choice([0, 0, 80, 120])          # sometimes from a platform / slide
        head_first = rng.random() < 0.35
    hip_y[:] = start_h
    n_fall = int(fps * (rng.uniform(0.25, 0.35) if kind == "dangerous_fall" else rng.uniform(0.35, 0.5)))
    final_angle = (rng.uniform(120, 150) if head_first else rng.uniform(75, 95)) * side
    prof = smooth_step(n_fall)
    angle[t0:t0 + n_fall] = prof * final_angle
    hip_y[t0:t0 + n_fall] = start_h + prof * (FLOOR_LYING_HIP - start_h)
    angle[t0 + n_fall:] = final_angle
    hip_y[t0 + n_fall:] = FLOOR_LYING_HIP
    if kind == "light_fall":
        lie = int(fps * rng.uniform(0.5, 1.5))
        wiggle[t0 + n_fall:t0 + n_fall + lie] = 8.0
        s = t0 + n_fall + lie
        n_up = int(fps * rng.uniform(0.8, 1.3))
        e = min(T, s + n_up)
        up = smooth_step(e - s)
        angle[s:e] = final_angle * (1 - up)
        hip_y[s:e] = FLOOR_LYING_HIP * (1 - up)
        angle[e:] = 0
        hip_y[e:] = 0
        wiggle[e:] = 2.0
    else:
        wiggle[t0 + n_fall:] = 0.3                          # lies still
    return render(angle, hip_x, hip_y, bend, rng, wiggle)


def to_camera(seq, rng, W=640, H=480):
    s = rng.uniform(0.6, 1.3)
    out = seq.copy()
    out[..., :2] *= s
    out[..., 0] += W / 2 + rng.uniform(-100, 100)
    out[..., 1] += H / 2 - 30 * s + rng.uniform(-20, 20)
    # random dropouts like a real pose estimator
    drop = rng.random(out.shape[:2]) < 0.04
    out[drop, 2] = rng.uniform(0, 0.2, drop.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "poses").mkdir(parents=True, exist_ok=True)
    kinds = ["normal", "light_fall", "dangerous_fall"]
    rows = []
    n_subjects = 12
    for i in range(args.n):
        kind = kinds[i % 3]
        subj = int(rng.integers(n_subjects))
        seq = to_camera(make_clip(kind, rng), rng)
        cid = f"syn_{i:04d}"
        np.savez_compressed(out / "poses" / f"{cid}.npz", keypoints=seq, fps=np.float32(30))
        split = "train" if subj < 8 else ("val" if subj < 10 else "test")
        rows.append(dict(clip_id=cid, label=kind, subject=f"s{subj}", source="synthetic", split=split))
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} synthetic clips to {out}")


if __name__ == "__main__":
    main()
