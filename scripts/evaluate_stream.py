"""Streaming evaluation on LONG videos: false alarms per hour and alert latency.

Inputs
  data/stream/poses/<video>.npz        long pose sequences (from extract_pose.py)
  data/stream/events.csv               video, t_start, t_end   (dangerous falls, seconds)
                                       videos with no dangerous fall still need to be
                                       listed in data/stream/videos.txt

    python scripts/evaluate_stream.py --root data/stream --ckpt runs/stgcn_seed0/best.pt
    python scripts/evaluate_stream.py --root data/stream --rules
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from childfall.metrics import alarms_from_stream, stream_metrics  # noqa: E402
from childfall.preprocess import sliding_windows  # noqa: E402
from childfall.rules import predict_batch  # noqa: E402

FPS_OUT = 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/stream")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--stride", type=int, default=15, help="frames at 15 fps between windows (15 = 1 s)")
    ap.add_argument("--refractory", type=float, default=10.0)
    args = ap.parse_args()

    root = Path(args.root)
    events = {}
    if (root / "events.csv").exists():
        for r in csv.DictReader(open(root / "events.csv")):
            events.setdefault(r["video"], []).append((float(r["t_start"]), float(r["t_end"])))
    videos = [l.strip() for l in open(root / "videos.txt") if l.strip()]

    if args.ckpt:
        c = torch.load(args.ckpt, map_location="cpu")
        from childfall.models import build
        model = build(c["model"], T=c["T"])
        model.load_state_dict(c["state"])
        model.eval()
        T = c["T"]
        predict = lambda X: model(torch.from_numpy(X)).argmax(1).numpy()
    else:
        T = 90
        predict = predict_batch

    all_alarms, all_events, total_s = [], [], 0.0
    offset = 0.0
    for v in videos:
        z = np.load(root / "poses" / f"{v}.npz")
        dur = z["keypoints"].shape[0] / float(z["fps"])
        wins = list(sliding_windows(z["keypoints"], float(z["fps"]), T, args.stride, FPS_OUT))
        times = [t for t, _ in wins]
        with torch.no_grad():
            preds = predict(np.stack([w for _, w in wins]))
        all_alarms += [offset + t for t in alarms_from_stream(times, preds, args.refractory)]
        all_events += [(offset + s, offset + e) for s, e in events.get(v, [])]
        offset += dur + 60.0  # gap so events never match across videos
        total_s += dur

    m = stream_metrics(all_alarms, all_events, total_s, window_s=T / FPS_OUT)
    m["hours"] = total_s / 3600
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
