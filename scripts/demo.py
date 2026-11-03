"""Privacy-preserving demo: video or webcam in, SKELETON-ONLY video out.

The output never contains camera pixels: each person is drawn as a skeleton on a
blank canvas, coloured by the predicted risk class. This is the presentation demo.

    python scripts/demo.py --source data/demo/fall.mp4 --ckpt runs/stgcn_seed0/best.pt --out demo_out.mp4
    python scripts/demo.py --source 0 --ckpt runs/stgcn_seed0/best.pt --show      # webcam
    python scripts/demo.py --source clip.mp4 --rules --out demo_rules.mp4
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from childfall.preprocess import interpolate_missing, normalise, resample  # noqa: E402
from childfall.rules import CLASSES, classify  # noqa: E402
from childfall.skeleton import EDGES  # noqa: E402

COLORS = {0: (120, 200, 120), 1: (0, 190, 255), 2: (60, 60, 230)}  # BGR: green, orange, red
FPS_OUT = 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or webcam index")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    pose = YOLO(args.model)
    T = 90
    clf = None
    if args.ckpt:
        from childfall.models import build
        c = torch.load(args.ckpt, map_location="cpu")
        clf = build(c["model"], T=c["T"])
        clf.load_state_dict(c["state"])
        clf.eval()
        T = c["T"]

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    buf_len = int(T * fps / FPS_OUT)
    buffers, state = {}, {}
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if args.out else None

    for f_idx, r in enumerate(pose.track(src, stream=True, persist=True, verbose=False)):
        canvas = np.full((H, W, 3), 30, np.uint8)   # blank: camera pixels are discarded here
        if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
            kps = r.keypoints.data.cpu().numpy()
            for tid, kp in zip(r.boxes.id.cpu().numpy().astype(int), kps):
                b = buffers.setdefault(tid, deque(maxlen=buf_len))
                b.append(kp)
                if len(b) >= buf_len // 2 and f_idx % max(1, int(fps // 3)) == 0:
                    seq = np.stack(b)
                    s = normalise(interpolate_missing(resample(seq, fps, FPS_OUT)))
                    if s.shape[0] < T:
                        s = np.concatenate([np.repeat(s[:1], T - s.shape[0], 0), s])
                    x = np.transpose(s[-T:], (2, 0, 1)).astype(np.float32)
                    if clf is not None:
                        with torch.no_grad():
                            state[tid] = int(clf(torch.from_numpy(x)[None]).argmax(1))
                    else:
                        state[tid] = classify(x)
                cls = state.get(tid, 0)
                col = COLORS[cls]
                for i, j in EDGES:
                    if kp[i, 2] > 0.3 and kp[j, 2] > 0.3:
                        cv2.line(canvas, tuple(map(int, kp[i, :2])), tuple(map(int, kp[j, :2])), col, 3)
                for x_, y_, c_ in kp:
                    if c_ > 0.3:
                        cv2.circle(canvas, (int(x_), int(y_)), 4, col, -1)
                top = kp[kp[:, 2] > 0.3]
                if len(top):
                    cv2.putText(canvas, f"#{tid} {CLASSES[cls]}", (int(top[:, 0].min()), int(top[:, 1].min()) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if any(v == 2 for v in state.values()):
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), COLORS[2], 8)
            cv2.putText(canvas, "DANGEROUS FALL - CHECK NOW", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[2], 2)
        if writer:
            writer.write(canvas)
        if args.show:
            cv2.imshow("child fall risk (skeleton only)", canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if writer:
        writer.release()
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
