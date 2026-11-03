"""Stage 1: video -> pose keypoints. Raw frames are never written to disk.

    python scripts/extract_pose.py --videos data/videos --out data/poses
    python scripts/extract_pose.py --videos data/videos --out data/poses --all-tracks

Uses Ultralytics YOLO-pose (COCO-17 keypoints) with tracking. By default it keeps
the person tracked for the most frames; --all-tracks saves every person separately
(<clip>__t<id>.npz), which is what you want for multi-child scenes.

--rgb-out saves small 16x112x112 RGB clips ONLY for the "pose vs raw video"
comparison experiment. Do not use it for anything else, and never on footage of
real children.
"""
import argparse
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract(video, model, all_tracks=False, rgb_out=None, conf=0.25):
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    tracks = {}       # track_id -> {frame_idx: (17, 3)}
    rgb_frames = []
    n_frames = 0
    for f_idx, r in enumerate(model.track(str(video), stream=True, persist=True, conf=conf, verbose=False)):
        n_frames = f_idx + 1
        if rgb_out is not None:
            rgb_frames.append(cv2.resize(r.orig_img, (112, 112))[:, :, ::-1])
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            continue
        kps = r.keypoints.data.cpu().numpy()                 # (N, 17, 3) pixels + conf
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.arange(len(kps))
        for tid, kp in zip(ids, kps):
            tracks.setdefault(int(tid), {})[f_idx] = kp
        del r  # drop the frame immediately

    out = {}
    for tid, frames in tracks.items():
        seq = np.zeros((n_frames, 17, 3), np.float32)
        for f, kp in frames.items():
            seq[f] = kp
        out[tid] = seq
    if not all_tracks and out:
        best = max(out, key=lambda t: len(tracks[t]))
        out = {best: out[best]}

    rgb = None
    if rgb_frames:
        idx = np.linspace(0, len(rgb_frames) - 1, 16).round().astype(int)
        rgb = np.stack([rgb_frames[i] for i in idx]).astype(np.uint8)  # (16, 112, 112, 3)
    return fps, out, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="data/poses")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="yolo11s/m-pose.pt are more accurate, slower")
    ap.add_argument("--all-tracks", action="store_true")
    ap.add_argument("--rgb-out", default=None, help="only for the pose-vs-RGB experiment")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rgb_dir = Path(args.rgb_out) if args.rgb_out else None
    if rgb_dir:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in Path(args.videos).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    for i, v in enumerate(videos, 1):
        model = YOLO(args.model)  # fresh tracker state per video
        fps, tracks, rgb = extract(v, model, args.all_tracks, rgb_dir, args.conf)
        if not tracks:
            print(f"[{i}/{len(videos)}] {v.name}: no person found")
            continue
        for tid, seq in tracks.items():
            name = v.stem if not args.all_tracks else f"{v.stem}__t{tid}"
            np.savez_compressed(out / f"{name}.npz", keypoints=seq, fps=np.float32(fps))
        if rgb is not None:
            np.savez_compressed(rgb_dir / f"{v.stem}.npz", frames=rgb)
        print(f"[{i}/{len(videos)}] {v.name}: {len(tracks)} track(s), {seq.shape[0]} frames @ {fps:.1f} fps")


if __name__ == "__main__":
    main()
