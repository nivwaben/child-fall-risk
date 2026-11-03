"""Comparison model for the privacy experiment: RAW RGB clips instead of pose.

Uses torchvision's R(2+1)D-18 pretrained on Kinetics-400, fine-tuned on the
16x112x112 clips that extract_pose.py --rgb-out writes. Only adult / team-member
footage may be used here.

    python scripts/train_rgb.py --data data --rgb data/rgb --epochs 15
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from childfall.data import LABEL_ID, read_labels  # noqa: E402
from childfall.metrics import clip_metrics  # noqa: E402

MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


class RGBClips(Dataset):
    def __init__(self, data, rgb, split, augment=False):
        self.rows = [r for r in read_labels(Path(data) / "labels.csv")
                     if r["split"] == split and (Path(rgb) / f"{r['clip_id']}.npz").exists()]
        self.rgb, self.augment = Path(rgb), augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        f = np.load(self.rgb / f"{r['clip_id']}.npz")["frames"]          # (16, 112, 112, 3) uint8
        x = torch.from_numpy(f).float().div(255).permute(3, 0, 1, 2)       # (3, 16, 112, 112)
        if self.augment and torch.rand(1) < 0.5:
            x = x.flip(-1)
        return (x - MEAN) / STD, LABEL_ID[r["label"]]


def build(pretrained=True, num_classes=3):
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--rgb", default="data/rgb")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="runs/rgb_r2plus1d")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = RGBClips(args.data, args.rgb, "train", True), RGBClips(args.data, args.rgb, "val")
    print(f"train {len(tr)}  val {len(va)}  device {device}")
    model = build(not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in DataLoader(tr, args.bs, shuffle=True):
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in DataLoader(va, args.bs):
                ps.append(model(x.to(device)).argmax(1).cpu().numpy())
                ys.append(y.numpy())
        m = clip_metrics(np.concatenate(ys), np.concatenate(ps)) if ys else {"macro_f1": 0, "dangerous_recall": 0}
        print(f"ep {ep} val F1 {m['macro_f1']:.3f} danger-recall {m['dangerous_recall']:.3f}")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), out / "best.pt")


if __name__ == "__main__":
    main()
