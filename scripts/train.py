"""Stage 2: train a temporal model on pose clips.

    python scripts/train.py --data data --model stgcn --epochs 60
    python scripts/train.py --data data --model lstm
    python scripts/train.py --data data --model transformer

Model selection uses the VALIDATION split only. The test split is touched once,
by scripts/evaluate.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from childfall.data import PoseClips  # noqa: E402
from childfall.metrics import clip_metrics  # noqa: E402
from childfall.models import build  # noqa: E402


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="stgcn", choices=["stgcn", "lstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=90, help="frames per clip at 15 fps (90 = 6 s)")
    ap.add_argument("--sources", nargs="*", default=None, help="restrict training sources, e.g. urfd le2i")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = PoseClips(args.data, "train", args.T, augment=True, sources=args.sources, seed=args.seed)
    va = PoseClips(args.data, "val", args.T, sources=args.sources)
    print(f"train {len(tr)} clips, val {len(va)} clips, device {device}")
    tl = DataLoader(tr, args.bs, shuffle=True, drop_last=len(tr) > args.bs)
    vl = DataLoader(va, args.bs)

    model = build(args.model, T=args.T).to(device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=tr.class_weights().to(device))

    run = Path(args.out) / f"{args.model}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        yv, pv = predict(model, vl, device)
        m = clip_metrics(yv, pv)
        # rank by macro F1 with dangerous recall as tie-break priority
        score = m["macro_f1"] + 0.5 * m["dangerous_recall"]
        history.append(dict(epoch=ep, loss=total / len(tr), **{k: m[k] for k in ("macro_f1", "dangerous_recall")}))
        flag = ""
        if score > best:
            best = score
            torch.save({"model": args.model, "T": args.T, "state": model.state_dict()}, run / "best.pt")
            flag = " *"
        print(f"ep {ep:3d} loss {total / len(tr):.3f} val F1 {m['macro_f1']:.3f} "
              f"danger-recall {m['dangerous_recall']:.3f} ({time.time() - t0:.1f}s){flag}")
    (run / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best checkpoint: {run / 'best.pt'}")


if __name__ == "__main__":
    main()
