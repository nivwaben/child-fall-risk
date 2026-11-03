"""Evaluate the rule baseline and trained models on the TEST split, clean and
under each robustness condition. Writes results/<name>.json and a markdown table.

    python scripts/evaluate.py --data data --ckpts runs/stgcn_seed0/best.pt runs/lstm_seed0/best.pt
    python scripts/evaluate.py --data data --ckpts runs/*/best.pt --conditions clean occlude_20pct child_shape
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from childfall.augment import ROBUSTNESS_CONDITIONS  # noqa: E402
from childfall.data import PoseClips  # noqa: E402
from childfall.metrics import clip_metrics  # noqa: E402
from childfall.models import build  # noqa: E402
from childfall.rules import predict_batch  # noqa: E402


def load(ckpt, device):
    c = torch.load(ckpt, map_location=device)
    m = build(c["model"], T=c["T"]).to(device)
    m.load_state_dict(c["state"])
    m.eval()
    return m, c["T"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", nargs="*", default=list(ROBUSTNESS_CONDITIONS))
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PoseClips(args.data, args.split, args.T)
    print(f"{args.split}: {len(ds)} clips")
    predictors = {"rules": lambda X: predict_batch(X)}
    for ck in args.ckpts:
        model, _ = load(ck, device)
        name = Path(ck).parent.name
        predictors[name] = (lambda m: (lambda X: m(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()))(model)

    results = {}
    for cond in args.conditions:
        rng = np.random.default_rng(args.seed)
        Xc = np.stack([ROBUSTNESS_CONDITIONS[cond](x, rng) for x in ds.X]).astype(np.float32)
        for name, f in predictors.items():
            t0 = time.time()
            with torch.no_grad():
                p = f(Xc)
            ms_per_clip = (time.time() - t0) / max(len(Xc), 1) * 1000
            results.setdefault(name, {})[cond] = {**clip_metrics(ds.y, p), "ms_per_clip": ms_per_clip}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{args.split}.json").write_text(json.dumps(results, indent=1))

    lines = ["| model | condition | dangerous recall | macro F1 | light→danger alarm rate | ms/clip |",
             "|---|---|---|---|---|---|"]
    for name, conds in results.items():
        for cond, m in conds.items():
            lines.append(f"| {name} | {cond} | {m['dangerous_recall']:.3f} | {m['macro_f1']:.3f} | "
                         f"{m['light_fall_alarm_rate']:.3f} | {m['ms_per_clip']:.2f} |")
    table = "\n".join(lines)
    (out / f"eval_{args.split}.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
