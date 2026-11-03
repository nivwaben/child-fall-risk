"""Dataset loading.

On-disk format (one file per clip, produced by scripts/extract_pose.py):
    data/poses/<clip_id>.npz  with  keypoints (T, 17, 3) float32, fps float
Labels: data/labels.csv with columns
    clip_id, label, subject, source, split
    label  in {normal, light_fall, dangerous_fall}
    subject: performer id, used to keep the same person out of both train and test
    source : urfd | le2i | upfall | ntu | team | synthetic
    split  : train | val | test   (assign by SUBJECT, never by clip)
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import random_train_aug
from .preprocess import to_model_input
from .rules import CLASSES

LABEL_ID = {c: i for i, c in enumerate(CLASSES)}


def read_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class PoseClips(Dataset):
    def __init__(self, root, split, T=90, augment=False, sources=None, seed=0):
        root = Path(root)
        rows = [r for r in read_labels(root / "labels.csv") if r["split"] == split]
        if sources:
            rows = [r for r in rows if r["source"] in sources]
        self.rows = rows
        self.X, self.y = [], []
        for r in rows:
            z = np.load(root / "poses" / f"{r['clip_id']}.npz")
            self.X.append(to_model_input(z["keypoints"], float(z["fps"]), T))
            self.y.append(LABEL_ID[r["label"]])
        self.X = np.stack(self.X) if self.X else np.zeros((0, 3, T, 17), np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = random_train_aug(x, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])

    def class_weights(self):
        counts = np.bincount(self.y, minlength=len(CLASSES)).astype(np.float32)
        w = counts.sum() / np.maximum(counts, 1) / len(CLASSES)
        return torch.tensor(w)
