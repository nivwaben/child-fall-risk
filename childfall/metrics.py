"""Evaluation metrics from the proposal.

Clip level : dangerous-fall recall, 3-class macro F1, confusion matrix.
Stream level: false alarms per hour and alert latency, measured by sliding
              a window over long videos (scripts/evaluate.py --stream).
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from .rules import CLASSES, DANGEROUS


def clip_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(CLASSES)))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    # "light fall raised as dangerous" is the false alarm we care about most
    light = y_true == 1
    light_to_danger = float((y_pred[light] == DANGEROUS).mean()) if light.any() else float("nan")
    return {
        "dangerous_recall": float(rec[DANGEROUS]),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {c: float(r) for c, r in zip(CLASSES, rec)},
        "light_fall_alarm_rate": light_to_danger,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def alarms_from_stream(times_s, preds, refractory_s=10.0):
    """Collapse per-window predictions into alarm events (one alarm per refractory period)."""
    alarms, last = [], -1e9
    for t, p in zip(times_s, preds):
        if p == DANGEROUS and t - last >= refractory_s:
            alarms.append(t)
            last = t
    return alarms


def stream_metrics(alarms, events, duration_s, window_s, tolerance_s=10.0):
    """alarms: alarm times (window START times). events: list of (t_start, t_end) true dangerous falls.

    An alarm is a hit if its window END (start + window_s) falls inside
    [event start, event end + tolerance]. Latency = window end - event start.
    """
    hits, latencies, false_alarms = set(), [], 0
    for a in alarms:
        end = a + window_s
        matched = False
        for k, (s, e) in enumerate(events):
            if s <= end <= e + tolerance_s:
                if k not in hits:
                    latencies.append(end - s)
                hits.add(k)
                matched = True
                break
        if not matched:
            false_alarms += 1
    hours = duration_s / 3600.0
    return {
        "event_recall": len(hits) / len(events) if events else float("nan"),
        "false_alarms_per_hour": false_alarms / hours if hours > 0 else float("nan"),
        "mean_latency_s": float(np.mean(latencies)) if latencies else float("nan"),
    }
