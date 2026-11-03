# Labeling guide

Every clip gets **one** label. Two people label each clip independently, and we report Cohen's κ (`sklearn.metrics.cohen_kappa_score`). If κ < 0.6, the definitions below are not clear enough: discuss them, revise them, and relabel.

## Classes

| Label | Conditions (all must hold) |
|---|---|
| `normal` | The hips do not drop **quickly** to near the floor. Deliberately sitting, lying down, rolling or squatting slowly counts as normal |
| `light_fall` | Falls quickly and unintentionally **AND** moves on their own to get up or sit up within 3 s **AND** does not land head-first |
| `dangerous_fall` | Falls unintentionally **AND** (almost no movement for **3 s or more** afterwards **OR** head or neck hits the floor first **OR** falls from a raised surface, at least knee height) |

## Decision order

1. Did the person go down fast and unintentionally? If not → `normal`.
2. Did the head hit first, or did the fall start from a raised surface? If yes → `dangerous_fall`.
3. Within 3 s, did the person move on purpose (push up, sit up, roll and stand)? If yes → `light_fall`. If not → `dangerous_fall`.

## Notes

- "Almost no movement" means only small twitches. Breathing and small hand movements count as no movement.
- The 3-second threshold is **our working definition**, not a medical standard. The report must say so. Post-fall immobility and impact posture are the cues we can measure from a skeleton. Fall height is hard to measure from 2D pose, so it is used only for labelling, never as a model input.
- Clips are 6–8 s long. For fall classes, the fall should happen in the **first half** so that what happens afterwards is visible.
- Public adult datasets: label URFD / UP-Fall / Le2i fall clips with this guide too. A fall followed by lying still for the rest of the clip is `dangerous_fall`. A fall followed by getting up is `light_fall`.

## Tracking

Store the second labeller's labels in `labels_b.csv` with the same `clip_id`s:

```python
import pandas as pd
from sklearn.metrics import cohen_kappa_score
a, b = pd.read_csv("data/labels.csv"), pd.read_csv("data/labels_b.csv")
m = a.merge(b, on="clip_id")
print(cohen_kappa_score(m.label_x, m.label_y))
```
