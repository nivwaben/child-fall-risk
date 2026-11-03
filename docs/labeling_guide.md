# 라벨링 가이드 (Labeling guide)

Every clip gets **one** label. Two people label each clip independently, and we report Cohen's κ (`sklearn.metrics.cohen_kappa_score`). If κ < 0.6, the definitions below are not clear enough: discuss them, revise them, and relabel.

## Classes

| 라벨 | 조건 (ALL must hold) |
|---|---|
| `normal` 정상 동작 | 엉덩이(hip)가 바닥 근처로 **빠르게** 내려가지 않음. 의도적으로 천천히 앉기·눕기·구르기·쪼그리기는 normal |
| `light_fall` 가벼운 낙상 | 의도치 않게 빠르게 넘어짐 **AND** 넘어진 뒤 3초 이내에 스스로 움직여 일어나거나 앉음 **AND** 머리부터 떨어지지 않음 |
| `dangerous_fall` 위험 낙상 | 의도치 않게 넘어짐 **AND** (넘어진 뒤 **3초 이상** 거의 움직임 없음 **OR** 머리·목이 먼저 바닥에 닿음 **OR** 높은 곳(≥ 무릎 높이)에서 떨어짐) |

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
