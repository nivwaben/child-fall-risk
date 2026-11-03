# 프라이버시 보존형 아동 낙상 위험도 판별

**Privacy-preserving child fall risk classification (pose only)**

Video → pose keypoints (raw frames discarded) → temporal model → one of three classes:

| class | label | meaning |
|---|---|---|
| 0 | `normal` | running, jumping, rolling or sitting during play |
| 1 | `light_fall` | falls during play and gets up within a few seconds |
| 2 | `dangerous_fall` | falls and does not move afterwards, or lands head-first |

The labelling rules are in [docs/labeling_guide.md](docs/labeling_guide.md).

## Setup (Colab or local)

```bash
pip install -r requirements.txt
python scripts/make_synthetic.py --out data_synth --n 300     # fake data to check the pipeline
python scripts/train.py --data data_synth --model stgcn --epochs 5 --out runs_synth
python scripts/evaluate.py --data data_synth --ckpts runs_synth/*/best.pt --out results_synth
```

Synthetic data are scripted stick figures used only to test the code. **Never report results on them.**

## Pipeline

| stage | script | output |
|---|---|---|
| 1. Pose extraction | `scripts/extract_pose.py --videos data/videos --out data/poses` | `data/poses/<clip>.npz` (T×17×3) |
| 1b. Child pose gap | `scripts/pose_gap_coco.py --ann coco_child.json ...` | keypoint AP, child vs. adult |
| 2. Labels | hand-made `data/labels.csv` (see below) | |
| 3. Train | `scripts/train.py --model stgcn / lstm / transformer` | `runs/<model>_seed<k>/best.pt` |
| 4. Clip evaluation | `scripts/evaluate.py --ckpts runs/*/best.pt` | `results/eval_test.md` |
| 5. Stream evaluation | `scripts/evaluate_stream.py --root data/stream --ckpt ...` | false alarms / hour, latency |
| 6. Privacy comparison | `extract_pose.py --rgb-out data/rgb`, then `scripts/train_rgb.py` | RGB R(2+1)D vs. pose model |
| 7. Demo | `scripts/demo.py --source clip.mp4 --ckpt ... --out demo.mp4` | skeleton-only video |

### `data/labels.csv`

```
clip_id,label,subject,source,split
urfd_fall-01,dangerous_fall,urfd_p1,urfd,train
team_s03_c12,light_fall,team_ben,team,test
```

- `subject` is the person in the clip. **Split by subject, never by clip.** The same person must not appear in both train and test, or the results will be inflated.
- `source` is one of `urfd`, `upfall`, `le2i`, `ntu`, `omnifall`, `team`.

## Experiments (core scope)

1. **Model comparison.** Compare the rule baseline (`childfall/rules.py`) with ST-GCN and one more model (LSTM or Transformer). The main metrics are dangerous-fall recall and the light→dangerous alarm rate.
2. **Pose-only vs. raw video.** Compare the pose ST-GCN with the RGB R(2+1)D model on the same clips. This shows what accuracy the privacy design gives up, if any.
3. **Robustness.** `evaluate.py` re-runs every model under these conditions: `rotate_30`, `side_view_0.5`, `occlude_20pct`, `occlude_40pct`, `low_res`.
4. **Adult→child gap (two parts).**
   - Stage 1: compare pose AP on COCO Child vs. COCO Adult with `pose_gap_coco.py`.
   - Stage 2: compare the classifier under the `child_shape` condition, which is a simulated body-proportion shift.

   Neither part is real child fall data. The report must say so.

Run every trained model with three seeds (`--seed 0 1 2`) and report the mean ± standard deviation.

## Layout

```
childfall/   skeleton.py preprocess.py augment.py rules.py models.py data.py metrics.py
scripts/     extract_pose.py make_synthetic.py train.py evaluate.py evaluate_stream.py
             train_rgb.py pose_gap_coco.py demo.py
docs/        labeling_guide.md recording_protocol.md
```

## Privacy rules for this project

- **Never film real children.** All team recordings are adult team members acting on safety mats.
- `extract_pose.py` and `demo.py` never write camera frames. The only exception is `--rgb-out`, which is used solely for the comparison experiment on adult footage.
- Personal Information Protection Act (개인정보 보호법) Art. 22-2: processing data of children under 14 requires the legal guardian's consent. This is why a real deployment would need a legal review, and why this project does not collect child data.
