# Reading list

Papers, datasets and sources for the child fall risk project, grouped by the stage they support.
Priority: **★ must read** · ◆ read for your stage · ○ reference only.

Suggested split: give each team member one ★ paper, and have them add 3 takeaways to the hub's Library.

## Start here (the 5 key papers)

| | Paper | Why it matters for us | Stage |
|---|---|---|---|
| ★ | [Yan, Xiong, Lin. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. AAAI 2018](https://arxiv.org/abs/1801.07455) | The base model for stage 2. Every skeleton fall-detection paper builds on it | Temporal model |
| ★ | [Keskes & Noumeir. Vision-Based Fall Detection Using ST-GCN. IEEE Access 2021](https://doi.org/10.1109/ACCESS.2021.3058219) | Closest prior work to our stage 2. Uses transfer learning from NTU RGB+D | Temporal model |
| ★ | [Exploring adult biases in child pose estimation. Utrecht University master's thesis](https://studenttheses.uu.nl/handle/20.500.12932/41410) | Shows pose models do worse on children. Releases Kinetikids-pose and COCO Adult / COCO Child subsets, which are the test sets for experiment 4(a) | Pose estimation, domain gap |
| ★ | [Synthetic-Child: AIGC-based synthetic data for child posture estimation. arXiv 2603.02598, 2026](https://arxiv.org/html/2603.02598v1) | A COCO-pretrained model got 58.7 AP on children; synthetic child data recovered +12.5 AP. Gives a number for the adult→child gap | Domain gap |
| ★ | [Robust 3D Skeletal Joint Fall Detection in Occluded and Rotated Views (VIRA-GCN)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12609388/) | Augments Le2i with simulated rotations and occlusions. Matches our robustness experiment | Robustness |

## Stage 1: pose estimation

| | Paper | Why it matters for us |
|---|---|---|
| ◆ | [Jiang et al. RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose, 2023](https://arxiv.org/abs/2303.07399) | Fast pretrained keypoint extractor. An alternative to YOLO11-pose |
| ◆ | [Xu et al. ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation. NeurIPS 2022](https://arxiv.org/abs/2204.12484) | A more accurate but slower extractor. Useful for checking whether pose quality limits our results |
| ◆ | [Patel et al. AGORA: Avatars in Geography Optimized for Regression Analysis. CVPR 2021](https://ar5iv.labs.arxiv.org/html/2104.14643) | Found most 3D pose methods perform poorly on children. An extra citation for the domain gap |
| ○ | [2D Pose Estimation based Child Action Recognition. arXiv 2212.09027](https://arxiv.org/pdf/2212.09027) | A child/adult-labelled Kinetics subset with OpenPose skeletons (KS-KSS), released on GitHub |
| ○ | [Ultralytics YOLO11-pose](https://docs.ultralytics.com/tasks/pose/) | The pose model `extract_pose.py` uses by default |

## Stage 2: temporal models

| | Paper | Why it matters for us |
|---|---|---|
| ◆ | [Chen et al. Channel-wise Topology Refinement Graph Convolution (CTR-GCN). ICCV 2021](https://arxiv.org/abs/2107.12213) | A stronger GCN than ST-GCN. A drop-in upgrade if time allows |
| ◆ | [Duan et al. Revisiting Skeleton-based Action Recognition (PoseC3D). CVPR 2022](https://arxiv.org/abs/2104.13586) | Introduces PYSKL, a toolbox with working code for ST-GCN, CTR-GCN and PoseC3D |
| ○ | [PYSKL code](https://github.com/kennymckormick/pyskl) | Reference implementations to compare against ours |
| ○ | Sedlar. Cross-Dataset Evaluation of Skeleton-Based Fall Detection Using BiLSTM and Transformer-LSTM. SST 2026 | Classical baselines stayed competitive on small datasets, and training-data diversity mattered more than architecture. Supports including the rule baseline. Only an author summary was found; get the full paper before citing it |

## Privacy experiment (pose-only vs raw video)

| | Paper | Why it matters for us |
|---|---|---|
| ◆ | [Wu et al. Towards Privacy-Preserving Visual Recognition via Adversarial Training: A Pilot Study. ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Zhenyu_Wu_Towards_Privacy-Preserving_Visual_ECCV_2018_paper.html) | How to frame and measure the privacy vs. utility trade-off |

## Datasets

| | Dataset | Notes |
|---|---|---|
| ◆ | [UR Fall Detection Dataset (URFD)](https://fenix.ur.edu.pl/~mkepski/ds/uf.html). Kwolek & Kepski, 2014 | 30 falls + 40 daily-activity sequences |
| ◆ | [UP-Fall Detection Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6539235/). Martínez-Villaseñor et al., Sensors 2019 | Multimodal; many fall directions |
| ◆ | [NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis](https://arxiv.org/abs/1604.02808). Shahroudy et al., CVPR 2016 | Includes a "falling down" class. Access request required |
| ◆ | [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/main/README.md) | Combines Le2i, UP-Fall, CMDFall and others; 16 classes incl. fall / fallen. CC BY-NC 4.0 |
| ○ | [Le2i Fall Detection Dataset](https://imvia.ube.fr/?p=346). Charfi et al., 2013 | 191 videos in 4 scenes. Official download link currently down; use OmniFall |

## Problem statement and law (Section 1 of the proposal)

| | Source | Use |
|---|---|---|
| ★ | [Korea Consumer Agency, 2025 Child Safety Accident Trend Analysis (summary)](https://www.yugacrew.com/crewletter/child-safety-accident-home-falls-2026) | 13,284 child (0–12) incidents; falls from height first at 33%. Cite the original KCA report in the final version |
| ◆ | [Personal Information Protection Act, Article 22-2](https://korea.legal/wiki/%ea%b0%9c%ec%9d%b8%ec%a0%95%eb%b3%b4-%eb%b3%b4%ed%98%b8%eb%b2%95-%ec%a0%9c22%ec%a1%b0%ec%9d%982/) | Processing data of children under 14 needs a legal guardian's consent |

## Labelling

| | Source | Use |
|---|---|---|
| ○ | [Cohen's kappa in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) | Inter-annotator agreement for the labelling guide |
