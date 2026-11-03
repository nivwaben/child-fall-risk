# 촬영 프로토콜 (Recording protocol for team-acted data)

## Safety first

- **Only adult team members act. Never film children.**
- Use thick gym/safety mats (≥ 10 cm), and have a spotter at every take.
- "Dangerous" falls are **acted gently**: a slow controlled drop onto the mat, followed by lying still for 5+ s. Nobody actually falls from a height or lands on their head. The model learns the **post-fall pattern** (immobility, posture), not the impact.
- For "head-first" and "from height" cases, do a controlled lie-down in that posture or a step-off from a low (≤ 30 cm) padded box. Stop immediately if anyone feels unsafe.

## Setup

| item | setting |
|---|---|
| Camera | phone, landscape, fixed on a tripod, 1080p, 30 fps |
| Height / angle | ~2.2 m, looking down about 30° (like a ceiling CCTV) |
| Views | main view plus one side view (for the robustness test) |
| Distance | whole body always in frame |
| Room | gym or lab with mats; ordinary lighting, plus one dim session |

## Shot list (per actor, per view)

| class | takes | variations |
|---|---|---|
| normal | 10 | run, jump, roll on purpose, sit down fast, crawl, squat |
| light_fall | 10 | trip forward / backward / sideways, then get up within 3 s |
| dangerous_fall | 8 | lie still 5–10 s after falling; 2 of them in a head-first posture |
| two-person occlusion | 6 | a second member walks in front of the camera during the fall |
| long stream | 1 × 10 min | continuous play with 2–3 falls; note the times for `events.csv` |

With 4 members and 2 views, that comes to about 450 clips plus 80 minutes of stream video.

## File naming

`team_<actor>_<view>_<class>_<nn>.mp4`, for example `team_ben_main_light_03.mp4`.

`subject` in `labels.csv` = `team_<actor>`.

## After recording

1. Run `extract_pose.py` on the videos.
2. **Delete the raw videos**, unless they are needed for the RGB comparison. Store any kept videos only on the team drive, not in the git repo.
3. Label the clips with the [labeling guide](labeling_guide.md).
