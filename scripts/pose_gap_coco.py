"""Stage-1 experiment: how much worse is the pose model on children than adults?

Runs a YOLO-pose model on a COCO-format keypoint set and reports OKS keypoint AP.
Run it twice, on the COCO Adult and COCO Child validation subsets released with
"Exploring adult biases in child pose estimation" (Utrecht thesis), or on
Kinetikids-pose if you get access, then compare the two numbers.

    python scripts/pose_gap_coco.py --ann coco_child.json --images coco/val2017 --model yolo11n-pose.pt
    python scripts/pose_gap_coco.py --ann coco_adult.json --images coco/val2017 --model yolo11n-pose.pt
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO keypoint annotation JSON")
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--out", default="results/pose_gap")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    gt = COCO(args.ann)
    model = YOLO(args.model)
    dets = []
    img_ids = gt.getImgIds()
    for k, img_id in enumerate(img_ids, 1):
        info = gt.loadImgs(img_id)[0]
        r = model(str(Path(args.images) / info["file_name"]), verbose=False, conf=0.01)[0]
        if r.keypoints is None:
            continue
        kps = r.keypoints.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        for kp, b, s in zip(kps, boxes, scores):
            flat = []
            for x, y, c in kp:
                flat += [float(x), float(y), float(c)]
            dets.append({"image_id": img_id, "category_id": 1, "keypoints": flat, "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
        if k % 100 == 0:
            print(f"{k}/{len(img_ids)} images")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.ann).stem
    det_file = out / f"{name}_{Path(args.model).stem}_dets.json"
    det_file.write_text(json.dumps(dets))
    if not dets:
        print("no detections")
        return
    ev = COCOeval(gt, gt.loadRes(str(det_file)), "keypoints")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    res = {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "AR": ev.stats[5]}
    (out / f"{name}_{Path(args.model).stem}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
