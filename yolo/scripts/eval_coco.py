#!/usr/bin/env python3
"""Offline accuracy and speed of one or more models on COCO val2017.

Needs no camera. Expects the dataset at datasets/coco (see README):
    datasets/coco/images/val2017/*.jpg
    datasets/coco/labels/val2017/*.txt

Usage: .venv/bin/python scripts/eval_coco.py WEIGHTS [WEIGHTS...]
       [--imgsz 640] [--limit N] [--out FILE.json]

mAP is computed by Ultralytics over all confidences. Precision and recall are
reported twice: at the confidence that maximises F1 (Ultralytics' default) and
at the confidence the stream actually uses (--conf, default 0.7).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils import ROOT

YOLO_DIR = Path(__file__).resolve().parent.parent
COCO = YOLO_DIR / "datasets" / "coco"


def dataset_yaml(limit):
    """Write a val-only dataset file with an absolute path; return its path."""
    images = sorted((COCO / "images" / "val2017").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images under {COCO / 'images' / 'val2017'}")
    if limit:
        images = images[:limit]
    listing = COCO / (f"val_{len(images)}.txt")
    listing.write_text("\n".join(str(p) for p in images) + "\n")
    names = yaml.safe_load((ROOT / "cfg" / "datasets" / "coco.yaml").read_text())["names"]
    out = COCO / f"val_{len(images)}.yaml"
    out.write_text(yaml.safe_dump({"path": str(COCO), "train": str(listing), "val": str(listing), "names": names}))
    return out, len(images)


def at_confidence(metrics, conf):
    """Mean precision and recall over classes at a fixed confidence (IoU 0.5)."""
    box = metrics.box
    x = np.linspace(0, 1, box.p_curve.shape[1])
    i = int(np.abs(x - conf).argmin())
    return float(box.p_curve[:, i].mean()), float(box.r_curve[:, i].mean())


def evaluate(weights, data, imgsz, conf):
    model = YOLO(weights, task="detect")
    start = time.perf_counter()
    m = model.val(data=str(data), imgsz=imgsz, batch=1, device="0", plots=False, verbose=False,
                  project=str(YOLO_DIR / "runs"), name="eval", exist_ok=True)
    wall = time.perf_counter() - start
    p_at, r_at = at_confidence(m, conf)
    return {
        "weights": weights,
        "imgsz": imgsz,
        "map50_95": round(float(m.box.map), 4),
        "map50": round(float(m.box.map50), 4),
        "precision_best_f1": round(float(m.box.mp), 4),
        "recall_best_f1": round(float(m.box.mr), 4),
        f"precision_at_conf_{conf}": round(p_at, 4),
        f"recall_at_conf_{conf}": round(r_at, 4),
        "speed_ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
        "wall_s": round(wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.7, help="operating confidence for the extra precision/recall")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = all 5000)")
    ap.add_argument("--out")
    a = ap.parse_args()
    data, n = dataset_yaml(a.limit)
    results = []
    for w in a.weights:
        results.append(evaluate(w, data, a.imgsz, a.conf))
        print(json.dumps(results[-1]))
    text = json.dumps({"dataset": "COCO val2017", "images": n, "results": results}, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
