#!/usr/bin/env python3
"""Time model inference on one 1280x720 frame, with no camera, encoder or pacing.

Calls the same Ultralytics predict() the application uses, as fast as it will
go, so the result is the ceiling the model alone puts on the frame rate.

Usage: .venv/bin/python scripts/bench_models.py IMAGE WEIGHTS [WEIGHTS...]
       [--runs 300] [--imgsz 640] [--out FILE.json]
"""
import argparse
import json
import statistics
import time

import cv2
from ultralytics import YOLO


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def bench(weights, image, runs, imgsz):
    model = YOLO(weights, task="detect")
    kw = dict(imgsz=imgsz, conf=0.7, device="0", verbose=False)
    for _ in range(20):
        model.predict(image, **kw)
    wall, stages = [], {"preprocess": [], "inference": [], "postprocess": []}
    start = time.perf_counter()
    for _ in range(runs):
        t = time.perf_counter()
        r = model.predict(image, **kw)[0]
        wall.append((time.perf_counter() - t) * 1e3)
        for k in stages:
            stages[k].append(r.speed[k])
    elapsed = time.perf_counter() - start
    return {
        "weights": weights,
        "imgsz": imgsz,
        "runs": runs,
        "max_fps": round(runs / elapsed, 1),
        "call_ms": {"mean": round(statistics.fmean(wall), 2), "p50": round(pct(wall, 50), 2),
                    "p95": round(pct(wall, 95), 2), "max": round(max(wall), 2)},
        "stage_mean_ms": {k: round(statistics.fmean(v), 2) for k, v in stages.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out")
    a = ap.parse_args()
    image = cv2.imread(a.image)
    if image is None:
        raise SystemExit(f"cannot read {a.image}")
    results = [bench(w, image, a.runs, a.imgsz) for w in a.weights]
    text = json.dumps({"image": a.image, "shape": list(image.shape), "results": results}, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
