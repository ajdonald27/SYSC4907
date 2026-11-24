# Hardware limits and model choice on the Jetson

Measured on 2026-11-24. This document stands on its own; the step-by-step
record is in [engineering_log.md](engineering_log.md).

## Questions and short answers

| Question | Answer |
|----------|--------|
| Can the Jetson run a larger model than YOLO26n? | Yes. YOLO26s and YOLO26m both held 30 frames per second (fps) through the whole pipeline. |
| Is 30 fps the limit? | No. The pipeline reached 41 to 48 fps depending on the model. The camera itself stops at 60 fps, and no model reached that. |
| Which model is most accurate? | YOLO26m, on a standard public test set. It finds about twice as many cars as YOLO26n at the same settings. |
| Does a larger input image help? | Not shown. It costs a lot of speed, and the public test set cannot answer the accuracy question for our camera. |
| What do we recommend? | YOLO26m, input size 640, 30 fps, with the confidence threshold lowered from 0.7 to about 0.4. See [Recommendation](#recommendation). |

## What was tested

The system under test is the detection pipeline in this folder:

    camera -> object detector -> draw boxes -> H.264 video encoder -> network stream

| Item | Value |
|------|-------|
| Computer | NVIDIA Jetson Orin Nano Super developer kit, 8 GB memory |
| Camera | Raspberry Pi HQ camera (IMX477 sensor). It offers 1920x1080 at up to 60 fps, or 3840x2160 at up to 30 fps. |
| Detector | YOLO26 in three sizes: n (nano, smallest), s (small), m (medium) |
| Power mode | 25W. The board has a faster mode, `MAXN_SUPER`, that was not tested. |
| Video sent to the detector | 1280x720 |

### Terms used below

- **TensorRT engine.** A copy of the model compiled for this specific Jetson.
  It runs about three times faster than the original model file (`.pt`). All
  results below use engines, built with 16-bit numbers (FP16).
- **Input size (`imgsz`).** The detector does not see the full camera frame.
  The frame is shrunk to a square of this many pixels first. 640 is the
  standard. A larger value keeps more detail but takes longer.
- **Confidence threshold.** Each detection comes with a score from 0 to 1. Boxes
  below the threshold are discarded. Our stream currently uses 0.7.
- **Precision.** Of the boxes the detector reports, the share that are correct.
  Low precision means false alarms.
- **Recall.** Of the objects really present, the share the detector finds. Low
  recall means misses.
- **AP and mAP.** A single accuracy score from 0 to 1 that combines precision
  and recall over all thresholds. AP is for one class; mAP is the average over
  all classes. Higher is better.
- **COCO val2017.** A public set of 5000 labelled photos with 80 everyday
  object classes, including person and car. It is the standard benchmark for
  these models.

## Result 1: speed

### Full pipeline

Each run lasted 70 seconds with one viewer watching the stream. "Requested" is
the frame rate the pipeline was asked to hold.

| Model | Requested | Achieved | Peak power | Peak temperature |
|-------|-----------|----------|------------|------------------|
| YOLO26s | 30 fps | 29.9 fps | 9.3 W | 50 C |
| YOLO26m | 30 fps | 30.0 fps | 11.1 W | 53 C |
| YOLO26n | 60 fps | 48.3 fps | 9.6 W | 52 C |
| YOLO26s | 60 fps | 45.2 fps | 10.6 W | 53 C |
| YOLO26m | 60 fps | 41.1 fps | 15.3 W | 58 C |

Memory use peaked at 3.7 of 7.6 GB, so memory is not a limit.

Two cautions about these runs:

- The room was nearly dark and the scene (a desk) contained nothing the model
  detects. The cost of drawing many boxes was therefore not exercised.
- 70 seconds is too short to show whether the board slows down once it is
  fully warm.

### Detector alone, by input size

This is the detector running as fast as it can on one stored 1280x720 frame,
with no camera or video encoder. It is an upper bound: at input size 640 the
full pipeline was 15 to 25 % slower than these figures.

| Model | Input 640 | Input 960 | Input 1280 |
|-------|-----------|-----------|------------|
| YOLO26n | 63.8 fps | 47.5 fps | 45.6 fps |
| YOLO26s | 58.4 fps | 40.8 fps | 30.2 fps |
| YOLO26m | 53.1 fps | 26.9 fps | 16.3 fps |

At input size 640 the three models are close in speed. At larger input sizes
they separate sharply: YOLO26m cannot reach 30 fps at 960 or 1280. The full
pipeline was not run at 960 or 1280.

## Result 2: accuracy

The scene in front of the camera does not change and contains no people or
vehicles, so accuracy could not be measured live. It was measured by running
each engine on the Jetson over COCO val2017.

### By model, input size 640

| Model | mAP (all 80 classes) | Person AP | Car AP |
|-------|----------------------|-----------|--------|
| YOLO26n | 0.403 | 0.518 | 0.370 |
| YOLO26s | 0.480 | 0.596 | 0.469 |
| YOLO26m | 0.527 | 0.631 | 0.521 |

These scores are close to the figures the model's authors publish, which
indicates the engines on the Jetson lose no accuracy.

### Effect of the confidence threshold

Each cell is precision / recall in percent, at input size 640.

| Class | Model | Threshold 0.3 | 0.4 | 0.5 | 0.7 (current) |
|-------|-------|---------------|-----|-----|---------------|
| Person | YOLO26n | 85 / 63 | 90 / 57 | 94 / 51 | 97 / 35 |
| Person | YOLO26s | 82 / 73 | 89 / 68 | 93 / 62 | 97 / 46 |
| Person | YOLO26m | 82 / 77 | 88 / 72 | 92 / 67 | 96 / 52 |
| Car | YOLO26n | 78 / 47 | 85 / 39 | 90 / 32 | 96 / 18 |
| Car | YOLO26s | 75 / 61 | 82 / 55 | 88 / 49 | 96 / 32 |
| Car | YOLO26m | 75 / 68 | 82 / 62 | 87 / 55 | 94 / 38 |

How to read it: with YOLO26m at the current threshold of 0.7, 96 % of reported
people are real, but only 52 % of the people in the photos are found. Lowering
the threshold to 0.4 finds 72 % of them, and 88 % of reports are still real.

These are figures for single photos. In live video the same person or car
appears in many frames, so the chance of detecting it at least once is higher.

### Effect of input size

| Model | Input | mAP | Person AP | Car AP |
|-------|-------|-----|-----------|--------|
| YOLO26n | 640 | 0.403 | 0.518 | 0.370 |
| YOLO26n | 960 | 0.391 | 0.535 | 0.423 |
| YOLO26n | 1280 | 0.331 | 0.480 | 0.434 |
| YOLO26s | 640 | 0.480 | 0.596 | 0.469 |
| YOLO26s | 960 | 0.452 | 0.592 | 0.506 |
| YOLO26s | 1280 | 0.383 | 0.533 | 0.499 |
| YOLO26m | 640 | 0.527 | 0.631 | 0.521 |
| YOLO26m | 960 | 0.465 | 0.588 | 0.541 |
| YOLO26m | 1280 | 0.387 | 0.498 | 0.524 |

Overall accuracy falls as the input grows. This is most likely a property of
the test set and not of the Jetson. COCO photos are about 640 pixels wide, so
an input of 960 or 1280 enlarges them, and objects become larger than the
models were trained to expect. Cars, which are often small in COCO photos, do
improve at 960.

This result must not be read as "a larger input is worse for our camera". Our
camera frames are 1280x720 or larger, so a larger input would keep real detail
instead of enlarging. Whether that helps with distant targets is still unknown.

## Recommendation

Use **YOLO26m at input size 640 and 30 fps**, and lower the confidence
threshold to about **0.4**.

- YOLO26m is the most accurate of the three and held 30 fps through the full
  pipeline with room to spare.
- Input size 640 is the only size at which YOLO26m can hold 30 fps.
- A threshold of 0.4 recovers many missed detections for a modest loss of
  precision. The radar check planned for this project is a second filter
  against false alarms.

The threshold has not been changed in the code. It was raised to 0.7 on
2026-11-24 because weak boxes cluttered the video, and the right value depends
on how the team weighs a missed detection against a false alarm.

## What is still unknown

| Open question | Why it matters | What is needed |
|---------------|----------------|----------------|
| Accuracy on our own target, at real distances | COCO photos rank the models but do not predict performance on our scene | Recorded and labelled footage from our camera |
| Whether a larger input helps with distant targets | The COCO result cannot answer this | The same footage |
| Low-light behaviour | The project targets night-time use, and the camera image was mostly noise in a dim room | Tests of exposure, gain and added lighting |
| Behaviour over a long run | 70-second runs do not show slowdown from heat | A run of 30 to 60 minutes |
| Speed in `MAXN_SUPER` power mode | The board may be faster than measured here | `sudo nvpmodel -m 2`, then repeat the speed runs |
| Delay from camera to screen | A proposed requirement with no measurement yet | Test T9 in [test_procedure.md](test_procedure.md) |
| What frame rate the system needs | Spare speed can be spent on accuracy only if the requirement is known | A decision by the team |

## How to repeat the measurements

All commands are run from `yolo/`. Dataset setup is in the
[README](../README.md#offline-evaluation).

```bash
# build an engine (11 to 19 minutes each); the output is yolo26m.engine
.venv/bin/yolo export model=yolo26m.pt format=engine quantize=16 imgsz=640 device=0

# full pipeline for 70 seconds at 30 fps, then at 60 fps
scripts/test_sustained.sh 70 --config config/trt_30fps.yaml --weights yolo26m.engine
scripts/test_sustained.sh 70 --config config/trt_60fps.yaml --weights yolo26m.engine

# detector alone on one frame
.venv/bin/python scripts/bench_models.py FRAME.jpg yolo26n.engine yolo26s.engine yolo26m.engine

# accuracy on COCO, with the person and car breakdown
.venv/bin/python scripts/eval_coco.py yolo26n.engine yolo26s.engine yolo26m.engine --classes person car
```

An engine built for one input size only works at that size. The 960 and 1280
engines were built with `imgsz=960` and `imgsz=1280` and renamed, for example
`yolo26m_960.engine`.

## Where the raw results are

All paths are under `experiments/`.

| Result | Location |
|--------|----------|
| Full pipeline runs | the five `20261124-09*_limits-*_detect` and `20261124-10*_limits-*_detect` folders |
| Detector alone | `20261124-limits_bench/bench_25W.json`, `20261124-coco-eval/bench_960_25W.json`, `bench_1280_25W.json` |
| COCO accuracy | `20261124-coco-eval/coco_val_640_25W_person_car.json`, and the `960` and `1280` files beside it |
