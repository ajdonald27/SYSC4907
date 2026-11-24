# Requirements and performance targets

Milestone: live IMX477 capture → YOLO26 detection → annotated RTSP stream
viewable in VLC on a Mac.

## Functional requirements

| ID | Requirement |
|----|-------------|
| F1 | Capture live video from the IMX477 (CAM1, Argus `sensor-id=0`) at 1920×1080, 30 fps through GStreamer. |
| F2 | Hold at most one pending frame between capture and processing; drop older frames and count them. |
| F3 | Run YOLO26n inference on the latest frame using pretrained weights. |
| F4 | Accept custom weights (e.g. a drone model, `.pt` or `.engine`) through configuration only. |
| F5 | Draw bounding boxes with class label and confidence on each processed frame. |
| F6 | Serve the annotated video as H.264 over RTSP at `rtsp://<jetson>:8554/detections`, 1280×720 at 15 fps. |
| F7 | Resolution, frame rates, port, path, encoder, bitrate, model and thresholds are set in a YAML file. |
| F8 | Raw-camera mode streams the camera without loading a model, for troubleshooting. |
| F9 | A viewer can disconnect and reconnect without restarting the application. |
| F10 | SIGINT/SIGTERM stops the application cleanly: camera released, exit code 0. |
| F11 | Every run writes a unique experiment folder: configuration, versions, Git commit, model identifier, log, sample images, timing, frame counts. |
| F12 | Use a software H.264 encoder unless a hardware encoder is explicitly configured and present. |

## Constraints

- Jetson Orin Nano Super, L4T 36.4.7, kernel 5.15.148-tegra, Python 3.10.
- The JetPack CUDA, TensorRT, GStreamer and camera packages must not be replaced or
  downgraded. Python packages go in a virtual environment.
- No NVENC on the Orin Nano (verified: `nvv4l2h264enc` is not installed).

## Out of scope for this milestone

Tracking, radar integration, model training, and any claim about drone detection
accuracy. Pretrained COCO detection only shows that the pipeline works.

## Performance targets: PROPOSED, not yet approved

These were written on 2026-11-23 before any detection run was measured. They have
not been reviewed by the team or supervisor and should be treated as proposals.

| ID | Target (proposed) | How it is measured |
|----|-------------------|--------------------|
| P1 | Mean processed frame rate ≥ 14.0 fps (stream nominal 15 fps) over a five-minute run. | `processed_fps` in `metrics.json` |
| P2 | No 5-second interval below 12 fps during the five-minute run. | `fps` column in `metrics.csv` |
| P3 | Inference time mean ≤ 40 ms and p95 ≤ 66 ms at `imgsz` 640 (fits the 66.7 ms frame period). | `timing.inference` in `metrics.json` |
| P4 | Five-minute run ends with exit code 0, no capture errors, zero refused RTSP pushes. | `exit_code`, `frames_push_failed`, `run.log` |
| P5 | A reconnecting viewer receives video within 5 s. | timed reconnect, `run.log` session entries |
| P6 | After SIGINT the process exits within 5 s with code 0 and the camera can be reopened immediately. | timed shutdown followed by a new capture |
| P7 | Glass-to-glass latency in VLC ≤ 1.5 s with `--network-caching=200`. | stopwatch in view of the camera, photographed with the VLC window |

Frames dropped at the capture buffer are reported but have no pass/fail target:
capturing at 30 fps and streaming at 15 fps drops about half of them by design.
