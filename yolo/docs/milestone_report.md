# Milestone report: camera → YOLO26 → RTSP

Date: 2026-11-23. Branch `yolo-rtsp-milestone`.

## Outcome

The Jetson captures the IMX477, runs YOLO26n on the GPU, draws the results and
serves a 1280×720, 15 fps H.264 RTSP stream at `/detections`. This was verified
on the Jetson with local RTSP clients, including a five-minute run.

The stream plays in VLC on the Mac **only through an SSH tunnel**; a direct
connection to port 8554 fails and the cause has not been found. The model has
not yet been shown a known object under test conditions. Details and remaining
steps are in [test_procedure.md](test_procedure.md) (T8 and T4).

## What was demonstrated

| Item | Result | Evidence |
|------|--------|----------|
| Camera capture, 1080p30 from the sensor, 720p to the application | 900 frames in 30.0 s | [raw run](../experiments/20261123-234529_camera-raw_raw/) |
| YOLO26n inference on CUDA with overlays in the stream | boxes and labels visible in samples and clip | [first detection run](../experiments/20261123-231308_detect-first_detect/) |
| RTSP at `/detections`, software H.264 (`x264enc`) | decoded by GStreamer clients at 15.0 fps | [viewer.json](../experiments/20261123-232211_sustained_detect/viewer.json) |
| Viewer reconnect | 3 consecutive viewers, first frame in 0.11–0.13 s | [reconnect_shutdown.txt](../experiments/20261123-232114_reconnect-shutdown_detect/reconnect_shutdown.txt) |
| Clean shutdown | SIGINT → exit code 0 in 3.34 s, camera reusable | same file |
| Five-minute run | 4500 frames in 300.0 s, no interval below 14.96 fps, 0 refused pushes, 0 log errors | [sustained run](../experiments/20261123-232211_sustained_detect/) |
| Raw-camera mode | works without PyTorch | [raw run](../experiments/20261123-234529_camera-raw_raw/) |
| TensorRT FP16 engine | inference 20.4 ms mean vs 45.9 ms for PyTorch, two-minute run | [TensorRT run](../experiments/20261123-234154_trt-fp16_detect/) |
| Automated tests | 40 pass, including an RTSP decode and reconnect round trip | `python3 -m unittest discover -s tests -t .` |

Against the proposed targets: frame rate (P1, P2), stability (P4), reconnect
(P5) and shutdown (P6) pass. Inference time (P3) fails on the mean with the
default PyTorch weights (45.9 ms against 40 ms) and passes with the TensorRT
engine. Latency in VLC (P7) was not measured. The targets themselves are
proposals and have not been approved.

## Not demonstrated

- **Direct VLC connection from the Mac.** Fails to connect on both the Wi-Fi
  and Tailscale addresses. VLC does play the stream through an SSH tunnel
  (seen by Andrew, 83 s session in the log), which shows VLC can decode it.
  No screenshot was kept, and latency and reconnect from the Mac were not
  measured.
- **Correct detections.** The camera was looking at the underside of a desk.
  Every box in the evidence is a low-confidence false positive. The evidence
  shows that model output reaches the stream, not that detections are right.
- **Drone detection.** The model is pretrained on COCO, which has no drone
  class. Nothing here validates drone detection. Custom weights are supported
  by configuration but none were available to load.

## Limitations

- Inference runs on a 720p frame resized to 640 px. A small, distant drone
  will cover very few pixels; this resolution choice needs revisiting with
  real drone footage.
- Detection runs on every second sensor frame (15 of 30 fps).
- Software encoding uses CPU; there is no hardware encoder on this board. CPU
  headroom with tracking or radar processing added has not been measured.
- The stream has no authentication or encryption and is reachable by anyone on
  the network.
- Opening the camera fails if the model is loaded first. The start-up order
  works around this; the cause is inferred from two log lines, not proven.
- An unclean kill (`kill -9`) can leave the camera daemon unusable for a while.
- File writes on this SD card can block for over a second. The processing loop
  no longer writes files, but anything added to it later must keep to that.
- The TensorRT engine was tested for two minutes only and its detections were
  not compared with the PyTorch model's. It is not the default.
- Installing PyTorch and Ultralytics took about 35 minutes on this SD card,
  and the TensorRT export about 11 minutes.

## Changes to the Jetson

Added with apt: `python3-pip`, `python3-venv`, `gir1.2-gst-rtsp-server-1.0`,
`libgstrtspserver-1.0-0`, cuDSS 0.7.1. No JetPack, CUDA, TensorRT or camera
package was removed, upgraded or downgraded. Everything else is inside
`yolo/.venv`.

## Next steps

1. Resolve or accept the tunnel for Mac viewing; finish T8, then T9 and T4.
2. Review the proposed targets.
3. Five-minute run with the TensorRT engine, then decide the default.
4. Obtain or train drone weights and evaluate on drone footage.
