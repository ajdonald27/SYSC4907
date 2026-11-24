# Test procedure and results

Status values: **PASS**, **FAIL**, **NOT RUN**, or **PARTIAL** (explained in the
row). Performance targets P1–P7 are the *proposed* targets in
[requirements.md](requirements.md); they have not been approved.

Test date: 2026-11-23. Hardware: Jetson Orin Nano Super, IMX477 on CAM1.
Software: see `versions.json` in each experiment folder (L4T R36.4.7,
torch 2.10.0, ultralytics 8.4.218, GStreamer 1.20.3). Model: pretrained
`yolo26n.pt`, SHA-256 `9b09cc8b…8ad4fef` (`model.json`).

All commands are run from `yolo/`. "Viewer" in tests T3–T7 means a GStreamer
RTSP client **on the Jetson itself** (`scripts/rtsp_probe.py` decodes H.264 and
counts frames; `scripts/record_stream.sh` records without re-encoding). Only T8
involves the Mac.

## Summary

| # | Test | Status | Evidence |
|---|------|--------|----------|
| T1 | Automated test suite | PASS (40/40) | console output; rerun with the command below |
| T2 | Camera capture (raw mode) | PASS | [raw run](../experiments/20261123-234529_camera-raw_raw/) |
| T3 | Detection pipeline runs and overlays are drawn | PASS | [sample frame](../experiments/20261123-231308_detect-first_detect/samples/frame_004_t0032.1s.jpg), [clip](../experiments/20261123-231308_detect-first_detect/stream_sample.mp4) |
| T4 | Detections are correct on a known object | NOT RUN | needs a person or other COCO object in view |
| T5 | Viewer reconnect (P5) | PASS | [reconnect_shutdown.txt](../experiments/20261123-232114_reconnect-shutdown_detect/reconnect_shutdown.txt) |
| T6 | Clean shutdown and camera reuse (P6) | PASS | same file |
| T7 | Five-minute sustained run (P1, P2, P4) | PASS | [sustained run](../experiments/20261123-232211_sustained_detect/) |
| T7a | Inference time during T7 (P3), PyTorch weights | FAIL on mean (45.9 ms vs ≤ 40 ms); p95 passes (50.1 ms vs ≤ 66 ms) | [metrics.json](../experiments/20261123-232211_sustained_detect/metrics.json) |
| T7b | Inference time with TensorRT FP16 engine (P3), two-minute run | PASS (mean 20.4 ms, p95 24.0 ms) | [metrics.json](../experiments/20261123-234154_trt-fp16_detect/metrics.json) |
| T8 | Playback in VLC on the Mac | PARTIAL: plays through an SSH tunnel; direct connection FAILS | [run.log](../experiments/20261124-001443_mac-vlc_detect/run.log), see T8 |
| T9 | Glass-to-glass latency in VLC (P7) | NOT RUN | needs the Mac |
| T10 | Custom drone weights | NOT RUN | no drone weights exist yet |

## T1 — Automated test suite

```bash
python3 -m unittest discover -s tests -t . -v
```

Expected: `Ran 40 tests … OK`, no skips on the Jetson.
Result: **PASS**, 40/40, with both `python3` and `.venv/bin/python`. Includes an
RTSP serve → H.264 decode → disconnect → reconnect round trip on localhost
using `videotestsrc`.

## T2 — Camera capture (raw mode)

```bash
scripts/run_with_viewer.sh 10 --raw --duration 30 --label camera-raw
```

Pass criteria: exit code 0; `frames_captured` ≈ 30 fps × duration;
`stream_sample.mp4` is 1280×720; the sample images show the camera scene.

Result: **PASS** — `20261123-234529_camera-raw_raw`. Exit code 0; 900 frames
captured in 30.0 s (30.0 fps), 450 processed (15.0 fps), 0 refused pushes;
recorded clip 1280×720 at 15/1 fps, 9.9 s. Two earlier raw runs reported about
13 fps; both numbers were measurement errors, explained in the engineering log
(22:35 and 23:44).

## T3 — Detection pipeline and overlays

```bash
scripts/run_with_viewer.sh 12 --duration 40 --label detect-first
```

Pass criteria: model loads on CUDA (`model.json` `device: "0"`), exit code 0,
boxes with label and confidence visible in `samples/` and in the recorded clip.

Result: **PASS** — `20261123-231308_detect-first_detect`. Device `0` (Orin),
543 frames processed, 318 boxes drawn on 291 frames, clip is 1280×720 at
15/1 fps. The first attempt (`20261123-231133_detect-first_detect`) failed with
a camera error; cause and fix are in the engineering log (23:11).

What this does not show: the camera was pointed at the underside of a desk, so
every box is a low-confidence false positive (for example `bed 0.26` on a
plastic bag). The test proves that inference results reach the stream as
overlays, nothing more.

## T4 — Detections on a known object — NOT RUN

```bash
.venv/bin/python -m yolo_stream --duration 60 --label known-object
```

Stand in front of the camera, or hold up a cup, bottle or phone. Pass criteria:
the object is boxed with the right label at confidence ≥ 0.5 in at least one
image in `samples/` and visibly follows the object in the stream.

Orientation was corrected on 2026-11-24 (`flip_method` 2 → 0); see T8.

## T5 / T6 — Reconnect, shutdown, camera reuse

```bash
scripts/test_reconnect_shutdown.sh
```

The script starts the application, connects three viewers one after another
(8 s each, with gaps), sends SIGINT, times the exit, then opens the camera with
`gst-launch-1.0`.

Pass criteria: each viewer decodes frames and gets its first frame within 5 s
(P5); exit code 0 within 5 s of SIGINT and the camera reopens (P6).

Result: **PASS** — `20261123-232114_reconnect-shutdown_detect`.

| Step | Measured |
|------|----------|
| Viewer 1 | first frame 0.11 s, 119 frames in 8 s |
| Viewer 2 (reconnect) | first frame 0.12 s, 120 frames |
| Viewer 3 (reconnect) | first frame 0.13 s, 119 frames |
| SIGINT → exit | 3.34 s, exit code 0 |
| Camera reopened | yes, 30 frames |

`run.log` records three "media session started" / "session ended" pairs.
Limit of this test: the viewers were on the Jetson over loopback; time to first
frame from VLC over Wi-Fi will be longer.

## T7 — Five-minute sustained run

```bash
scripts/test_sustained.sh 300
```

A viewer decodes the stream for 275 s; a second viewer records 10 s;
`tegrastats` logs every 5 s.

Result — `20261123-232211_sustained_detect`, commit `d890101`:

| Target (proposed) | Measured | Status |
|-------------------|----------|--------|
| P1 mean ≥ 14.0 fps | 15.0 fps (4500 frames / 300.0 s) | PASS |
| P2 no 5 s interval < 12 fps | minimum 14.96 fps over 59 intervals | PASS |
| P3 inference mean ≤ 40 ms | 45.9 ms | **FAIL** |
| P3 inference p95 ≤ 66 ms | 50.1 ms (max 69.2 ms) | PASS |
| P4 exit code 0, no capture errors, 0 refused pushes | exit 0, 0 errors/warnings in `run.log`, 0 refused | PASS |

Other measurements: 9002 frames captured, 4502 dropped at the buffer (by
design); capture-to-push age mean 63.5 ms, p95 80.5 ms; viewer decoded 4125
frames at 15.0 fps; junction temperature 47.6 → 49.6 °C.

Note on P3: "inference" is the whole Ultralytics `predict()` call
(preprocess + model + postprocess) with PyTorch FP32 weights. It misses the
proposed mean but fits inside the 66.7 ms frame period, which is why the frame
rate target is still met. A TensorRT FP16
engine of the same model was then tested for two minutes
([`20261123-234154_trt-fp16_detect`](../experiments/20261123-234154_trt-fp16_detect/)):
inference mean 20.4 ms, p95 24.0 ms, max 51.8 ms, 15.0 fps, exit code 0. That
meets P3, but it is a two-minute run and the default configuration still uses
the `.pt` weights, so the five-minute result above stands as recorded.

## T8 — VLC on the Mac — PARTIAL

Result (2026-11-24, VLC 3.0.24 on macOS):

| Path | Result |
|------|--------|
| `rtsp://192.168.102.51:8554/detections` (Jetson Wi-Fi address) | FAIL — `live555 demux error: Failed to connect`. The Mac was on a different network (10.0.0.x). |
| `rtsp://100.95.175.87:8554/detections` (Tailscale), with and without `--rtsp-tcp` | FAIL — same error. No connection reached the Jetson (`run.log` shows none), although SSH over the same Tailscale link works. Cause not found. |
| SSH tunnel (`LocalForward 8554 localhost:8554`), `--rtsp-tcp rtsp://localhost:8554/detections` | PASS — Andrew saw the annotated video. `run.log` shows a client connected from 00:15:26 to 00:16:49 (83 s); it appears as 127.0.0.1 because it arrives through the tunnel. |

Not yet done for this test: a screenshot of the VLC window, the 60 s
no-freeze check, and a close-and-reopen from the Mac. Viewing showed the
picture was upside down; `camera.flip_method` was changed from 2 to 0 and the
next sample frames are upright
([frame](../experiments/20261124-001754_mac-vlc_detect/samples/frame_002_t0012.1s.jpg)).

Original procedure:

On the Jetson:

```bash
hostname -I                                  # note the address on the Mac's network
.venv/bin/python -m yolo_stream --label mac-vlc
```

On the Mac:

```bash
/Applications/VLC.app/Contents/MacOS/VLC --network-caching=200 rtsp://<jetson-ip>:8554/detections
```

Pass criteria: video with the status line in the top-left corner appears within
10 s and plays for at least 60 s without freezing. Then close VLC, reopen the
same URL, and confirm video returns (reconnect from the Mac). Stop the
application with Ctrl-C.

Evidence to keep: a screenshot of the VLC window, and the experiment folder —
`run.log` will contain `rtsp client connected: <mac-ip>` lines. If nothing
appears: check `nc -vz <jetson-ip> 8554` from the Mac, then try forcing TCP in
VLC (Settings → Input/Codecs → Live555 stream transport → RTP over RTSP).

## T9 — Glass-to-glass latency — NOT RUN

With T8 running, point the camera at a stopwatch shown on the Mac screen so the
VLC window and the stopwatch are both visible, and take a photo. Latency is the
difference between the two readings. Repeat five times. Proposed target P7:
≤ 1.5 s with `--network-caching=200`.

## T10 — Custom drone weights — NOT RUN

```bash
.venv/bin/python -m yolo_stream --weights /path/to/drone.pt --label drone-weights
```

Pass criteria: `model.json` shows the drone model's class names and SHA-256, and
the run starts. Loading is covered in the unit tests only with a stand-in model.
