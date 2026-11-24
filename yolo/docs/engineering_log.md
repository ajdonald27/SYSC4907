# Engineering log

All times are local (EST) on the Jetson. Experiment folders are under
[`../experiments/`](../experiments/). Numbers are copied from the files named
next to them; nothing here is estimated.

## 2026-11-23

### 22:18 — Environment survey

Commands: `cat /etc/nv_tegra_release`, `uname -r`, `dpkg -l | grep …`,
`gst-inspect-1.0 <element>`, `python3 -c "import gi …"`.

| Item | Finding |
|------|---------|
| Platform | L4T R36.4.7, kernel 5.15.148-tegra, Python 3.10.12, 7.4 GB RAM, power mode 25W |
| CUDA / cuDNN / TensorRT | CUDA 12.6, cuDNN 9.3.0, TensorRT 10.3.0 (apt, with `python3-libnvinfer`) |
| GStreamer | 1.20.3; `nvarguscamerasrc`, `nvvidconv`, `appsrc`, `appsink`, `rtph264pay` present |
| H.264 encoders | `nvv4l2h264enc` and `nvv4l2h265enc` **not installed**; `x264enc` and `openh264enc` present |
| System OpenCV | 4.8.0 with GStreamer support |
| Missing | `pip`, `venv`, PyTorch, Ultralytics, GstRtspServer bindings (`Namespace GstRtspServer not available`) |

Decision: software encoding with `x264enc`. Capture and RTSP through PyGObject
rather than OpenCV, because Ultralytics installs a PyPI OpenCV wheel without
GStreamer support.

Documentation read: Ultralytics Jetson guide and YOLO26 model page. Taken from
them: PyTorch 2.10.0 / torchvision 0.25.0 wheels for JetPack 6 from the
Ultralytics assets release, cuDSS as a required extra library, `yolo26n.pt` as
the pretrained detection weight, TensorRT engines are device-specific.

### 22:25 — Dependency installation

- An attempt to install tools into the home directory without `sudo` (a
  standalone `uv` binary and unpacked `.deb` files) was blocked by the session's
  permission policy. Not pursued further.
- 22:34 — Andrew installed the system packages:
  `sudo apt install python3-pip python3-venv gir1.2-gst-rtsp-server-1.0 libgstrtspserver-1.0-0`.
  No existing JetPack package was changed (checked in `/var/log/dpkg.log`).
- 22:35 — `python3 -m venv --system-site-packages .venv`, then the two PyTorch
  wheels and `pip install ultralytics`. The install took about 35 minutes; the
  SD card was the bottleneck (60 % I/O wait in `top`).
- Andrew installed cuDSS 0.7.1 following the Ultralytics guide.
- 23:10 — Result: torch 2.10.0 with `cuda=True` (device "Orin"), torchvision
  0.25.0, ultralytics 8.4.218, numpy 2.2.6 and opencv-python 5.1.0 inside the
  venv; system `gi` and `tensorrt` 10.3.0 still importable from it.

### 22:32 — Unit tests

`python3 -m unittest discover -s tests -t .` — 40 tests. One failure in my own
test (it sampled a pixel that the status line covers); fixed the test. After the
RTSP packages were installed the RTSP round-trip test stopped being skipped and
passed. 40/40 pass with both the system Python and the venv.

### 22:35 — First camera run, raw mode

`python3 -m yolo_stream --raw --duration 25 --label camera-raw`, with
`scripts/record_stream.sh` as viewer.
Folder: `20261123-223513_camera-raw_raw`.

- Camera opened, 720p stream recorded (2.8 MB, 9.9 s), exit code 0.
- Processed rate 13.0 fps, below the 15 fps nominal. This run overlapped the
  pip install, so the number is not representative. Root causes were found
  later (23:17). Raw mode was re-measured at the end of the session.

### 22:36 — Error: Argus stops creating capture sessions

A throwaway timing script of mine raised an exception and exited without
setting the capture pipeline to `NULL`. Afterwards:

```
(Argus) Error Timeout: … ClientSocketManager.cpp, function send()
… gstnvarguscamerasrc.cpp, execute:805 Failed to create CaptureSession
```

and a plain `gst-launch-1.0 nvarguscamerasrc … ! fakesink` hung. I could not
restart the daemon (no `sudo`). At 22:42 the same `gst-launch` test worked
again without a restart.

Lessons applied: the application always sets the capture pipeline to `NULL` in a
`finally` block; the README warns against `kill -9`; the capture module's
error path was confirmed to report this failure ("capture pipeline reached end
of stream") instead of hanging.

### 23:11 — Error: camera fails when the model is loaded first

`.venv/bin/python -m yolo_stream --duration 40 --label detect-first`
Folder: `20261123-231133_detect-first_detect` (kept as the failure record).

```
gstnvarguscamerasrc.cpp, execute:805 Failed to create CaptureSession
RuntimeError: capture failed: capture pipeline reached end of stream
```

`journalctl -u nvargus-daemon` showed
`SCF: Error InsufficientMemory … CaptureServiceDeviceViCsi.cpp, function open()`.
`/proc/meminfo` right after: `CmaFree: 25996 kB` of `CmaTotal: 262144 kB`,
while 5.4 GB of ordinary memory was available. The camera worked standalone
seconds later.

Fix: open the camera before loading the model (`app.py`). Retest at 23:13
passed and all nine later detection runs started cleanly.
I did not prove the mechanism; "CUDA initialisation leaves too little
contiguous memory for the CSI capture buffers" is my interpretation of the two
log lines above.

Also fixed here: Ultralytics 8.4.218 warns that `half` is deprecated in favour
of `quantize`; the detector now passes `quantize=16` only when `model.half` is
true.

### 23:13 — First successful detection run

Folder: `20261123-231308_detect-first_detect`.

- YOLO26n on CUDA, boxes drawn, stream recorded by a local viewer
  (`stream_sample.mp4`: 1280×720, 15/1 fps, 11.9 s), exit code 0.
- Inference 48.0 ms mean (this is the whole `predict()` call: preprocessing,
  model, postprocessing).
- Reported 13.54 fps with dips in individual 5 s intervals (6.6, 13.3, 14.2).
- Scene: the camera was looking at the underside of a desk at night. The only
  detections were low-confidence false positives, e.g. `bed 0.26` on a plastic
  bag (`samples/frame_004_t0032.1s.jpg`). This shows the overlay path works; it
  says nothing about detection quality.

### 23:14–23:19 — Finding the frame-rate dips

Added timing for the wait on the frame buffer, the capture-to-push age, and
the interval between processed frames.

| Observation | Evidence | Fix |
|-------------|----------|-----|
| The loop never waits for the camera (max 0.6 ms) and inference never exceeds the frame period by much (max 69 ms) | `20261123-231445_detect-timing_detect/metrics.json` | — |
| Summary frame rate was diluted by camera teardown time (elapsed was read after shutdown) | 369 frames at 25.15 s, but only 408 at "30.15 s" | Stop the metrics clock when the loop ends |
| Dips line up with sample-image saves at 2, 12, 22 s | 13.33 fps in the 10–15 s interval | Write samples on a background thread |
| One 1625 ms gap between frames at the first statistics write | ad-hoc 20 s run printing `loop_interval` values over 90 ms: `[(75, 1625)]` | Write CSV rows and the log file on background threads too |

Retest (ad-hoc 30 s run): no interval over 90 ms, `loop_interval` max 76.2 ms,
15.0 fps. Conclusion: any file write from the processing thread can stall for
0.5–1.6 s on this SD card, so the loop now does no file I/O.

### 23:21 — Reconnect and shutdown test

`scripts/test_reconnect_shutdown.sh`
Folder: `20261123-232114_reconnect-shutdown_detect` (`reconnect_shutdown.txt`).

- Three viewers in sequence: first decoded frame after 0.11 s, 0.12 s, 0.13 s;
  119–120 frames each in 8 s. `run.log` shows three media sessions.
- SIGINT to exit: 3.34 s, exit code 0. Camera reopened immediately afterwards
  with `gst-launch-1.0` (30 frames).
- The first attempt at this script reported "exit code 127" because `wait` ran
  inside a pipe subshell. That was a script bug, fixed; the first folder was
  discarded and the test rerun.

### 23:22 — Five-minute sustained run

`scripts/test_sustained.sh 300`
Folder: `20261123-232211_sustained_detect`. Git commit `d890101`, no
uncommitted changes.

| Measure | Value | Source |
|---------|-------|--------|
| Duration | 300.0 s | `metrics.json` |
| Frames captured / processed | 9002 / 4500 | `metrics.json` |
| Processed frame rate | 15.0 fps; per-5 s interval min 14.96, max 15.06 | `metrics.json`, `metrics.csv` |
| Frames dropped at buffer | 4502 (by design: 30 fps in, 15 fps out) | `metrics.json` |
| RTSP pushes refused | 0 | `metrics.json` |
| Inference | mean 45.9 ms, p50 45.9, p95 50.1, max 69.2 | `metrics.json` |
| Capture-to-push age | mean 63.5 ms, p95 80.5, max 91.6 | `metrics.json` |
| Viewer | first frame 0.12 s, 4125 frames in 275 s, 15.0 fps, no error | `viewer.json` |
| Errors or warnings in log | 0 | `run.log` |
| Exit | "duration reached", code 0 | `metrics.json` |
| Temperature | tj 47.6 °C at start, 49.6 °C at end | `tegrastats.log` |

Detections during this run (3490 boxes on 2731 frames) are again false
positives on a static scene with no COCO objects in it.

### 23:28 — TensorRT export

`.venv/bin/yolo export model=yolo26n.pt format=engine quantize=16 imgsz=640 device=0`

- First attempt failed: Ultralytics tried to auto-install
  `onnxruntime-gpu[cuda]`, which has no PyPI build for this platform
  (`No matching distribution found`), so `onnx` was not installed either
  (`No module named 'onnx'`).
- Fix: `.venv/bin/pip install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.71"` plus the
  Jetson `onnxruntime_gpu-1.23.0` wheel named in the Ultralytics Jetson guide.
  Installed onnx 1.23.2, onnxslim 0.1.99, onnxruntime-gpu 1.23.0. System
  TensorRT 10.3.0 was used for the build and was not modified.
- Second attempt succeeded: engine built in 564 s (682 s including ONNX
  export), `yolo26n.engine`, 8.0 MB.

### 23:41 — TensorRT FP16 run (two minutes)

`scripts/test_sustained.sh 120 --weights yolo26n.engine`
Folder: `20261123-234154_trt-fp16_detect` (renamed from `…_sustained_detect`
so it is not confused with the five-minute run).

| Measure | PyTorch FP32 (5 min) | TensorRT FP16 (2 min) |
|---------|----------------------|-----------------------|
| Inference mean / p95 / max | 45.9 / 50.1 / 69.2 ms | 20.4 / 24.0 / 51.8 ms |
| Capture-to-push age mean / p95 | 63.5 / 80.5 ms | 38.5 / 56.1 ms |
| Processed frame rate | 15.0 fps | 15.0 fps |
| Refused pushes, exit code | 0, 0 | 0, 0 |

Both figures include Python pre- and post-processing. No accuracy comparison
between the two was made.

### 23:44 — Raw mode re-measured

A rerun of raw mode again showed 13.4 fps with 26.7 fps "captured". Cause: in
raw mode there is no model load, so the metrics clock started about 3 s before
the sensor delivered its first frame. Fix: wait for the first frame before
starting the clock (`App._wait_for_first_frame`). The run with the wrong clock
was discarded.

Retest, folder `20261123-234529_camera-raw_raw`: 900 frames captured and 450
processed in 30.0 s (30.0 / 15.0 fps), exit code 0.

### Next actions

1. Open the stream in VLC on the Mac and record the result (test T8), then
   measure latency (T9).
2. Put a person or known object in front of the camera and confirm correct
   boxes and picture orientation (T4).
3. Decide whether the TensorRT engine becomes the default, after a five-minute
   run with it.
4. Have the proposed performance targets reviewed.
5. Later milestones: drone weights, tracking, radar integration.

## 2026-11-24

### 00:03 — VLC on the Mac

- Direct to `192.168.102.51` and to the Tailscale address `100.95.175.87`:
  `live555 demux error: Failed to connect`. On the Jetson, `ss -ltn` showed the
  server listening on `0.0.0.0:8554`, `scripts/rtsp_probe.py` decoded the
  stream through both addresses, and `run.log` showed no attempt from the Mac.
  `tailscale status` showed the Mac on a 10.0.0.x network, which explains the
  Wi-Fi address. Why port 8554 is unreachable over Tailscale while SSH works
  was not determined (candidates: a Tailscale ACL, macOS blocking VLC).
- Workaround: `LocalForward 8554 localhost:8554` in the Mac's SSH config, then
  `VLC --rtsp-tcp rtsp://localhost:8554/detections`. Andrew saw the annotated
  stream; the log shows an 83 s viewer session (00:15:26–00:16:49) in
  `20261124-001443_mac-vlc_detect`.
- 00:13 — the session running the stream restarted and the application received
  SIGTERM; it exited with code 0 (`20261124-000844_mac-vlc_detect/run.log`).
- A 20-minute run before that (`20261123-234719_mac-vlc_detect`) processed
  17998 frames in 1200 s (15.0 fps), exit code 0, with no viewer for most of it.

### 00:17 — Picture upside down

Seen in VLC. Changed the default `camera.flip_method` from 2 to 0 (code,
`config/default.yaml`, test, README). The value 2 came from the `nv3dsink`
preview command. Retest: `20261124-001754_mac-vlc_detect` samples are upright
(printed text reads normally).

### 00:19 — One more frame-rate dip

In `20261124-001754_mac-vlc_detect/metrics.csv` the 5–10 s interval shows
12.09 fps. In this run the console output was redirected to a file on the SD
card, and console logging was still written directly from the processing
thread. Console logging now goes through a queue as well (`__main__.py`). This
cause is assumed from the earlier stall findings; it has not been confirmed by
a rerun with redirected output.

### Next actions

1. Find out why port 8554 is blocked between the Mac and the Jetson, or accept
   the SSH tunnel as the viewing method.
2. Finish T8 (screenshot, 60 s, reopen from the Mac), then T9 (latency) and T4
   (known object).
3. Rerun with output redirected to a file to confirm the console-logging fix.
