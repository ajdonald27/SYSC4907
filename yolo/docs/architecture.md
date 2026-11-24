# Architecture and design choices

## Data flow

```
 IMX477 ── nvarguscamerasrc ── nvvidconv ── videoconvert ── appsink      capture thread
           1080p30 NV12        flip, scale   BGRx→BGR       1 buffer,     (capture.py)
           (NVMM)              to 720p                       drop=true
                                                                │ copy
                                                                ▼
                                                     LatestFrameBuffer    (frame_buffer.py)
                                                     one slot, counts drops
                                                                │ newest frame, paced to 15 fps
                                                                ▼
                                 Detector.infer ─► draw boxes + HUD       main thread
                                 (detector.py)     (overlay.py)           (app.py)
                                                                │ BGR 720p
                                                                ▼
 VLC ◄── RTSP/RTP ◄── rtph264pay ◄── x264enc ◄── videoconvert ◄── appsrc  GLib thread
         :8554/detections            zerolatency               2 buffers,  (rtsp.py)
                                                               leaky
```

Alongside: `metrics.py` times each stage, `experiment.py` writes the run folder,
`config.py` loads and validates YAML, `pipelines.py` builds the GStreamer strings.

## Modules

| Module | Responsibility |
|--------|----------------|
| `config.py` | Typed dataclasses, defaults, validation, rejection of unknown keys. |
| `pipelines.py` | Capture and RTSP pipeline strings; encoder selection. No GStreamer import, so it is unit-testable anywhere. |
| `capture.py` | Runs the capture pipeline, converts samples to NumPy arrays, reports pipeline errors, releases the camera on stop. |
| `frame_buffer.py` | Single-slot latest-frame buffer with drop counting. |
| `detector.py` | Ultralytics wrapper returning plain `Detection` records; model identity (path, SHA-256, classes). |
| `overlay.py` | Boxes, labels and the status line. |
| `rtsp.py` | GstRtspServer with a shared media factory fed by `appsrc`; tracks viewers and sessions. |
| `metrics.py` | Per-stage timing, frame accounting, interval and final summaries. |
| `experiment.py` | Unique run folder: config, versions, Git commit, logs, samples, metrics. |
| `app.py` | Start-up order, the processing loop, signal handling, shutdown. |

## Design choices and tradeoffs

**GStreamer through PyGObject, not `cv2.VideoCapture`.** Ultralytics installs the
PyPI OpenCV wheel into the virtual environment, and that wheel has no GStreamer
support, so `cv2.VideoCapture(gst_pipeline)` would stop working as soon as
Ultralytics is installed. Using `gi` directly for capture and RTSP keeps OpenCV
for drawing only, and gives access to bus errors and clean state changes.
Cost: a little more code than OpenCV's one-liner.

**Scale to 720p in `nvvidconv` before frames reach Python.** The sensor runs at
1080p but the detector and the stream both receive 720p. This moves scaling to
hardware and makes each frame copy 2.8 MB instead of 6.2 MB. The model resizes to
`imgsz` 640 anyway, so it loses nothing at the default setting. Cost: a later
configuration with `imgsz` above 1280 would not see extra detail; small-drone
work may want full-resolution tiles, which needs a pipeline change.

**Single-slot latest-frame buffer plus `appsink drop=true`.** A slow model lowers
the frame rate instead of increasing latency or memory. Cost: frames are skipped;
at 30 fps in and 15 fps out about half are dropped by design. Drops are counted.

**The processing loop is paced to the stream rate.** Each tick takes the newest
frame, so the encoder gets evenly spaced frames and the GPU is not run at 30 fps
for a 15 fps stream. Cost: detections are computed on every second frame only.

**Software H.264 (`x264enc`, `ultrafast`, `zerolatency`).** The Orin Nano has no
hardware encoder; `nvv4l2h264enc` is not installed on this unit. `encoder: auto`
only considers software encoders. `nvv4l2h264enc` remains selectable for other
Jetsons but is untested here. Cost: CPU time and lower compression efficiency
than slower presets; 2.5 Mbit/s at 720p15 is the starting point.

**One-second keyframe interval, SPS/PPS with every keyframe.** A viewer that
connects or reconnects can decode within about a second. Cost: more bits spent
on keyframes.

**Shared RTSP media, created on demand.** The encoder pipeline exists only while
at least one viewer is connected, and all viewers share it. When the last viewer
leaves, the pipeline is destroyed and the next viewer gets a fresh one; this is
what makes reconnection work. Inference keeps running with no viewer so metrics
and samples do not depend on someone watching.

**Ultralytics for inference, model chosen by path.** The same code path loads
`yolo26n.pt`, custom drone weights, or an exported TensorRT `.engine`. Cost: the
PyTorch runtime is heavy; TensorRT export is the expected route to lower
inference time and is documented but optional.

**Virtual environment with `--system-site-packages`.** The JetPack-provided
`gi`, `tensorrt` and CUDA libraries stay untouched and visible; PyTorch,
Ultralytics and their dependencies live in `yolo/.venv`.

**Experiment folder per run.** Created at start-up so that failed runs also leave
evidence. The model file is hashed so results can be tied to exact weights.

## Threads and shutdown

- capture thread: pulls samples, fills the buffer, watches the pipeline bus.
- main thread: inference, overlay, push, metrics, pacing.
- GLib thread: RTSP server main loop on its own `GLib.MainContext`.

SIGINT/SIGTERM set a stop event. The loop exits, then in a `finally` block the
capture pipeline is set to `NULL` (this closes the Argus session), viewers are
disconnected, the server loop quits, and `metrics.json` is written. A capture
error or 10 s without frames ends the run with exit code 1 through the same path.

`SIGKILL` cannot be handled: killing the process that way leaves the Argus
session open and `nvargus-daemon` may then refuse new sessions until it is
restarted (`sudo systemctl restart nvargus-daemon`). See the engineering log.
