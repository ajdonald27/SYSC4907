# yolo: IMX477 → YOLO26 → RTSP

Captures the IMX477 camera on a Jetson Orin Nano Super, runs YOLO26 object
detection, draws the boxes, and serves the result as an RTSP stream that VLC on
another machine can open:

    rtsp://<jetson-ip>:8554/detections

Pretrained COCO weights are used to prove the pipeline. That is not evidence of
drone detection performance; see [docs/milestone_report.md](docs/milestone_report.md).

| Document | Contents |
|----------|----------|
| [docs/requirements.md](docs/requirements.md) | Requirements and proposed performance targets |
| [docs/architecture.md](docs/architecture.md) | Data flow, modules, design choices and tradeoffs |
| [docs/test_procedure.md](docs/test_procedure.md) | Test steps and pass/fail/not-run results with evidence |
| [docs/engineering_log.md](docs/engineering_log.md) | Dated log of commands, errors, fixes and retests |
| [docs/milestone_report.md](docs/milestone_report.md) | What was demonstrated and what is still limited |

## Setup (Jetson, JetPack 6 / L4T 36.4)

System packages. These add to JetPack; nothing is replaced:

```bash
sudo apt install python3-pip python3-venv gir1.2-gst-rtsp-server-1.0 libgstrtspserver-1.0-0
```

cuDSS, which the PyTorch 2.10 Jetson wheel needs
([Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)):

```bash
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get -y install cudss
```

Python environment. `--system-site-packages` keeps the JetPack `gi` and
`tensorrt` modules visible:

```bash
cd yolo
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
.venv/bin/pip install ultralytics
PYTHON=.venv/bin/python scripts/check_env.sh     # everything should say "ok"
```

Do not `pip install torch` from PyPI: that build has no CUDA support on Jetson.

## Run

All commands are run from `yolo/`.

```bash
.venv/bin/python -m yolo_stream                      # detection, until Ctrl-C
.venv/bin/python -m yolo_stream --duration 300 --label sustained
.venv/bin/python -m yolo_stream --raw                # camera only, no model
.venv/bin/python -m yolo_stream --source test        # test pattern, no camera
.venv/bin/python -m yolo_stream --weights /path/to/drone.pt
.venv/bin/python -m yolo_stream --config my.yaml
.venv/bin/python -m yolo_stream --print-pipelines    # show the GStreamer pipelines
```

Raw mode and the test source need no PyTorch, so `python3 -m yolo_stream --raw`
also works with the system Python.

View on the Mac (same network as the Jetson; `hostname -I` on the Jetson gives the address):

```bash
/Applications/VLC.app/Contents/MacOS/VLC --network-caching=200 rtsp://<jetson-ip>:8554/detections
```

or in VLC: File → Open Network… and paste the URL. If the picture never appears,
set VLC → Settings → Input/Codecs → Live555 stream transport → "RTP over RTSP (TCP)".

Stop with Ctrl-C. Avoid `kill -9`: it leaves the camera session open and
`nvargus-daemon` may need `sudo systemctl restart nvargus-daemon` afterwards.

## Configuration

[config/default.yaml](config/default.yaml) lists every key with its default.
Unknown keys and out-of-range values are rejected at start-up. The main ones:

| Key | Default | Meaning |
|-----|---------|---------|
| `mode` | `detect` | `raw` skips the model |
| `camera.width/height/fps` | 1920 / 1080 / 30 | sensor mode requested from Argus |
| `camera.flip_method` | 0 | `nvvidconv` flip: 0 = none, 2 = 180° |
| `stream.width/height/fps` | 1280 / 720 / 15 | size and rate of the processed stream |
| `stream.port`, `stream.path` | 8554, `/detections` | RTSP address |
| `stream.encoder` | `auto` | first installed software encoder (`x264enc`, then `openh264enc`) |
| `model.weights` | `yolo26n.pt` | pretrained name (downloaded on first use) or a path |
| `model.imgsz`, `model.conf` | 640, 0.7 | inference size and confidence threshold |
| `model.classes` | all | list of class ids to keep |

### Custom drone weights and TensorRT

Point `model.weights` (or `--weights`) at the trained `.pt` file. Class names
come from the weights. To use TensorRT, export on this Jetson (engines are
device-specific) and point `model.weights` at the result:

```bash
# once: export dependencies (the automatic install fails on Jetson)
.venv/bin/pip install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.71" \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
.venv/bin/yolo export model=yolo26n.pt format=engine quantize=16 imgsz=640 device=0   # about 11 minutes
.venv/bin/python -m yolo_stream --weights yolo26n.engine --label trt-fp16
```

## Experiment folders

Each run creates `experiments/<timestamp>_<label>_<mode>/`:

| File | Contents |
|------|----------|
| `config.yaml` | the resolved configuration |
| `git.json` | commit, branch, whether the tree had uncommitted changes |
| `versions.json` | L4T, kernel, Python, PyTorch, Ultralytics, TensorRT, GStreamer, CUDA device |
| `model.json` | weights path, SHA-256, classes, device |
| `run.log` | full log including pipelines and viewer connects/disconnects |
| `metrics.csv` | one row per logging interval: fps, inference time, drops |
| `metrics.json` | final frame counts, per-stage timing (mean/p50/p95/max), exit reason |
| `samples/` | annotated frames saved during the run |

Helper scripts (all act as RTSP viewers on the Jetson itself):

| Script | Purpose |
|--------|---------|
| `scripts/check_env.sh` | list required GStreamer elements and Python modules |
| `scripts/record_stream.sh OUT.mp4 [SECONDS] [URL]` | record the stream without re-encoding |
| `scripts/rtsp_probe.py [--seconds N] [URL]` | decode the stream; report time to first frame and frame count |
| `scripts/run_with_viewer.sh SECONDS [args…]` | run the app and record a clip into the experiment folder |
| `scripts/test_reconnect_shutdown.sh` | reconnect, SIGINT timing and camera-reuse test |
| `scripts/test_sustained.sh [SECONDS]` | sustained run with a viewer and `tegrastats` log |

## Tests

```bash
python3 -m unittest discover -s tests -t . -v
```

The suite needs no camera, model or network. It covers configuration, pipeline
construction, encoder selection, the frame buffer, metrics, overlay drawing,
detection conversion, the application loop with fakes, and, when the GStreamer
bindings are present, a real capture from `videotestsrc` and an RTSP
serve/decode/reconnect round trip on localhost. Hardware tests are manual; see
[docs/test_procedure.md](docs/test_procedure.md).
