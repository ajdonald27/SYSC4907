#!/usr/bin/env bash
# Report what this milestone depends on. Read-only; installs nothing.
py=${PYTHON:-python3}
echo "== platform"; head -1 /etc/nv_tegra_release 2>/dev/null; uname -r
echo "== gstreamer elements"
for e in nvarguscamerasrc nvvidconv appsink appsrc videoconvert x264enc openh264enc nvv4l2h264enc h264parse rtph264pay rtspsrc avdec_h264 mp4mux; do
  if gst-inspect-1.0 "$e" >/dev/null 2>&1; then echo "  ok       $e"; else echo "  MISSING  $e"; fi
done
echo "== python ($py)"
"$py" - <<'EOF'
import importlib
def show(name, fn):
    try: print(f"  ok       {name}: {fn()}")
    except Exception as exc: print(f"  MISSING  {name}: {exc}")
def gst():
    import gi; gi.require_version("Gst", "1.0"); from gi.repository import Gst; return Gst.version_string()
def rtsp():
    import gi; gi.require_version("GstRtspServer", "1.0"); from gi.repository import GstRtspServer; return "bindings present"
def torch_():
    import torch; return f"{torch.__version__} cuda={torch.cuda.is_available()}"
show("gi/Gst", gst); show("GstRtspServer", rtsp)
for m in ("numpy", "yaml", "cv2", "tensorrt", "ultralytics"):
    show(m, lambda m=m: importlib.import_module(m).__version__)
show("torch", torch_)
EOF
