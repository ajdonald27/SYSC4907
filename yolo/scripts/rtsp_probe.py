#!/usr/bin/env python3
"""Connect to the RTSP stream as a viewer, decode it, and report the time to
the first decoded frame and the frame count. Prints one JSON line.
Usage: scripts/rtsp_probe.py [--seconds 8] [URL]"""

import argparse
import json
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="rtsp://127.0.0.1:8554/detections")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    Gst.init(None)
    pipe = Gst.parse_launch(
        f"rtspsrc location={args.url} protocols=tcp latency=0 ! rtph264depay ! h264parse ! "
        f"avdec_h264 ! fakesink name=out sync=false signal-handoffs=true"
    )
    state = {"first": None, "frames": 0}

    def on_frame(*_):
        if state["first"] is None:
            state["first"] = time.monotonic()
        state["frames"] += 1

    pipe.get_by_name("out").connect("handoff", on_frame)
    start = time.monotonic()
    pipe.set_state(Gst.State.PLAYING)
    error = None
    while time.monotonic() - start < args.seconds:
        msg = pipe.get_bus().timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR)
        if msg is not None:
            error = msg.parse_error()[0].message
            break
    end = time.monotonic()
    pipe.set_state(Gst.State.NULL)
    first = state["first"]
    result = {
        "url": args.url,
        "watched_s": round(end - start, 2),
        "time_to_first_frame_s": round(first - start, 2) if first else None,
        "frames_decoded": state["frames"],
        "fps_after_first_frame": round((state["frames"] - 1) / (end - first), 2) if first and end > first else None,
        "error": error,
    }
    print(json.dumps(result))
    return 0 if state["frames"] > 0 and error is None else 1


if __name__ == "__main__":
    sys.exit(main())
