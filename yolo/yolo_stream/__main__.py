"""Command line: python3 -m yolo_stream --config config/default.yaml"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import queue
import sys
from pathlib import Path

from .app import App
from .config import ConfigError, load_config

YOLO_DIR = Path(__file__).resolve().parent.parent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="yolo_stream", description=__doc__)
    p.add_argument("--config", default=str(YOLO_DIR / "config" / "default.yaml"))
    p.add_argument("--raw", action="store_true", help="raw-camera mode: stream without running the model")
    p.add_argument("--source", choices=("argus", "test"), help="override camera.source")
    p.add_argument("--weights", help="override model.weights (pretrained name or path to .pt/.engine)")
    p.add_argument("--duration", type=float, help="stop after this many seconds (0 = until Ctrl-C)")
    p.add_argument("--label", help="experiment label used in the folder name")
    p.add_argument("--print-pipelines", action="store_true", help="print the GStreamer pipelines and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Console logging goes through a queue: when stdout is redirected to a file,
    # a direct write could block the processing loop on a slow disk.
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log_queue: queue.Queue = queue.Queue()
    listener = logging.handlers.QueueListener(log_queue, console)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.handlers.QueueHandler(log_queue))
    listener.start()
    try:
        return _run(args)
    finally:
        listener.stop()


def _run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
        if args.raw:
            cfg.mode = "raw"
        if args.source:
            cfg.camera.source = args.source
        if args.weights:
            cfg.model.weights = args.weights
        if args.duration is not None:
            cfg.duration_s = args.duration
        if args.label:
            cfg.experiment.label = args.label
        cfg.validate()
    except (ConfigError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.print_pipelines:
        from .pipelines import capture_pipeline, rtsp_launch

        print("capture:", capture_pipeline(cfg))
        for enc in ("x264enc", "openh264enc"):
            print(f"rtsp ({enc}):", rtsp_launch(cfg, enc))
        return 0
    return App(cfg, YOLO_DIR).run()


if __name__ == "__main__":
    sys.exit(main())
