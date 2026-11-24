"""Wires capture -> (detect -> overlay) -> RTSP and owns startup/shutdown."""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .experiment import Experiment
from .frame_buffer import LatestFrameBuffer
from .metrics import Metrics
from .overlay import draw_detections, draw_hud

log = logging.getLogger(__name__)

_FRAME_TIMEOUT_S = 1.0
_STALL_LIMIT_S = 10.0  # give up if the camera delivers nothing for this long


class App:
    """`capture`, `rtsp` and `detector` can be injected for tests; otherwise
    they are built from the configuration in `run`."""

    def __init__(self, cfg: AppConfig, base_dir: Path, capture=None, rtsp=None, detector=None):
        self.cfg = cfg
        self.base_dir = base_dir
        self.buffer = LatestFrameBuffer()
        self.metrics = Metrics()
        self.stop_event = threading.Event()
        self.exit_reason = "not started"
        self._capture, self._rtsp, self._detector = capture, rtsp, detector
        self.experiment: Optional[Experiment] = None

    def request_stop(self, reason: str) -> None:
        if not self.stop_event.is_set():
            self.exit_reason = reason
            self.stop_event.set()

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, _f: self.request_stop(f"signal {signal.Signals(s).name}"))

    def _build(self) -> None:
        cfg = self.cfg
        if self._detector is None and cfg.mode == "detect":
            from .detector import Detector

            log.info("loading model %s", cfg.model.weights)
            self._detector = Detector(cfg.model)
            self._detector.warmup(cfg.stream.width, cfg.stream.height)
        if self._rtsp is None:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            from .pipelines import rtsp_launch, select_encoder
            from .rtsp import RtspServer

            Gst.init(None)
            encoder = select_encoder(cfg.stream.encoder, lambda n: Gst.ElementFactory.find(n) is not None)
            log.info("h264 encoder: %s", encoder)
            self._rtsp = RtspServer(rtsp_launch(cfg, encoder), cfg.stream.port, cfg.stream.path)
        if self._capture is None:
            from .capture import Capture
            from .pipelines import capture_pipeline

            self._capture = Capture(capture_pipeline(cfg), cfg.stream.width, cfg.stream.height, self.buffer)

    def run(self) -> int:
        """Run until stopped. Returns a process exit code (0 = clean)."""
        cfg = self.cfg
        self.experiment = exp = Experiment(cfg, self.base_dir)
        handler = logging.FileHandler(exp.log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        log.info("experiment folder: %s", exp.dir)
        self._install_signal_handlers()
        code = 0
        try:
            self._build()
            if self._detector is not None:
                exp.write_json("model.json", self._detector.identity())
            else:
                exp.write_json("model.json", {"weights": None, "note": "raw mode, no model loaded"})
            self._rtsp.start()
            self._capture.start()
            self.metrics = Metrics()  # start the clock after model load and warmup
            self._loop()
        except Exception:
            log.exception("fatal error")
            self.exit_reason = "error"
            code = 1
        finally:
            for name, part in (("capture", self._capture), ("rtsp", self._rtsp)):
                try:
                    if part is not None:
                        part.stop()
                except Exception:
                    log.exception("error stopping %s", name)
                    code = 1
            summary = self.metrics.summary(self.buffer.put_count, self.buffer.dropped_count)
            summary["exit_reason"] = self.exit_reason
            summary["exit_code"] = code
            summary["mode"] = cfg.mode
            for attr in ("clients_connected", "clients_closed", "sessions_started"):
                summary[f"rtsp_{attr}"] = getattr(self._rtsp, attr, None)
            exp.write_json("metrics.json", summary)
            exp.close()
            log.info("summary: %s", summary)
            log.info("stopped (%s), exit code %d", self.exit_reason, code)
            logging.getLogger().removeHandler(handler)
            handler.close()
        return code

    def _loop(self) -> None:
        cfg, m, exp = self.cfg, self.metrics, self.experiment
        period = 1.0 / cfg.stream.fps
        next_tick = time.monotonic()
        next_log = m.start_time + cfg.experiment.log_interval_s
        next_sample = m.start_time + min(2.0, cfg.experiment.sample_interval_s)
        last_frame = time.monotonic()
        tag = "RAW" if cfg.mode == "raw" else Path(cfg.model.weights).stem
        while not self.stop_event.is_set():
            now = time.monotonic()
            if cfg.duration_s and now - m.start_time >= cfg.duration_s:
                self.request_stop("duration reached")
                break
            frame = self.buffer.get(timeout=_FRAME_TIMEOUT_S)
            if frame is None:
                if self._capture.error:
                    raise RuntimeError(f"capture failed: {self._capture.error}")
                if time.monotonic() - last_frame > _STALL_LIMIT_S:
                    raise RuntimeError(f"no frames from camera for {_STALL_LIMIT_S:.0f} s")
                continue
            last_frame = time.monotonic()
            image = frame.image

            t0 = time.perf_counter()
            detections = self._detector.infer(image) if self._detector is not None else []
            t1 = time.perf_counter()
            draw_detections(image, detections, cfg.overlay)
            hud = f"{tag} | {m.last_fps:4.1f} fps"
            if self._detector is not None:
                hud += f" | inf {m.last_inference_ms:4.1f} ms | {len(detections)} det"
            draw_hud(image, hud, cfg.overlay)
            t2 = time.perf_counter()
            pushed = self._rtsp.push(image)
            t3 = time.perf_counter()
            m.record(t1 - t0, t2 - t1, t3 - t2, len(detections), pushed)

            now = time.monotonic()
            if now >= next_sample and exp.wants_sample():
                exp.save_sample(image, now - m.start_time)
                next_sample = now + cfg.experiment.sample_interval_s
            if now >= next_log:
                row = m.interval()
                row["buffer_dropped"] = self.buffer.dropped_count
                row["viewer"] = int(pushed is not None)
                exp.log_interval(row)
                log.info("stats: %s", row)
                next_log = now + cfg.experiment.log_interval_s

            # Pace to the stream rate; the buffer supplies the newest frame next tick.
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                self.stop_event.wait(delay)
            else:
                next_tick = time.monotonic()
