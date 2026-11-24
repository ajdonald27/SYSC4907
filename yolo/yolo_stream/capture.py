"""GStreamer capture: pulls BGR frames from an appsink into a LatestFrameBuffer."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from .frame_buffer import LatestFrameBuffer  # noqa: E402
from .pipelines import APPSINK_NAME  # noqa: E402

log = logging.getLogger(__name__)

_PULL_TIMEOUT_NS = 200 * Gst.MSECOND


class CaptureError(RuntimeError):
    pass


class Capture:
    def __init__(self, pipeline: str, width: int, height: int, buffer: LatestFrameBuffer):
        Gst.init(None)
        self._description = pipeline
        self._width, self._height = width, height
        self._buffer = buffer
        self._pipeline: Optional[Gst.Pipeline] = None
        self._sink = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.error: Optional[str] = None  # set if the pipeline fails or ends

    def start(self) -> None:
        log.info("capture pipeline: %s", self._description)
        self._pipeline = Gst.parse_launch(self._description)
        self._sink = self._pipeline.get_by_name(APPSINK_NAME)
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._teardown()
            raise CaptureError("capture pipeline failed to start")
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        bus = self._pipeline.get_bus()
        while not self._stop.is_set():
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is not None:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    self.error = f"{err.message} ({debug})"
                else:
                    self.error = "capture pipeline reached end of stream"
                log.error("capture stopped: %s", self.error)
                return
            sample = self._sink.emit("try-pull-sample", _PULL_TIMEOUT_NS)
            if sample is not None:
                self._buffer.put(self._to_array(sample))

    def _to_array(self, sample) -> np.ndarray:
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise CaptureError("could not map capture buffer")
        try:
            # Rows may be padded; derive the stride from the buffer size.
            stride = info.size // self._height
            rows = np.frombuffer(info.data, np.uint8, self._height * stride)
            return rows.reshape(self._height, stride)[:, : self._width * 3].reshape(
                self._height, self._width, 3
            ).copy()
        finally:
            buf.unmap(info)

    def stop(self) -> None:
        """Stop the thread and release the camera (state NULL frees the Argus session)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._teardown()

    def _teardown(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._sink = None
