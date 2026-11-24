"""GStreamer integration tests using videotestsrc (no camera needed).
Skipped where the GStreamer Python bindings or RTSP server library are missing."""

import socket
import threading
import time
import unittest

from yolo_stream.config import from_dict
from yolo_stream.frame_buffer import LatestFrameBuffer
from yolo_stream.pipelines import capture_pipeline, rtsp_launch, select_encoder

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    HAVE_GST = True
except (ImportError, ValueError):
    HAVE_GST = False

try:
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import GstRtspServer  # noqa: F401

    HAVE_RTSP = HAVE_GST
except (NameError, ImportError, ValueError):
    HAVE_RTSP = False

CFG = {"camera": {"source": "test"}, "stream": {"width": 320, "height": 240, "fps": 15}}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_GST, "GStreamer Python bindings not available")
class CaptureTests(unittest.TestCase):
    def test_capture_delivers_bgr_frames_and_stops_cleanly(self):
        from yolo_stream.capture import Capture

        cfg = from_dict(CFG)
        buf = LatestFrameBuffer()
        cap = Capture(capture_pipeline(cfg), 320, 240, buf)
        cap.start()
        try:
            frame = buf.get(timeout=5)
            self.assertIsNotNone(frame, "no frame within 5 s")
            self.assertEqual(frame.image.shape, (240, 320, 3))
            time.sleep(0.5)
        finally:
            cap.stop()
        self.assertIsNone(cap.error)
        self.assertGreater(buf.put_count, 5)
        self.assertGreater(buf.dropped_count, 0)  # nobody was reading during the sleep

    def test_bad_pipeline_reports_error(self):
        from yolo_stream.capture import Capture

        buf = LatestFrameBuffer()
        cap = Capture("filesrc location=/nonexistent ! appsink name=sink", 320, 240, buf)
        try:
            try:
                cap.start()
            except Exception:
                return  # failing at start is acceptable
            deadline = time.monotonic() + 5
            while cap.error is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNotNone(cap.error)
        finally:
            cap.stop()


@unittest.skipUnless(HAVE_RTSP, "GstRtspServer bindings not available (see README setup)")
class RtspRoundTripTests(unittest.TestCase):
    """Serve frames on localhost and read them back with an RTSP client, twice,
    to cover viewer reconnection."""

    def _watch(self, url: str, seconds: float) -> int:
        """Play the stream with a client pipeline; return decoded frame count."""
        client = Gst.parse_launch(
            f"rtspsrc location={url} protocols=tcp latency=0 ! rtph264depay ! h264parse ! "
            f"avdec_h264 ! fakesink name=out sync=false signal-handoffs=true"
        )
        count = [0]
        client.get_by_name("out").connect("handoff", lambda *_: count.__setitem__(0, count[0] + 1))
        client.set_state(Gst.State.PLAYING)
        time.sleep(seconds)
        client.set_state(Gst.State.NULL)
        return count[0]

    def test_stream_decodes_and_viewer_can_reconnect(self):
        import numpy as np

        from yolo_stream.rtsp import RtspServer

        cfg = from_dict(CFG)
        port = free_port()
        encoder = select_encoder("auto", lambda n: Gst.ElementFactory.find(n) is not None)
        server = RtspServer(rtsp_launch(cfg, encoder), port, "/detections")
        server.start()
        stop = threading.Event()

        def feed():
            i = 0
            while not stop.wait(1 / 15):
                img = np.full((240, 320, 3), (i * 8) % 255, np.uint8)
                server.push(img)
                i += 1

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        try:
            url = f"rtsp://127.0.0.1:{port}/detections"
            first = self._watch(url, 4)
            self.assertGreater(first, 10, "first viewer decoded too few frames")
            deadline = time.monotonic() + 5
            while server.has_viewer and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse(server.has_viewer, "media not released after viewer left")
            second = self._watch(url, 4)
            self.assertGreater(second, 10, "reconnected viewer decoded too few frames")
            self.assertEqual(server.sessions_started, 2)
        finally:
            stop.set()
            feeder.join(timeout=2)
            server.stop()


if __name__ == "__main__":
    unittest.main()
