"""App loop and experiment-folder tests using fakes (no camera, model or network)."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from yolo_stream.app import App
from yolo_stream.config import from_dict
from yolo_stream.detector import Detection


class FakeCapture:
    def __init__(self, fps=60, fail_after=None):
        self.error = None
        self.buffer = None
        self.stopped = False
        self._fps, self._fail_after = fps, fail_after
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        n = 0
        while not self._stop.wait(1 / self._fps):
            if self._fail_after is not None and n >= self._fail_after:
                self.error = "simulated camera failure"
                return
            self.buffer.put(np.zeros((72, 128, 3), np.uint8))
            n += 1

    def stop(self):
        self._stop.set()
        self.stopped = True


class FakeRtsp:
    clients_connected = clients_closed = sessions_started = 0

    def __init__(self, viewer=True):
        self.viewer, self.frames, self.started, self.stopped = viewer, [], False, False

    def start(self):
        self.started = True

    def push(self, image):
        if not self.viewer:
            return None
        self.frames.append(image)
        return True

    def stop(self):
        self.stopped = True


class FakeDetector:
    def infer(self, image):
        return [Detection(10, 10, 60, 50, 0.8, 0, "person")]

    def identity(self):
        return {"weights": "fake.pt", "sha256": None}


def make_app(tmp, overrides=None, **parts):
    data = {
        "duration_s": 0.6,
        "stream": {"width": 128, "height": 72, "fps": 15},
        "experiment": {"root": str(tmp), "label": "unit", "log_interval_s": 0.2, "sample_interval_s": 0.2, "max_samples": 2},
    }
    data.update(overrides or {})
    app = App(from_dict(data), Path(tmp), **parts)
    parts["capture"].buffer = app.buffer
    return app


class AppTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_detect_run_writes_complete_experiment_folder(self):
        cap, rtsp = FakeCapture(), FakeRtsp()
        app = make_app(self.tmp, capture=cap, rtsp=rtsp, detector=FakeDetector())
        self.assertEqual(app.run(), 0)
        self.assertTrue(cap.stopped and rtsp.stopped)
        d = app.experiment.dir
        for name in ("config.yaml", "git.json", "versions.json", "model.json", "run.log", "metrics.csv", "metrics.json"):
            self.assertTrue((d / name).is_file(), name)
        self.assertEqual(len(list((d / "samples").glob("*.jpg"))), 2)
        m = json.loads((d / "metrics.json").read_text())
        self.assertEqual(m["exit_reason"], "duration reached")
        self.assertEqual(m["frames_processed"], len(rtsp.frames))
        self.assertEqual(m["frames_with_detections"], m["frames_processed"])
        self.assertGreater(m["frames_dropped_at_buffer"], 0)  # 60 fps source, 15 fps consumer
        # Paced to the stream rate: about 9 frames in 0.6 s, never the source's 36.
        self.assertTrue(5 <= m["frames_processed"] <= 12, m["frames_processed"])
        self.assertTrue(rtsp.frames[0][45, 10].any())  # box left edge was drawn

    def test_raw_mode_runs_without_detector(self):
        cap, rtsp = FakeCapture(), FakeRtsp(viewer=False)
        app = make_app(self.tmp, {"mode": "raw", "overlay": {"show_hud": False}}, capture=cap, rtsp=rtsp)
        self.assertEqual(app.run(), 0)
        m = json.loads((app.experiment.dir / "metrics.json").read_text())
        self.assertEqual(m["mode"], "raw")
        self.assertEqual(m["detections_total"], 0)
        self.assertEqual(m["frames_without_viewer"], m["frames_processed"])
        self.assertIsNone(json.loads((app.experiment.dir / "model.json").read_text())["weights"])

    def test_stop_request_shuts_down_promptly(self):
        cap, rtsp = FakeCapture(), FakeRtsp()
        app = make_app(self.tmp, {"duration_s": 0}, capture=cap, rtsp=rtsp, detector=FakeDetector())
        threading.Timer(0.3, app.request_stop, args=("test stop",)).start()
        t0 = time.monotonic()
        self.assertEqual(app.run(), 0)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(app.exit_reason, "test stop")
        self.assertTrue(cap.stopped and rtsp.stopped)

    def test_capture_failure_exits_nonzero_and_still_cleans_up(self):
        cap, rtsp = FakeCapture(fail_after=3), FakeRtsp()
        app = make_app(self.tmp, {"duration_s": 0}, capture=cap, rtsp=rtsp, detector=FakeDetector())
        self.assertEqual(app.run(), 1)
        self.assertTrue(cap.stopped and rtsp.stopped)
        m = json.loads((app.experiment.dir / "metrics.json").read_text())
        self.assertEqual((m["exit_reason"], m["exit_code"]), ("error", 1))


if __name__ == "__main__":
    unittest.main()
