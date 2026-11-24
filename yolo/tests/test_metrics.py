import unittest

from yolo_stream.metrics import Metrics, summarize


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class SummarizeTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(summarize([]), {"count": 0})

    def test_percentiles(self):
        s = summarize([float(i) for i in range(1, 101)])
        self.assertEqual(s["count"], 100)
        self.assertEqual(s["mean_ms"], 50.5)
        self.assertEqual(s["max_ms"], 100.0)
        self.assertAlmostEqual(s["p50_ms"], 50.5, delta=0.6)
        self.assertAlmostEqual(s["p95_ms"], 95.0, delta=1.0)


class MetricsTests(unittest.TestCase):
    def test_summary_accounts_for_every_frame(self):
        clock = FakeClock()
        m = Metrics(clock)
        m.record(0.020, 0.001, 0.002, 2, True)
        m.record(0.030, 0.001, 0.002, 0, None)
        m.record(0.040, 0.001, 0.002, 1, False)
        clock.t += 2.0
        s = m.summary(captured=6, buffer_dropped=3)
        self.assertEqual(s["frames_processed"], 3)
        self.assertEqual(s["frames_captured"], 6)
        self.assertEqual(s["frames_dropped_at_buffer"], 3)
        self.assertEqual((s["frames_pushed_to_rtsp"], s["frames_without_viewer"], s["frames_push_failed"]), (1, 1, 1))
        self.assertEqual(s["detections_total"], 3)
        self.assertEqual(s["frames_with_detections"], 2)
        self.assertEqual(s["processed_fps"], 1.5)
        self.assertEqual(s["capture_fps"], 3.0)
        self.assertEqual(s["timing"]["inference"]["mean_ms"], 30.0)
        self.assertEqual(s["timing"]["total"]["max_ms"], 43.0)

    def test_interval_reports_rate_and_resets(self):
        clock = FakeClock()
        m = Metrics(clock)
        for _ in range(30):
            m.record(0.010, 0, 0, 0, True)
        clock.t += 2.0
        row = m.interval()
        self.assertEqual(row["fps"], 15.0)
        self.assertEqual(row["inference_mean_ms"], 10.0)
        clock.t += 1.0
        self.assertEqual(m.interval()["fps"], 0.0)
        self.assertEqual(m.processed, 30)


if __name__ == "__main__":
    unittest.main()
