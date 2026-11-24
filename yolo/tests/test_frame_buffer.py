import threading
import time
import unittest

from yolo_stream.frame_buffer import LatestFrameBuffer


class LatestFrameBufferTests(unittest.TestCase):
    def test_get_returns_newest_and_counts_drops(self):
        buf = LatestFrameBuffer()
        for i in range(5):
            buf.put(f"img{i}")
        frame = buf.get(timeout=0)
        self.assertEqual((frame.image, frame.seq), ("img4", 5))
        self.assertEqual(buf.put_count, 5)
        self.assertEqual(buf.dropped_count, 4)

    def test_frame_is_delivered_once(self):
        buf = LatestFrameBuffer()
        buf.put("a")
        self.assertIsNotNone(buf.get(timeout=0))
        self.assertIsNone(buf.get(timeout=0))
        self.assertEqual(buf.dropped_count, 0)

    def test_get_times_out_when_empty(self):
        buf = LatestFrameBuffer()
        t0 = time.monotonic()
        self.assertIsNone(buf.get(timeout=0.05))
        self.assertGreaterEqual(time.monotonic() - t0, 0.04)

    def test_get_wakes_when_frame_arrives(self):
        buf = LatestFrameBuffer()
        threading.Timer(0.05, buf.put, args=("late",)).start()
        frame = buf.get(timeout=2)
        self.assertEqual(frame.image, "late")

    def test_slow_consumer_accounting_is_consistent(self):
        buf = LatestFrameBuffer()
        total, received = 500, []

        def produce():
            for i in range(total):
                buf.put(i)
                time.sleep(0.0002)

        t = threading.Thread(target=produce)
        t.start()
        while t.is_alive() or received[-1:] != [total - 1]:
            frame = buf.get(timeout=0.5)
            if frame is not None:
                received.append(frame.image)
                time.sleep(0.001)
        t.join()
        self.assertEqual(received, sorted(set(received)))  # in order, no repeats
        self.assertEqual(len(received) + buf.dropped_count, total)


if __name__ == "__main__":
    unittest.main()
