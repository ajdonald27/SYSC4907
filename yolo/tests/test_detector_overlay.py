import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from yolo_stream.config import ModelConfig, OverlayConfig
from yolo_stream.detector import Detection, Detector, sha256_of, to_detections
from yolo_stream.overlay import color_for, draw_detections, draw_hud


class FakeBoxes:
    def __init__(self, rows):
        a = np.array(rows, dtype=np.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]

    def __len__(self):
        return len(self.conf)


class FakeModel:
    """Stands in for ultralytics.YOLO; records the arguments it was called with."""

    names = {0: "person", 4: "airplane"}
    task = "detect"

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=FakeBoxes(self.rows), names=self.names)]


class DetectorTests(unittest.TestCase):
    def test_results_become_detections(self):
        model = FakeModel([[10.6, 20.2, 110.9, 220.0, 0.91, 0], [300, 40, 360, 90, 0.4, 4]])
        det = Detector(ModelConfig(conf=0.3, imgsz=320, classes=[0, 4]), model=model)
        out = det.infer(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(out[0], Detection(10, 20, 110, 220, out[0].confidence, 0, "person"))
        self.assertAlmostEqual(out[0].confidence, 0.91, places=5)
        self.assertEqual(out[1].label, "airplane")
        call = model.calls[0]
        self.assertEqual((call["conf"], call["imgsz"], call["classes"]), (0.3, 320, [0, 4]))
        self.assertFalse(call["verbose"])

    def test_no_boxes(self):
        self.assertEqual(Detector(ModelConfig(), model=FakeModel([])).infer(np.zeros((8, 8, 3), np.uint8)), [])
        self.assertEqual(to_detections(SimpleNamespace(boxes=None, names={})), [])

    def test_unknown_class_id_falls_back_to_number(self):
        out = Detector(ModelConfig(), model=FakeModel([[0, 0, 5, 5, 0.5, 9]])).infer(np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(out[0].label, "9")

    def test_identity_records_model_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / "drone.pt"
            weights.write_bytes(b"not a real model")
            model = FakeModel([])
            model.ckpt_path = str(weights)
            ident = Detector(ModelConfig(weights=str(weights)), model=model).identity()
            self.assertEqual(ident["sha256"], sha256_of(weights))
        self.assertEqual(len(ident["sha256"]), 64)
        self.assertEqual(ident["class_names"], ["person", "airplane"])
        self.assertIsNone(sha256_of("/nonexistent/weights.pt"))


class OverlayTests(unittest.TestCase):
    def test_box_is_drawn_in_class_colour(self):
        img = np.zeros((200, 300, 3), np.uint8)
        draw_detections(img, [Detection(50, 80, 150, 160, 0.9, 0, "person")], OverlayConfig())
        self.assertEqual(tuple(int(v) for v in img[120, 50]), color_for(0))  # left edge
        self.assertEqual(tuple(int(v) for v in img[120, 100]), (0, 0, 0))  # interior untouched

    def test_no_detections_leaves_frame_unchanged(self):
        img = np.full((100, 100, 3), 7, np.uint8)
        draw_detections(img, [], OverlayConfig())
        self.assertTrue((img == 7).all())

    def test_boxes_at_frame_edges_do_not_crash(self):
        img = np.zeros((100, 100, 3), np.uint8)
        dets = [Detection(-20, -20, 30, 30, 0.5, 1, "a"), Detection(80, 80, 140, 140, 0.5, 2, "long label here")]
        draw_detections(img, dets, OverlayConfig())
        self.assertGreater(int(img.sum()), 0)

    def test_hud_can_be_disabled(self):
        img = np.zeros((100, 400, 3), np.uint8)
        draw_hud(img, "RAW | 15.0 fps", OverlayConfig(show_hud=False))
        self.assertEqual(int(img.sum()), 0)
        draw_hud(img, "RAW | 15.0 fps", OverlayConfig())
        self.assertGreater(int(img.sum()), 0)


if __name__ == "__main__":
    unittest.main()
