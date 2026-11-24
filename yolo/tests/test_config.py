import tempfile
import unittest
from pathlib import Path

from yolo_stream.config import AppConfig, ConfigError, from_dict, load_config

DEFAULT_YAML = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class ConfigTests(unittest.TestCase):
    def test_empty_config_gives_milestone_defaults(self):
        cfg = from_dict(None)
        self.assertEqual((cfg.camera.width, cfg.camera.height, cfg.camera.fps), (1920, 1080, 30))
        self.assertEqual((cfg.stream.width, cfg.stream.height, cfg.stream.fps), (1280, 720, 15))
        self.assertEqual(cfg.stream.path, "/detections")
        self.assertEqual(cfg.model.weights, "yolo26n.pt")
        self.assertEqual(cfg.mode, "detect")

    def test_shipped_default_yaml_matches_code_defaults(self):
        self.assertEqual(load_config(DEFAULT_YAML).to_dict(), AppConfig().to_dict())

    def test_partial_override_keeps_other_defaults(self):
        cfg = from_dict({"mode": "raw", "stream": {"fps": 10}, "model": {"weights": "drone.pt"}})
        self.assertEqual(cfg.mode, "raw")
        self.assertEqual(cfg.stream.fps, 10)
        self.assertEqual(cfg.stream.width, 1280)
        self.assertEqual(cfg.model.weights, "drone.pt")

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            from_dict({"stream": {"fpz": 15}})
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            from_dict({"tracker": {}})

    def test_section_must_be_mapping(self):
        with self.assertRaisesRegex(ConfigError, "expected a mapping"):
            from_dict({"camera": 3})

    def test_invalid_values_are_rejected(self):
        bad = [
            {"mode": "track"},
            {"camera": {"source": "usb"}},
            {"stream": {"fps": 60}},  # faster than the camera
            {"stream": {"width": 1281}},  # odd width
            {"stream": {"path": "detections"}},
            {"stream": {"encoder": "nvenc"}},
            {"stream": {"port": 80}},
            {"model": {"conf": 1.5}},
            {"model": {"imgsz": 500}},
            {"duration_s": -1},
        ]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ConfigError):
                from_dict(data)

    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.yaml"
            path.write_text("mode: raw\nduration_s: 30\n")
            cfg = load_config(path)
        self.assertEqual((cfg.mode, cfg.duration_s), ("raw", 30))


if __name__ == "__main__":
    unittest.main()
