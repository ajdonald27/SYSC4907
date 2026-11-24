import unittest

from yolo_stream.config import ConfigError, from_dict
from yolo_stream.pipelines import capture_pipeline, rtsp_launch, select_encoder


class CapturePipelineTests(unittest.TestCase):
    def test_argus_pipeline_matches_confirmed_camera_settings(self):
        p = capture_pipeline(from_dict(None))
        self.assertIn("nvarguscamerasrc sensor-id=0", p)
        self.assertIn("video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1", p)
        self.assertIn("nvvidconv flip-method=0", p)
        self.assertIn("width=1280,height=720,format=BGRx", p)
        self.assertIn("format=BGR ! appsink", p)

    def test_appsink_is_bounded_and_drops(self):
        p = capture_pipeline(from_dict(None))
        self.assertIn("max-buffers=1 drop=true", p)

    def test_test_source_needs_no_camera(self):
        p = capture_pipeline(from_dict({"camera": {"source": "test"}}))
        self.assertIn("videotestsrc", p)
        self.assertNotIn("nvarguscamerasrc", p)
        self.assertIn("width=1280,height=720", p)


class EncoderSelectionTests(unittest.TestCase):
    def test_auto_prefers_x264_then_openh264(self):
        self.assertEqual(select_encoder("auto", lambda n: True), "x264enc")
        self.assertEqual(select_encoder("auto", lambda n: n == "openh264enc"), "openh264enc")

    def test_auto_never_picks_hardware_encoder(self):
        with self.assertRaises(ConfigError):
            select_encoder("auto", lambda n: n == "nvv4l2h264enc")

    def test_explicit_encoder_must_be_installed(self):
        self.assertEqual(select_encoder("nvv4l2h264enc", lambda n: True), "nvv4l2h264enc")
        with self.assertRaisesRegex(ConfigError, "not installed"):
            select_encoder("nvv4l2h264enc", lambda n: n == "x264enc")


class RtspLaunchTests(unittest.TestCase):
    def test_x264_launch(self):
        p = rtsp_launch(from_dict(None), "x264enc")
        self.assertTrue(p.startswith("( appsrc name=src") and p.endswith(")"))
        self.assertIn("caps=video/x-raw,format=BGR,width=1280,height=720,framerate=15/1", p)
        self.assertIn("max-buffers=2 leaky-type=downstream", p)
        self.assertIn("x264enc tune=zerolatency speed-preset=ultrafast bitrate=2500 key-int-max=15", p)
        self.assertIn("rtph264pay name=pay0 pt=96 config-interval=1", p)

    def test_keyframe_interval_follows_fps(self):
        cfg = from_dict({"stream": {"fps": 10, "keyframe_interval_s": 2.0, "bitrate_kbps": 1000}})
        self.assertIn("key-int-max=20", rtsp_launch(cfg, "x264enc"))
        self.assertIn("openh264enc bitrate=1000000 gop-size=20", rtsp_launch(cfg, "openh264enc"))

    def test_unknown_encoder_rejected(self):
        with self.assertRaises(ConfigError):
            rtsp_launch(from_dict(None), "vp8enc")


if __name__ == "__main__":
    unittest.main()
