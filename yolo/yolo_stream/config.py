"""YAML configuration: typed, validated, unknown keys rejected."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional

import yaml

MODES = ("detect", "raw")
SOURCES = ("argus", "test")
ENCODERS = ("auto", "x264enc", "openh264enc", "nvv4l2h264enc")


class ConfigError(ValueError):
    pass


@dataclass
class CameraConfig:
    source: str = "argus"  # "argus" (IMX477 via nvarguscamerasrc) or "test" (videotestsrc)
    sensor_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    flip_method: int = 0


@dataclass
class StreamConfig:
    width: int = 1280
    height: int = 720
    fps: int = 15
    port: int = 8554
    path: str = "/detections"
    encoder: str = "auto"  # auto = first available software H.264 encoder
    bitrate_kbps: int = 2500
    keyframe_interval_s: float = 1.0


@dataclass
class ModelConfig:
    weights: str = "yolo26n.pt"  # pretrained name, or path to custom .pt / .engine
    imgsz: int = 640
    conf: float = 0.25
    device: str = "auto"  # auto = CUDA device 0 if available, else cpu
    half: bool = False
    classes: Optional[List[int]] = None  # None = all classes


@dataclass
class OverlayConfig:
    show_hud: bool = True
    line_thickness: int = 2
    font_scale: float = 0.6


@dataclass
class ExperimentConfig:
    root: str = "experiments"
    label: str = "run"
    log_interval_s: float = 5.0
    sample_interval_s: float = 10.0
    max_samples: int = 12


@dataclass
class AppConfig:
    mode: str = "detect"  # "detect" or "raw" (camera -> RTSP, no model)
    duration_s: float = 0.0  # 0 = run until interrupted
    camera: CameraConfig = field(default_factory=CameraConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "AppConfig":
        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(msg)

        check(self.mode in MODES, f"mode must be one of {MODES}, got {self.mode!r}")
        check(self.duration_s >= 0, "duration_s must be >= 0")
        c, s, m, e = self.camera, self.stream, self.model, self.experiment
        check(c.source in SOURCES, f"camera.source must be one of {SOURCES}")
        check(c.width > 0 and c.height > 0, "camera.width/height must be positive")
        check(c.fps > 0, "camera.fps must be positive")
        check(0 <= c.flip_method <= 7, "camera.flip_method must be 0..7")
        check(s.width > 0 and s.height > 0, "stream.width/height must be positive")
        check(s.width % 2 == 0 and s.height % 2 == 0, "stream.width/height must be even (H.264 4:2:0)")
        check(0 < s.fps <= c.fps, "stream.fps must be in 1..camera.fps")
        check(1024 <= s.port <= 65535, "stream.port must be 1024..65535")
        check(s.path.startswith("/") and len(s.path) > 1, "stream.path must start with '/'")
        check(s.encoder in ENCODERS, f"stream.encoder must be one of {ENCODERS}")
        check(s.bitrate_kbps > 0, "stream.bitrate_kbps must be positive")
        check(s.keyframe_interval_s > 0, "stream.keyframe_interval_s must be positive")
        check(bool(m.weights), "model.weights must not be empty")
        check(m.imgsz >= 32 and m.imgsz % 32 == 0, "model.imgsz must be a multiple of 32")
        check(0.0 < m.conf < 1.0, "model.conf must be in (0, 1)")
        check(e.log_interval_s > 0, "experiment.log_interval_s must be positive")
        check(e.sample_interval_s > 0, "experiment.sample_interval_s must be positive")
        check(e.max_samples >= 0, "experiment.max_samples must be >= 0")
        return self


def _build(cls: type, data: Any, where: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    kwargs = {}
    for name, value in data.items():
        sub = _SECTIONS.get(name) if cls is AppConfig else None
        kwargs[name] = _build(sub, value, f"{where}.{name}") if sub else value
    return cls(**kwargs)


_SECTIONS = {
    "camera": CameraConfig,
    "stream": StreamConfig,
    "model": ModelConfig,
    "overlay": OverlayConfig,
    "experiment": ExperimentConfig,
}


def from_dict(data: Optional[dict]) -> AppConfig:
    return _build(AppConfig, data, "config").validate()


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return from_dict(yaml.safe_load(fh))
