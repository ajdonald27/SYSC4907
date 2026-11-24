"""YOLO inference via Ultralytics. Imports are lazy so raw-camera mode and the
unit tests work without torch/ultralytics installed."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .config import ModelConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str


def to_detections(result: Any) -> List[Detection]:
    """Convert one Ultralytics `Results` object to plain detections."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    xyxy = _numpy(boxes.xyxy)
    conf = _numpy(boxes.conf)
    cls = _numpy(boxes.cls)
    out = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        k = int(k)
        out.append(Detection(int(x1), int(y1), int(x2), int(y2), float(c), k, str(names.get(k, k))))
    return out


def _numpy(t: Any) -> np.ndarray:
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def sha256_of(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Detector:
    def __init__(self, cfg: ModelConfig, model: Any = None):
        """`model` may be injected (tests); otherwise weights are loaded with
        Ultralytics, which downloads pretrained names such as yolo26n.pt."""
        self.cfg = cfg
        self.device = cfg.device
        if model is None:
            import torch
            from ultralytics import YOLO

            if cfg.device == "auto":
                self.device = "0" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                log.warning("running inference on CPU; CUDA is not available to torch")
            model = YOLO(cfg.weights, task="detect")
        self.model = model

    def infer(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            image,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            device=self.device,
            half=self.cfg.half,
            classes=self.cfg.classes,
            verbose=False,
        )
        return to_detections(results[0])

    def warmup(self, width: int, height: int, runs: int = 3) -> None:
        """First calls include CUDA/cuDNN initialisation; keep them out of the metrics."""
        blank = np.zeros((height, width, 3), np.uint8)
        for _ in range(runs):
            self.infer(blank)

    def identity(self) -> dict:
        """What model actually ran, for the experiment record."""
        path = getattr(self.model, "ckpt_path", None) or self.cfg.weights
        names = getattr(self.model, "names", None) or {}
        return {
            "weights": self.cfg.weights,
            "resolved_path": str(path),
            "sha256": sha256_of(path),
            "task": getattr(self.model, "task", None),
            "num_classes": len(names),
            "class_names": [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names),
            "device": self.device,
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.conf,
            "half": self.cfg.half,
        }
