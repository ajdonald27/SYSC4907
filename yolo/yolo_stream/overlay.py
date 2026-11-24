"""Bounding-box and status overlays, drawn in place on BGR frames."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from .config import OverlayConfig
from .detector import Detection

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (56, 56, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
    (23, 204, 146), (134, 219, 61), (187, 212, 0), (255, 194, 0), (255, 115, 100),
    (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255),
)


def color_for(class_id: int) -> Tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _label(image: np.ndarray, text: str, x: int, y: int, bg, cfg: OverlayConfig) -> None:
    """Text on a filled background with its top-left corner clamped into the frame."""
    (tw, th), base = cv2.getTextSize(text, _FONT, cfg.font_scale, 1)
    h, w = image.shape[:2]
    x = max(0, min(x, w - tw - 4))
    y = max(0, min(y, h - th - base - 4))
    cv2.rectangle(image, (x, y), (x + tw + 4, y + th + base + 4), bg, -1)
    cv2.putText(image, text, (x + 2, y + th + 2), _FONT, cfg.font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_detections(image: np.ndarray, detections: Iterable[Detection], cfg: OverlayConfig) -> np.ndarray:
    for d in detections:
        color = color_for(d.class_id)
        cv2.rectangle(image, (d.x1, d.y1), (d.x2, d.y2), color, cfg.line_thickness)
        (_, th), base = cv2.getTextSize("X", _FONT, cfg.font_scale, 1)
        _label(image, f"{d.label} {d.confidence:.2f}", d.x1, d.y1 - th - base - 4, color, cfg)
    return image


def draw_hud(image: np.ndarray, text: str, cfg: OverlayConfig) -> np.ndarray:
    if cfg.show_hud:
        _label(image, text, 8, 8, (0, 0, 0), cfg)
    return image
