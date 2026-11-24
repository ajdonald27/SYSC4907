"""Per-frame timing and frame accounting."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

STAGES = ("inference", "overlay", "push", "total")


def summarize(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {"count": 0}
    v = sorted(values_ms)

    def pct(p: float) -> float:
        return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]

    return {
        "count": len(v),
        "mean_ms": round(sum(v) / len(v), 3),
        "p50_ms": round(pct(0.50), 3),
        "p95_ms": round(pct(0.95), 3),
        "max_ms": round(v[-1], 3),
    }


class Metrics:
    """Collects stage timings for every processed frame. At 15 fps a
    five-minute run is 4500 samples per stage, so everything is kept."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.start_time = clock()
        self._times: Dict[str, List[float]] = {s: [] for s in STAGES}
        self.processed = 0
        self.detections = 0
        self.frames_with_detections = 0
        self.pushed = 0  # frames accepted by the RTSP appsrc
        self.push_failed = 0  # a viewer was connected but the push was refused
        self.no_client = 0  # processed while no viewer was connected
        self._interval_start = self.start_time
        self._interval_frames = 0
        self._interval_inference: List[float] = []
        self.last_fps = 0.0
        self.last_inference_ms = 0.0

    def record(self, inference_s: float, overlay_s: float, push_s: float,
               num_detections: int, pushed: Optional[bool]) -> None:
        """`pushed`: True = sent, False = refused, None = no viewer connected."""
        ms = (inference_s * 1e3, overlay_s * 1e3, push_s * 1e3)
        for stage, value in zip(STAGES, (*ms, sum(ms))):
            self._times[stage].append(value)
        self.processed += 1
        self.detections += num_detections
        self.frames_with_detections += 1 if num_detections else 0
        if pushed is None:
            self.no_client += 1
        elif pushed:
            self.pushed += 1
        else:
            self.push_failed += 1
        self._interval_frames += 1
        self._interval_inference.append(ms[0])

    def interval(self) -> Dict[str, float]:
        """Close the current reporting interval and return its figures."""
        now = self._clock()
        elapsed = max(now - self._interval_start, 1e-9)
        inf = self._interval_inference
        self.last_fps = self._interval_frames / elapsed
        self.last_inference_ms = sum(inf) / len(inf) if inf else 0.0
        row = {
            "elapsed_s": round(now - self.start_time, 2),
            "fps": round(self.last_fps, 2),
            "inference_mean_ms": round(self.last_inference_ms, 2),
            "inference_max_ms": round(max(inf), 2) if inf else 0.0,
            "processed": self.processed,
        }
        self._interval_start, self._interval_frames, self._interval_inference = now, 0, []
        return row

    def summary(self, captured: int, buffer_dropped: int) -> Dict[str, object]:
        elapsed = max(self._clock() - self.start_time, 1e-9)
        return {
            "elapsed_s": round(elapsed, 2),
            "frames_captured": captured,
            "frames_processed": self.processed,
            "frames_dropped_at_buffer": buffer_dropped,
            "frames_pushed_to_rtsp": self.pushed,
            "frames_push_failed": self.push_failed,
            "frames_without_viewer": self.no_client,
            "capture_fps": round(captured / elapsed, 2),
            "processed_fps": round(self.processed / elapsed, 2),
            "detections_total": self.detections,
            "frames_with_detections": self.frames_with_detections,
            "timing": {stage: summarize(v) for stage, v in self._times.items()},
        }
