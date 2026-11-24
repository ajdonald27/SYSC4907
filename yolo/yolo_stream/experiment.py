"""Experiment folders: one unique directory per run holding everything needed
to say what ran and how it performed."""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import AppConfig

log = logging.getLogger(__name__)

_PACKAGES = ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "PyYAML", "tensorrt")


def _run(cmd: list, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(repo_dir: Path) -> Dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], repo_dir)
    status = _run(["git", "status", "--porcelain"], repo_dir)
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_dir),
        "dirty": bool(status) if status is not None else None,
    }


def collect_versions() -> Dict[str, Any]:
    v: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "kernel": platform.release(),
        "machine": platform.machine(),
    }
    try:
        v["l4t"] = Path("/etc/nv_tegra_release").read_text().splitlines()[0]
    except OSError:
        v["l4t"] = None
    for pkg in _PACKAGES:
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    try:
        import cv2

        v["cv2"] = cv2.__version__
    except ImportError:
        v["cv2"] = None
    if v["tensorrt"] is None:  # the JetPack apt package has no pip metadata
        try:
            import tensorrt

            v["tensorrt"] = tensorrt.__version__
        except ImportError:
            pass
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        v["gstreamer"] = Gst.version_string()
    except (ImportError, ValueError):
        v["gstreamer"] = None
    if v["torch"] is not None:
        try:
            import torch

            v["cuda_available"] = torch.cuda.is_available()
            v["cuda"] = torch.version.cuda
            v["cuda_device"] = torch.cuda.get_device_name(0) if v["cuda_available"] else None
        except Exception as exc:  # a broken torch install must not stop a raw-mode run
            v["torch_error"] = repr(exc)
    return v


class Experiment:
    def __init__(self, cfg: AppConfig, base_dir: Path, now: Optional[datetime] = None):
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        root = Path(cfg.experiment.root)
        root = root if root.is_absolute() else base_dir / root
        self.dir = root / f"{stamp}_{cfg.experiment.label}_{cfg.mode}"
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "samples").mkdir()
        self._cfg = cfg
        self._samples = 0
        self._csv_file = None
        self._csv = None
        self.write_yaml("config.yaml", cfg.to_dict())
        self.write_json("git.json", git_info(base_dir))
        self.write_json("versions.json", collect_versions())

    @property
    def log_path(self) -> Path:
        return self.dir / "run.log"

    def write_json(self, name: str, data: Any) -> None:
        (self.dir / name).write_text(json.dumps(data, indent=2, default=str) + "\n")

    def write_yaml(self, name: str, data: Any) -> None:
        (self.dir / name).write_text(yaml.safe_dump(data, sort_keys=False))

    def log_interval(self, row: Dict[str, Any]) -> None:
        if self._csv is None:
            self._csv_file = (self.dir / "metrics.csv").open("w", newline="")
            self._csv = csv.DictWriter(self._csv_file, fieldnames=list(row))
            self._csv.writeheader()
        self._csv.writerow(row)
        self._csv_file.flush()

    def wants_sample(self) -> bool:
        return self._samples < self._cfg.experiment.max_samples

    def save_sample(self, image, elapsed_s: float) -> None:
        import cv2

        self._samples += 1
        name = f"frame_{self._samples:03d}_t{elapsed_s:06.1f}s.jpg"
        cv2.imwrite(str(self.dir / "samples" / name), image)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
