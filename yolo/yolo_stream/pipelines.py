"""GStreamer pipeline strings. Pure functions so they can be unit-tested and
copy-pasted into gst-launch-1.0 when troubleshooting."""

from __future__ import annotations

from typing import Callable, Sequence

from .config import AppConfig, ConfigError

SOFTWARE_ENCODERS: Sequence[str] = ("x264enc", "openh264enc")
APPSINK_NAME = "sink"
APPSRC_NAME = "src"


def capture_pipeline(cfg: AppConfig) -> str:
    """Camera -> BGR frames at stream resolution in an appsink.

    Scaling and flipping run in nvvidconv (hardware); only the final
    BGRx -> BGR repack is on the CPU. The appsink keeps one buffer and drops
    the rest so a slow consumer never builds latency.
    """
    cam, out = cfg.camera, cfg.stream
    sink = f"appsink name={APPSINK_NAME} max-buffers=1 drop=true sync=false emit-signals=false"
    if cam.source == "test":
        return (
            f"videotestsrc is-live=true pattern=ball ! "
            f"video/x-raw,width={out.width},height={out.height},framerate={cam.fps}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! {sink}"
        )
    return (
        f"nvarguscamerasrc sensor-id={cam.sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={cam.width},height={cam.height},"
        f"format=NV12,framerate={cam.fps}/1 ! "
        f"nvvidconv flip-method={cam.flip_method} ! "
        f"video/x-raw,width={out.width},height={out.height},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! {sink}"
    )


def select_encoder(requested: str, is_available: Callable[[str], bool]) -> str:
    """Resolve the configured encoder to an installed GStreamer element.

    "auto" only considers software encoders: the Orin Nano has no NVENC, so a
    hardware encoder is used only when explicitly requested and present.
    """
    if requested == "auto":
        for name in SOFTWARE_ENCODERS:
            if is_available(name):
                return name
        raise ConfigError(f"no software H.264 encoder found (tried {list(SOFTWARE_ENCODERS)})")
    if not is_available(requested):
        raise ConfigError(f"encoder {requested!r} is not installed; use encoder: auto")
    return requested


def _encoder_chain(encoder: str, cfg: AppConfig) -> str:
    s = cfg.stream
    gop = max(1, round(s.fps * s.keyframe_interval_s))
    if encoder == "x264enc":
        return (
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast "
            f"bitrate={s.bitrate_kbps} key-int-max={gop}"
        )
    if encoder == "openh264enc":
        return (
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"openh264enc bitrate={s.bitrate_kbps * 1000} gop-size={gop}"
        )
    if encoder == "nvv4l2h264enc":
        return (
            f"videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! "
            f"video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvv4l2h264enc bitrate={s.bitrate_kbps * 1000} iframeinterval={gop} "
            f"insert-sps-pps=true"
        )
    raise ConfigError(f"unsupported encoder {encoder!r}")


def rtsp_launch(cfg: AppConfig, encoder: str) -> str:
    """Launch line for the RTSP media factory: appsrc (BGR) -> H.264 -> RTP.

    The appsrc holds at most two frames and leaks old ones, so the encoder
    always works on recent frames. Frequent keyframes plus SPS/PPS on every
    one let a reconnecting viewer start decoding within a second.
    """
    s = cfg.stream
    caps = f"video/x-raw,format=BGR,width={s.width},height={s.height},framerate={s.fps}/1"
    return (
        f"( appsrc name={APPSRC_NAME} is-live=true format=time do-timestamp=true "
        f"block=false max-buffers=2 leaky-type=downstream caps={caps} ! "
        f"{_encoder_chain(encoder, cfg)} ! "
        f"h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )
