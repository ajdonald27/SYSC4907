"""RTSP server fed from Python through an appsrc.

One shared media pipeline is created when the first viewer connects and torn
down when the last one leaves; the next viewer gets a fresh pipeline. `push`
is a no-op while nobody is watching, so the encoder only runs on demand.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402

from .pipelines import APPSRC_NAME  # noqa: E402

log = logging.getLogger(__name__)


class RtspServer:
    def __init__(self, launch: str, port: int, path: str):
        Gst.init(None)
        self._launch, self._port, self._path = launch, port, path
        self._lock = threading.Lock()
        self._appsrc = None
        self._context = GLib.MainContext.new()
        self._loop = GLib.MainLoop.new(self._context, False)
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[GstRtspServer.RTSPServer] = None
        self._source_id = 0
        self.clients_connected = 0  # total connections accepted
        self.clients_closed = 0
        self.sessions_started = 0  # media pipelines created (first viewer after idle)

    def start(self) -> None:
        log.info("rtsp launch: %s", self._launch)
        self._server = GstRtspServer.RTSPServer.new()
        self._server.set_service(str(self._port))
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(self._launch)
        factory.set_shared(True)
        factory.connect("media-configure", self._on_media_configure)
        self._server.get_mount_points().add_factory(self._path, factory)
        self._server.connect("client-connected", self._on_client_connected)
        self._source_id = self._server.attach(self._context)
        if self._source_id == 0:
            raise RuntimeError(f"could not bind RTSP server to port {self._port}")
        self._thread = threading.Thread(target=self._loop.run, name="rtsp", daemon=True)
        self._thread.start()
        log.info("rtsp server listening on rtsp://<jetson-ip>:%d%s", self._port, self._path)

    def _on_client_connected(self, _server, client) -> None:
        self.clients_connected += 1
        ip = client.get_connection().get_ip()
        log.info("rtsp client connected: %s (#%d)", ip, self.clients_connected)
        client.connect("closed", self._on_client_closed, ip)

    def _on_client_closed(self, _client, ip) -> None:
        self.clients_closed += 1
        log.info("rtsp client closed: %s", ip)

    def _on_media_configure(self, _factory, media) -> None:
        appsrc = media.get_element().get_by_name(APPSRC_NAME)
        media.connect("unprepared", self._on_media_unprepared)
        with self._lock:
            self._appsrc = appsrc
        self.sessions_started += 1
        log.info("rtsp media session #%d started", self.sessions_started)

    def _on_media_unprepared(self, _media) -> None:
        with self._lock:
            self._appsrc = None
        log.info("rtsp media session ended (no viewers)")

    @property
    def has_viewer(self) -> bool:
        with self._lock:
            return self._appsrc is not None

    def push(self, image: np.ndarray) -> Optional[bool]:
        """Send one BGR frame. Returns None if no viewer is connected,
        True if the frame was accepted, False if it was refused."""
        with self._lock:
            appsrc = self._appsrc
        if appsrc is None:
            return None
        buf = Gst.Buffer.new_wrapped(image.tobytes())
        return appsrc.emit("push-buffer", buf) == Gst.FlowReturn.OK

    def stop(self) -> None:
        """Disconnect viewers and stop serving."""
        with self._lock:
            self._appsrc = None
        if self._server is not None:
            self._server.client_filter(lambda _s, _c: GstRtspServer.RTSPFilterResult.REMOVE)
        if self._source_id:
            source = self._context.find_source_by_id(self._source_id)
            if source is not None:
                source.destroy()
            self._source_id = 0
        self._loop.quit()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._server = None
