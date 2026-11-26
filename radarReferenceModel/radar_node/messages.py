"""
Radar messages (FR11): timestamped UDP JSON with type, source, sequence number, timestamp,
payload and checksum. The checksum is CRC-32 (hex) of the message without its checksum
field, serialized as compact JSON with sorted keys.

Message types
  radar_detection  one per frame:
      {"detected", "range_m", "speed_m_s", "snr_db", "class", "confidence",
       "track_id", "frame", "latency_ms", "features": {...}}
  radar_status     about once a second (FR9 subsystem state):
      {"state": "initializing|running|disconnected|fault|stopped", "fps", "frames",
       "sdk_errors", "fft_backend", "proc_ms_mean", "proc_ms_max", "detail"}
timestamp = Unix time (s) when the frame was acquired on the PYNQ. Receivers compare it with
their own clock, so the PYNQ and the receiver clocks must be synced (NTP) for stale checks.
"""
import json
import socket
import time
import zlib


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def seal(msg):
    body = {k: v for k, v in msg.items() if k != "checksum"}
    msg["checksum"] = f"{zlib.crc32(_canonical(body).encode()) & 0xFFFFFFFF:08x}"
    return msg


def verify(msg):
    """True if the checksum matches the rest of the message."""
    if not isinstance(msg, dict) or "checksum" not in msg:
        return False
    body = {k: v for k, v in msg.items() if k != "checksum"}
    try:
        return f"{zlib.crc32(_canonical(body).encode()) & 0xFFFFFFFF:08x}" == msg["checksum"]
    except (TypeError, ValueError):
        return False


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python numbers."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x


class Sender:
    REQUIRED = ("type", "source", "seq", "timestamp", "payload", "checksum")

    def __init__(self, host, port, source):
        self.addr = (host, port)
        self.source = source
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if host.endswith(".255"):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send(self, mtype, payload, timestamp=None):
        msg = seal(dict(type=mtype, source=self.source, seq=self.seq,
                        timestamp=round(time.time() if timestamp is None else timestamp, 4),
                        payload=_clean(payload)))
        self.seq += 1
        try:
            self.sock.sendto(_canonical(msg).encode(), self.addr)
        except OSError:
            pass   # receiver down or no route: keep running, the status shows it on the other side
        return msg
