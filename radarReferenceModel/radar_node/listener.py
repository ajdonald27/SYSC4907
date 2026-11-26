"""
Test listener for radar messages (run it on the Mac or the Jetson).
Checks what FR12 asks the fusion stage to check: bad checksum or format (malformed),
gaps in sequence numbers (lost), and timestamps older than the fusion window (stale).

    python listener.py                  # listen on UDP 5005
    python listener.py --port 5005 --window 1.0 --all
"""
import argparse
import json
import socket
import time

from messages import Sender, verify


class Checker:
    def __init__(self, window_s=1.0):
        self.window = window_s
        self.last_seq = {}
        self.stats = dict(ok=0, malformed=0, lost=0, stale=0)

    def check(self, raw, now=None):
        """-> (status, msg). status: ok | malformed | stale; lost messages are counted in stats."""
        now = time.time() if now is None else now
        try:
            msg = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self.stats["malformed"] += 1
            return "malformed", None
        if not all(k in msg for k in Sender.REQUIRED) or not verify(msg):
            self.stats["malformed"] += 1
            return "malformed", msg
        src, seq = msg["source"], msg["seq"]
        prev = self.last_seq.get(src)
        if prev is not None and seq > prev + 1:
            self.stats["lost"] += seq - prev - 1
        self.last_seq[src] = max(seq, prev if prev is not None else seq)
        if now - msg["timestamp"] > self.window:
            self.stats["stale"] += 1
            return "stale", msg
        self.stats["ok"] += 1
        return "ok", msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--window", type=float, default=1.0, help="stale if older than this (s)")
    ap.add_argument("--all", action="store_true", help="print every detection, not just changes")
    a = ap.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", a.port))
    ck, last = Checker(a.window), None
    print(f"listening on UDP {a.port} (stale window {a.window} s). Ctrl+C for totals.")
    try:
        while True:
            raw, addr = sock.recvfrom(65535)
            status, msg = ck.check(raw)
            if status == "malformed":
                print(f"MALFORMED from {addr[0]}")
                continue
            p = msg["payload"]
            tag = "" if status == "ok" else f"  [{status.upper()} {time.time() - msg['timestamp']:.2f} s old]"
            if msg["type"] == "radar_status":
                print(f"STATUS {p.get('state')}  fps={p.get('fps')}  errors={p.get('sdk_errors')}  "
                      f"fft={p.get('fft_backend')}  proc={p.get('proc_ms_mean')} ms{tag}")
            elif msg["type"] == "radar_detection":
                key = (p["detected"], p["class"])
                if a.all or key != last or status != "ok":
                    if p["detected"]:
                        print(f"seq {msg['seq']:6d}  {p['range_m']:5.2f} m  {p['speed_m_s']:+5.2f} m/s  "
                              f"{p['snr_db']:4.1f} dB  {p['class']} ({p['confidence']}){tag}")
                    else:
                        print(f"seq {msg['seq']:6d}  no target{tag}")
                last = key
    except KeyboardInterrupt:
        print("\n" + "  ".join(f"{k}={v}" for k, v in ck.stats.items()))


if __name__ == "__main__":
    main()
