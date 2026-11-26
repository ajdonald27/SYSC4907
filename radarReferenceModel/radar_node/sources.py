"""
Frame sources: live radar (Infineon SDK) or replay of record.py recordings.
Both yield (acquisition_time_unix_s, frame float [256, 128]).
"""
import json
import time

import numpy as np

from config import M, N, RADAR


class ReplaySource:
    """Plays S_system recordings (.npz from record.py). realtime=True paces at 20 fps."""
    name = "replay"

    def __init__(self, paths, realtime=True, loop=False):
        self.paths, self.realtime, self.loop = list(paths), realtime, loop
        self.errors = 0
        for p in self.paths:
            z = np.load(p, allow_pickle=False)
            shape = z["frames"].shape[1:]
            if shape != (M, N):
                preset = json.loads(str(z["meta"])).get("preset")
                raise SystemExit(f"{p}: frames are {shape} ({preset}); the node needs S_system ({M}, {N})")

    def __iter__(self):
        period = RADAR["frame_repetition_time_s"]
        while True:
            for p in self.paths:
                frames = np.load(p, allow_pickle=False)["frames"]
                for f in frames:
                    t = time.time()
                    yield t, f.astype(np.float64)
                    if self.realtime:
                        time.sleep(max(0.0, period - (time.time() - t)))
            if not self.loop:
                return


class LiveSource:
    """The radar itself, configured with config.RADAR.

    A background thread reads frames from the SDK as soon as they arrive, so a slow frame in
    the processing loop never overflows the radar's buffer. If processing falls behind, the
    oldest waiting frame is dropped and counted in `skipped` (not an SDK error).
    SDK errors (dropped by the radar itself) are counted in `errors`."""
    name = "live"

    def __init__(self, queue_len=2):
        import queue
        import threading
        from ifxradarsdk.fmcw import DeviceFmcw
        from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
        self.errors, self.skipped, self.exc = 0, 0, None
        self.dev = DeviceFmcw()
        r = RADAR
        cfg = FmcwSimpleSequenceConfig(
            frame_repetition_time_s=r["frame_repetition_time_s"],
            chirp_repetition_time_s=r["chirp_repetition_time_s"],
            num_chirps=r["num_chirps"], tdm_mimo=False,
            chirp=FmcwSequenceChirp(
                start_frequency_Hz=r["start_frequency_Hz"], end_frequency_Hz=r["end_frequency_Hz"],
                sample_rate_Hz=r["sample_rate_Hz"], num_samples=r["num_samples"],
                rx_mask=1, tx_mask=1, tx_power_level=r["tx_power_level"],
                lp_cutoff_Hz=r["lp_cutoff_Hz"], hp_cutoff_Hz=r["hp_cutoff_Hz"], if_gain_dB=r["if_gain_dB"]))
        self.dev.set_acquisition_sequence(self.dev.create_simple_sequence(cfg))
        self._queue_mod = queue
        self.q = queue.Queue(maxsize=queue_len)
        self._stop = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        Full, Empty = self._queue_mod.Full, self._queue_mod.Empty
        while not self._stop:
            try:
                fc = self.dev.get_next_frame()          # blocks in C; other threads keep running
            except Exception as e:
                if "FrameAcquisitionFailed" in type(e).__name__ or "Fifo" in type(e).__name__:
                    self.errors += 1
                    continue
                self.exc = e
                self.q.put(None)
                return
            item = (time.time(), np.asarray(fc[0])[0].astype(np.float64))
            try:
                self.q.put_nowait(item)
            except Full:
                try:
                    self.q.get_nowait()
                    self.skipped += 1
                except Empty:
                    pass
                self.q.put_nowait(item)

    def __iter__(self):
        Empty = self._queue_mod.Empty
        while True:
            try:
                item = self.q.get(timeout=2.0)       # timeout keeps Ctrl+C responsive
            except Empty:
                raise RuntimeError("no frames from the radar for 2 s (unplugged or stalled?)")
            if item is None:
                raise self.exc
            yield item

    def close(self):
        self._stop = True
