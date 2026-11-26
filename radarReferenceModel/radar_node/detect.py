"""
Detection and single-target tracking on the range-Doppler map.

Per frame: moving-energy range profile -> strongest bin (inside the track gate first) ->
SNR test -> sub-bin range -> Doppler peak (speed) and Doppler spread at that range.
"""
from dataclasses import dataclass, field

import numpy as np

from config import DETECT, RANGE_AXIS, VEL_AXIS


@dataclass
class Detection:
    detected: bool
    range_m: float = float("nan")
    speed_m_s: float = float("nan")
    snr_db: float = float("nan")
    spread_m_s: float = float("nan")
    bin: int = -1
    track_id: int = 0
    extra: dict = field(default_factory=dict)


def _parabolic(y, k):
    """Sub-bin offset of a peak at index k from its two neighbours (y in dB)."""
    if 0 < k < len(y) - 1:
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        if den != 0:
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0


class Tracker:
    def __init__(self, cfg=DETECT):
        self.cfg = cfg
        self.valid = (RANGE_AXIS >= cfg["min_range_m"]) & (RANGE_AXIS <= cfg["max_range_m"])
        self.r = None          # current track range (m)
        self.misses = 0
        self.track_id = 0
        self.next_id = 1

    def update(self, RD):
        P = RD.real ** 2 + RD.imag ** 2                      # power [doppler, range] (no sqrt)
        prof = 10 * np.log10(P.sum(axis=0) + 1e-30)           # moving energy per range bin
        noise = float(np.median(prof[self.valid]))
        thr = self.cfg["snr_db"]

        k = None
        if self.r is not None:                               # prefer the existing track
            gate = self.valid & (np.abs(RANGE_AXIS - self.r) <= self.cfg["gate_m"])
            if gate.any():
                kg = int(np.flatnonzero(gate)[np.argmax(prof[gate])])
                if prof[kg] - noise >= thr:
                    k = kg
        if k is None:
            kv = int(np.flatnonzero(self.valid)[np.argmax(prof[self.valid])])
            if prof[kv] - noise >= thr:
                k = kv

        if k is None:
            self.misses += 1
            if self.misses > self.cfg["max_misses"]:
                self.r, self.track_id = None, 0
            return Detection(False, track_id=self.track_id)

        step = RANGE_AXIS[1] - RANGE_AXIS[0]
        r = float(RANGE_AXIS[k] + _parabolic(prof, k) * step)
        if self.r is None:
            self.track_id, self.next_id = self.next_id, self.next_id + 1
        self.r, self.misses = r, 0

        lo, hi = max(0, k - 1), min(P.shape[1], k + 2)
        spec = P[:, lo:hi].sum(axis=1)
        sdb = 10 * np.log10(spec + 1e-30)
        j = int(np.argmax(sdb))
        vstep = VEL_AXIS[1] - VEL_AXIS[0]
        v = float(VEL_AXIS[j] + _parabolic(sdb, j) * vstep)
        # Doppler spread: span of speeds within 25 dB of the peak, but never counting bins
        # less than 10 dB above this spectrum's median (noise would otherwise widen it)
        thr_d = max(sdb.max() - 25, float(np.median(sdb)) + 10)
        above = VEL_AXIS[sdb > thr_d]
        spread = float(above.max() - above.min()) if above.size else 0.0
        return Detection(True, r, v, float(prof[k] - noise), spread, k, self.track_id)
