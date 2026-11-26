"""
Blade channel: a second detector for small drones whose body echo is too weak for detect.py.

Why: on Nov 25 a sub-249 g drone hovering at ~2.5 m outdoors never passed the 15 dB
single-frame test, but its spinning blades raised the fast-Doppler energy at its range by
~3 dB over the empty scene, very steadily, once averaged over 0.5 s.

Per frame:
1. blade-band energy per range bin: |v| >= v_lo (full band) and |v| >= v_hi (high band),
   averaged over the last window_frames frames, in dB relative to the median range bin
2. excess = that profile minus the empty-scene profile (learned at start, or loaded from an
   empty recording, then slowly updated while nothing is there)
3. a bin is a blade hit when full-band excess >= thr_db, high-band excess >= thr_hi_db, and
   the two are within max_tilt_db (blades alias over the whole band; a walker's limbs don't)
4. drone when at least persist_min of the last persist_frames frames hit within +/- persist_gate_m

combine() merges this with the main tracker/classifier result (used by node.py and blade_eval.py).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from config import BLADE, FFT_SHIFT, RANGE_AXIS, VEL_AXIS


@dataclass
class BladeResult:
    state: str                       # warming | learning | running
    drone: bool = False
    range_m: float = float("nan")
    excess_db: float = float("nan")  # full band, at the strongest bin
    excess_hi_db: float = float("nan")
    hits: int = 0                    # blade hits in the persistence window
    confidence: float = 0.0
    bin: int = -1


def _db(x):
    return 10 * np.log10(x + 1e-30)


class BladeDetector:
    def __init__(self, cfg=BLADE, background=None):
        self.cfg = cfg
        self.valid = (RANGE_AXIS >= cfg["min_range_m"]) & (RANGE_AXIS <= cfg["max_range_m"])
        self.vi = np.flatnonzero(self.valid)
        av = np.abs(VEL_AXIS)
        self.lo = av >= cfg["v_lo_m_s"]
        self.hi = av >= cfg["v_hi_m_s"]
        self.buf = deque(maxlen=cfg["window_frames"])
        self.hits = deque(maxlen=cfg["persist_frames"])
        self.learn = []
        self.base = None                              # (full, hi) dB profiles of the empty scene
        self.was_drone = False
        self.mask_r, self.mask_left = None, 0         # range of the main tracker's target, frames left
        if background is not None:
            self.base = (np.array(background[0], float), np.array(background[1], float))

    # ---- profiles ----
    def _profiles(self, RD):
        P = RD.real ** 2 + RD.imag ** 2
        self.buf.append((P[self.lo].sum(axis=0), P[self.hi].sum(axis=0)))
        if len(self.buf) < self.buf.maxlen:
            return None
        full = _db(np.mean([b[0] for b in self.buf], axis=0))
        hi = _db(np.mean([b[1] for b in self.buf], axis=0))
        return full - np.median(full[self.valid]), hi - np.median(hi[self.valid])

    @property
    def learned(self):
        return self.base is not None

    def save(self, path):
        np.save(path, np.stack(self.base))

    # ---- per frame ----
    def update(self, RD, det=None):
        """det: the main tracker's Detection this frame (or None). Its target, and the place it
        just left, are masked for mask_hold_frames: a walker's limbs also raise blade-band energy."""
        main_detected = bool(det is not None and det.detected)
        if main_detected:
            self.mask_r, self.mask_left = det.range_m, self.cfg["mask_hold_frames"]
        elif self.mask_left > 0:
            self.mask_left -= 1
        prof = self._profiles(RD)
        if prof is None:
            return BladeResult("warming")
        full, hi = prof
        if self.base is None:                         # learn the empty scene
            self.learn.append(prof)
            if len(self.learn) >= self.cfg["learn_frames"]:
                arr = np.array(self.learn)            # [frames, 2, NB]
                self.base = (np.median(arr[:, 0], axis=0), np.median(arr[:, 1], axis=0))
                self.learn = []
            return BladeResult("learning")

        ex, ex_hi = full - self.base[0], hi - self.base[1]
        ok = (self.valid & (ex >= self.cfg["thr_db"]) & (ex_hi >= self.cfg["thr_hi_db"])
              & (ex - ex_hi <= self.cfg["max_tilt_db"]))
        if self.mask_left > 0:                        # not where the main tracker's target is / just was,
            g = self.cfg["mask_gate_m"]               # nor at its 2x / 3x echoes (radar-person bounces)
            rs_m = [self.mask_r] + ([2 * self.mask_r, 3 * self.mask_r]
                                    if self.mask_r < self.cfg["mask_multipath_below_m"] else [])
            for rm in rs_m:
                ok &= np.abs(RANGE_AXIS - rm) > g
            self.hits = deque((h if h is None or all(abs(h - rm) > g for rm in rs_m) else None
                               for h in self.hits), maxlen=self.cfg["persist_frames"])
        hit = bool(ok.any())                          # strongest bin that passes all tests,
        pool = np.flatnonzero(ok) if hit else self.vi  # so clutter elsewhere can't hide the drone
        k = int(pool[np.argmax(ex[pool])])
        r = float(RANGE_AXIS[k] + _parabolic(ex, k) * (RANGE_AXIS[1] - RANGE_AXIS[0]))
        self.hits.append(r if hit else None)

        # slow background update while nothing is in front (never absorb a target)
        if not hit and not main_detected:
            a = self.cfg["bg_alpha"]
            quiet = ex < 0.5 * self.cfg["thr_db"]
            self.base = (np.where(quiet, (1 - a) * self.base[0] + a * full, self.base[0]),
                         np.where(quiet, (1 - a) * self.base[1] + a * hi, self.base[1]))

        rs = [h for h in self.hits if h is not None]
        drone, conf, rr = False, 0.0, r
        if rs:
            med = float(np.median(rs))
            near = [h for h in rs if abs(h - med) <= self.cfg["persist_gate_m"]] or [med]
            conf = len(near) / self.cfg["persist_frames"]
            need = self.cfg["persist_min"] // 2 if self.was_drone else self.cfg["persist_min"]
            drone = len(near) >= need and conf > 0    # hysteresis: harder to start than to keep
            rr = float(np.median(near))
        self.was_drone = drone
        return BladeResult("running", drone, rr if drone else r, float(ex[k]), float(ex_hi[k]),
                           len(rs), round(conf, 2), k)


def _parabolic(y, k):
    if 0 < k < len(y) - 1:
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        if den != 0:
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0


def background_from_recording(path, frames=120):
    """Empty-scene profile from an empty recording (.npz from record.py) or a saved .npy."""
    if path.endswith(".npy"):
        b = np.load(path)
        return b[0], b[1]
    from dsp import NumpyRangeFFT, preprocess, range_doppler
    cfg = dict(BLADE, learn_frames=max(1, frames - BLADE["window_frames"] + 1))
    bd, fft = BladeDetector(cfg), NumpyRangeFFT(FFT_SHIFT)
    for f in np.load(path)["frames"][:frames]:
        _, RD = range_doppler(fft(preprocess(f)))
        bd.update(RD)
    if bd.base is None:
        raise SystemExit(f"{path}: too few frames to learn the empty scene")
    return bd.base


def combine(det, label, conf, feats, b, cfg=BLADE):
    """Merge the main result with the blade channel -> (range_m, speed_m_s, snr_db, label, conf, feats, source).
    source: 'main' (detect.py) or 'blade' (this channel)."""
    feats = dict(feats, blade_state=b.state)
    if b.state == "running":
        feats.update(blade_excess_db=round(b.excess_db, 2), blade_excess_hi_db=round(b.excess_hi_db, 2),
                     blade_hits=b.hits)
    rng, spd, snr = det.range_m, det.speed_m_s, det.snr_db
    if not b.drone:
        return rng, spd, snr, label, conf, feats, "main"
    same = det.detected and abs(det.range_m - b.range_m) <= 1.0
    if det.detected and not same and label in ("person", "drone"):
        return rng, spd, snr, label, conf, feats, "main"           # main target elsewhere: keep it
    if same and label == "person" and b.excess_hi_db < cfg["person_override_hi_db"]:
        return rng, spd, snr, label, conf, feats, "main"           # a walker, not enough blade evidence
    if same:                                                       # main sees it moving: keep its range/speed
        return rng, spd, snr, "drone", max(conf, b.confidence), feats, "main+blade"
    return b.range_m, float("nan"), b.excess_db, "drone", b.confidence, feats, "blade"
