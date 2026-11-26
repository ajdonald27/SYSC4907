"""
Rule-based classifier v1: drone / person / unknown.

Features at the tracked range bin, per frame:
- Doppler spread (from the detector): spinning blade tips move at 20-30 m/s, so their
  echoes (aliased) cover most of the +/-8.3 m/s span; a person's limbs stay under ~4 m/s.
- envelope kurtosis: blades flash when they line up with the radar, so the fast-Doppler
  envelope is bursty (kurtosis well above 3). A frame counts as a "blade frame" only when
  BOTH hold: kurtosis alone also spikes when a limb crosses the 1 m/s filter edge, and
  spread alone could be widened by noise. (The Nov 18 false alarm came from reading a
  walking person's falling low-frequency spectrum as a 26 Hz flash; this rule doesn't use it.)
- flash rate (reported only): from the envelope's autocorrelation, 60-1000 Hz.

Rules (over the last CLASSIFY['history'] frames with a detection):
- drone  : blade frames in >= drone_min_blade_frames frames
- person : at least 60 % of recent frames look like a walker (no blade frame, spread within
           person_spread_m_s, |speed| <= person_max_speed_m_s) AND the median |speed| is at least
           person_min_speed_m_s (a still or barely-moving echo stays "unknown")
- unknown: anything else (also while fewer than min_frames detections)
The thresholds are first guesses: tune them on the drone/person recordings (NFR2).
"""
from collections import deque

import numpy as np

from config import CLASSIFY, LAMBDA, M, TC

_WIN = np.hanning(M)
_FD = np.fft.fftfreq(M, TC)
_FE = np.fft.rfftfreq(M, TC)


def blade_features(Rm, k, cfg=CLASSIFY):
    """Rm: static-removed range spectra [M, NB]; k: target bin -> (kurtosis, flash_hz or nan)."""
    s = Rm[:, max(0, k - 1):k + 2].sum(axis=1)               # slow-time signal at the target
    S = np.fft.fft(s)
    S[np.abs(_FD * LAMBDA / 2) < cfg["flash_vmin_m_s"]] = 0   # keep only the fast parts
    env = np.abs(np.fft.ifft(S))
    var = float(np.var(env))
    if var <= 0:
        return 0.0, float("nan")
    e = env - env.mean()
    kurt = float(np.mean(e ** 4) / var ** 2)
    ac = np.correlate(e, e, "full")[M - 1:]
    lags = np.arange(M) * TC
    lo, hi = cfg["flash_band_hz"]
    sel = (lags >= 1 / hi) & (lags <= 1 / lo)
    j = int(np.flatnonzero(sel)[np.argmax(ac[sel])])
    return kurt, float(1 / lags[j])


class Classifier:
    def __init__(self, cfg=CLASSIFY):
        self.cfg = cfg
        self.hist = deque(maxlen=cfg["history"])
        self.track_id = 0

    def update(self, det, Rm):
        """-> (label, confidence 0..1, features dict)."""
        cfg = self.cfg
        if det.track_id != self.track_id:          # new track: forget the old one
            self.hist.clear()
            self.track_id = det.track_id
        if not det.detected:
            self.hist.append(None)
            return "none", 0.0, {}
        kurt, fhz = blade_features(Rm, det.bin, cfg)
        flash = det.spread_m_s >= cfg["blade_spread_m_s"] and kurt >= cfg["blade_kurtosis"]
        self.hist.append(dict(flash=flash, spread=det.spread_m_s, speed=det.speed_m_s))
        feats = dict(flash=bool(flash), flash_hz=round(fhz, 1) if flash and not np.isnan(fhz) else None,
                     kurtosis=round(kurt, 2), doppler_spread_m_s=round(det.spread_m_s, 2))

        recent = [h for h in self.hist if h]
        if len(recent) < cfg["min_frames"]:
            return "unknown", 0.0, feats
        nflash = sum(h["flash"] for h in recent)
        if nflash >= cfg["drone_min_blade_frames"]:
            return "drone", round(nflash / len(recent), 2), feats
        lo, hi = cfg["person_spread_m_s"]
        person_like = [(not h["flash"]) and lo <= h["spread"] <= hi and abs(h["speed"]) <= cfg["person_max_speed_m_s"]
                       for h in recent]
        frac = sum(person_like) / len(recent)
        moving = np.median([abs(h["speed"]) for h in recent]) >= cfg["person_min_speed_m_s"]
        if frac >= 0.6 and moving:
            return "person", round(frac, 2), feats
        return "unknown", round(1 - frac, 2), feats
