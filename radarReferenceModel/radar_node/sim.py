"""
Synthetic S_system frames for self-tests (same physics as characterization/synth.py),
returned as signed floats like the SDK.
kinds: "walker" (person walking toward the radar), "rotor" (spinning blades seen side-on,
a stand-in for drone props), "empty".
"""
import numpy as np

from config import C, M, N, RADAR, TC


def frames(kind, seconds=2.0, distance=3.0, rps=40.0, blades=2, seed=0, rotor_amp=1.0, walker_amp=1.0):
    """rotor_amp scales the rotor echo (1.0 = the strong default; ~0.03 = a weak sub-249 g drone)."""
    rng = np.random.default_rng(seed)
    r = RADAR
    fs, Tf = r["sample_rate_Hz"], r["frame_repetition_time_s"]
    B = r["end_frequency_Hz"] - r["start_frequency_Hz"]
    lam = C / (0.5 * (r["end_frequency_Hz"] + r["start_frequency_Hz"]))
    S = B / (N / fs)
    n = np.arange(N) / fs
    out = np.zeros((int(seconds / Tf), M, N))

    def add(fr, ranges, amp):
        rr = np.atleast_2d(ranges.T).T if ranges.ndim > 1 else ranges[:, None]
        ph = 2 * np.pi * (2 * rr[:, :, None] * S / C) * n[None, None, :] + (4 * np.pi * rr / lam)[:, :, None]
        fr += (amp * np.cos(ph)).sum(axis=1)

    for f in range(out.shape[0]):
        tm = f * Tf + np.arange(M) * TC
        fr = np.zeros((M, N))
        add(fr, np.full(M, 0.05), 0.05)          # TX leakage
        add(fr, np.full(M, 6.0), 0.01)           # back wall
        if kind == "walker":
            body = distance - 1.2 * tm
            swing = 0.25 * np.sin(2 * np.pi * 0.9 * tm)
            add(fr, body, 0.02 * walker_amp)
            add(fr, body + swing, 0.006 * walker_amp)
            add(fr, body - swing, 0.006 * walker_amp)
        elif kind == "rotor":
            add(fr, np.full(M, distance), 0.004 * rotor_amp)  # body (static)
            rho = np.linspace(0.02, 0.12, 10)
            for b in range(blades):
                th = 2 * np.pi * rps * tm + 2 * np.pi * b / blades
                add(fr, distance + rho[None, :] * np.cos(th)[:, None], 0.0012 * rotor_amp)
        out[f] = fr
    out += rng.normal(0, 2e-4, out.shape)
    return out
