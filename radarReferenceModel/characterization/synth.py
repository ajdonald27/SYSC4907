"""
RATS radar characterization - synthetic recordings in the same format as record.py.

Use it to check the analysis chain before (or without) the real sensor, and later as
an FPGA test vector generator with known ground truth.

    python synth.py reflector --distance 2.0
    python synth.py fan --distance 1.5 --rps 20 --blades 3
    python synth.py walker --distance 3.0
    python synth.py empty

Physics: FMCW beat signal per scatterer, x[m, n] = a * cos(2*pi*fIF*n/fs + 4*pi*R(t_m)/lam).
Rotating blades are modelled as lines of point scatterers in a plane that contains the
radar line of sight (a drone seen from the side, or a fan turned side-on). The blade
returns add coherently when a blade is perpendicular to the line of sight -> "flash".
Flash rate = N*rps for an even blade count, 2*N*rps for an odd count.
"""
import argparse
import datetime as dt
import json
import os

import numpy as np

from presets import PRESETS, derived

C = 299_792_458.0


def synth(kind, preset, distance, seconds, rps, blades, seed=0):
    rngen = np.random.default_rng(seed)
    p = PRESETS[preset]
    N, M, fs, Tc, Tf = p["num_samples"], p["num_chirps"], p["sample_rate_Hz"], p["chirp_repetition_time_s"], p["frame_repetition_time_s"]
    B = p["end_frequency_Hz"] - p["start_frequency_Hz"]
    fc = 0.5 * (p["end_frequency_Hz"] + p["start_frequency_Hz"])
    lam, S = C / fc, B / (N / fs)
    F = int(seconds / Tf)
    n = np.arange(N) / fs
    frames = np.zeros((F, M, N))

    def add(frame, m_times, ranges, amp):
        # ranges: [M] or [M, K] (K scatterers)
        r = np.atleast_2d(np.asarray(ranges).T).T if np.ndim(ranges) > 1 else np.asarray(ranges)[:, None]
        a = np.broadcast_to(amp, r.shape)
        phase = 2 * np.pi * (2 * r[:, :, None] * S / C) * n[None, None, :] + (4 * np.pi * r / lam)[:, :, None]
        frame += (a[:, :, None] * np.cos(phase)).sum(axis=1)

    for f in range(F):
        tm = f * Tf + np.arange(M) * Tc
        fr = np.zeros((M, N))
        add(fr, tm, np.full(M, 0.05), 0.3)          # TX->RX leakage near 0 m
        add(fr, tm, np.full(M, 6.0), 0.02)          # back wall
        if kind == "reflector":
            add(fr, tm, np.full(M, distance), 0.05 * (1.0 / distance) ** 2)
        elif kind == "fan":
            add(fr, tm, np.full(M, distance), 0.01)  # fan body / hub
            rho = np.linspace(0.04, 0.15, 12)        # scatterers along each blade (m)
            for b in range(blades):
                th = 2 * np.pi * rps * tm + 2 * np.pi * b / blades
                rr = distance + rho[None, :] * np.cos(th)[:, None]
                add(fr, tm, rr, 0.0015)
        elif kind == "walker":
            v = -1.2                                 # walking toward the radar
            body = distance + v * tm
            add(fr, tm, body, 0.03)
            swing = 0.25 * np.sin(2 * np.pi * 0.9 * tm)
            add(fr, tm, body + swing, 0.008)        # arms
            add(fr, tm, body - swing, 0.008)        # legs (opposite phase)
        frames[f] = fr
    frames += rngen.normal(0, 2e-4, frames.shape)
    adc = np.clip(np.round((0.5 + frames) * 4095), 0, 4095).astype(np.uint16)
    d, _ = derived(p)
    meta = dict(preset=preset, params=p, derived=d, label=f"SYNTH_{kind}", distance_m=distance if kind != "empty" else None,
                notes=f"synthetic rps={rps} blades={blades}", sdk_version="synthetic", achieved_fps=1 / Tf, sdk_errors=[],
                started=dt.datetime.now().isoformat(timespec="seconds"))
    return adc, np.arange(F) * Tf, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["reflector", "fan", "walker", "empty"])
    ap.add_argument("--preset", default=None)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--rps", type=float, default=20.0)
    ap.add_argument("--blades", type=int, default=3)
    ap.add_argument("--outdir", default="recordings")
    a = ap.parse_args()
    preset = a.preset or ("A_range" if a.kind == "reflector" else "B_microdoppler")
    adc, t, meta = synth(a.kind, preset, a.distance, a.seconds, a.rps, a.blades)
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"synth_{preset}_{a.kind}_{a.distance:g}m.npz")
    np.savez_compressed(path, frames=adc, t=t, meta=json.dumps(meta))
    print("Saved", path, adc.shape)
