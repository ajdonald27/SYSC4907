"""Blade energy integrated over 0.5 s, and ADC headroom: python3 blade.py <file.npz> [more] [--at 2.0]"""
import sys
import numpy as np
from config import RANGE_AXIS, VEL_AXIS, FFT_SHIFT
from dsp import preprocess, NumpyRangeFFT, range_doppler

args = sys.argv[1:]
at = 2.0
if "--at" in args:
    i = args.index("--at"); at = float(args[i + 1]); del args[i:i + 2]
fft = NumpyRangeFFT(FFT_SHIFT)
valid = (RANGE_AXIS >= 0.7) & (RANGE_AXIS <= 15.5)
near = valid & (np.abs(RANGE_AXIS - at) <= 0.75)
fast = np.abs(VEL_AXIS) > float(__import__("os").environ.get("VMIN", "1.5"))
W = 10                                  # frames integrated (0.5 s)
for path in args:
    frames = np.load(path)["frames"]
    peak = float(np.abs(frames).max())
    clip = float(np.mean(np.abs(frames) > 0.98) * 100)
    bp = []
    for f in frames:
        _, RD = range_doppler(fft(preprocess(f)))
        P = RD.real ** 2 + RD.imag ** 2
        bp.append(P[fast].sum(axis=0))
    bp = np.array(bp)
    n = len(bp) // W
    win = bp[:n * W].reshape(n, W, -1).mean(axis=1)              # [windows, range]
    db = 10 * np.log10(win + 1e-30)
    snr = db - np.median(db[:, valid], axis=1, keepdims=True)    # per window, vs median range bin
    near_snr = snr[:, near].max(axis=1)
    kmax = np.flatnonzero(valid)[np.argmax(snr[:, valid], axis=1)]
    on = np.mean(np.abs(RANGE_AXIS[kmax] - at) <= 0.75) * 100
    print(f"\n{path.split('/')[-1]}")
    print(f"  ADC peak {peak:.2f} of 1.00, samples near full scale {clip:.3f} %")
    print(f"  blade energy near {at:g} m, 0.5 s windows: median {np.median(near_snr):4.1f} dB, "
          f"min {near_snr.min():4.1f} dB, windows >= 6 dB {np.mean(near_snr >= 6) * 100:3.0f} %")
    print(f"  strongest blade range is near {at:g} m in {on:3.0f} % of windows")
    prof = np.median(snr, axis=0)
    print("  median profile (dB): " + "  ".join(f"{r:.2f}:{s:4.1f}" for r, s in zip(RANGE_AXIS, prof) if 0.7 <= r <= at + 2.0))
