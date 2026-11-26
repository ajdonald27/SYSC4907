"""Quick look at a recording around one range: python3 diag.py <file.npz> [more files] [--at 2.0]"""
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
fast = np.abs(VEL_AXIS) > 1.0
for path in args:
    frames = np.load(path)["frames"][::2]
    mov, blade, static, peak = [], [], [], []
    for f in frames:
        R = fft(preprocess(f))
        _, RD = range_doppler(R)
        P = RD.real ** 2 + RD.imag ** 2
        prof = 10 * np.log10(P.sum(axis=0) + 1e-30)
        noise = np.median(prof[valid])
        mov.append(prof[near].max() - noise)
        bp = 10 * np.log10(P[fast].sum(axis=0) + 1e-30)
        blade.append(bp[near].max() - np.median(bp[valid]))
        static.append(20 * np.log10(np.abs(R.mean(axis=0)) + 1e-30))
        peak.append(RANGE_AXIS[np.flatnonzero(valid)[np.argmax(prof[valid])]])
    mov, blade, st = np.array(mov), np.array(blade), np.mean(static, axis=0)
    print(f"\n{path.split('/')[-1]}  ({len(frames)} frames looked at, near = {at - 0.75:.2f}-{at + 0.75:.2f} m)")
    print(f"  moving energy near: median {np.median(mov):5.1f} dB, 90th pct {np.percentile(mov, 90):5.1f} dB, "
          f"frames >= 15 dB {np.mean(mov >= 15) * 100:3.0f} %, >= 10 dB {np.mean(mov >= 10) * 100:3.0f} %")
    print(f"  fast Doppler (|v| > 1 m/s) near: median {np.median(blade):5.1f} dB, 90th pct {np.percentile(blade, 90):5.1f} dB")
    print(f"  strongest moving range (median over frames): {np.median(peak):.2f} m")
    print("  static echo (dB) by range: " + "  ".join(f"{r:.2f}:{s:5.1f}" for r, s in zip(RANGE_AXIS, st) if 0.7 <= r <= at + 1.5))
