"""
Score the blade channel (and the whole node) on recordings, no radar needed:

    python3 blade_eval.py --bg <empty.npz> <drone and walk recordings ...> [--at 2.5]
    python3 blade_eval.py --bg <empty.npz> files ... --set thr_db=1.0 --set window_frames=20

Per file: how often the node says drone / person / none, the blade range, and the blade-band
excess (full band and |v| >= 5 m/s) so the thresholds in config.BLADE can be tuned.
--set changes a BLADE setting for this run only (copy good values into config.py).
"""
import argparse
import os

import numpy as np

import config
from blade_det import BladeDetector, background_from_recording, combine
from classify import Classifier
from detect import Tracker
from dsp import NumpyRangeFFT, preprocess, range_doppler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bg", help="empty recording (.npz) or saved profile (.npy); default: learn from each file's start")
    ap.add_argument("--at", type=float, help="true target range (m), to report range error")
    ap.add_argument("--set", action="append", default=[], help="BLADE setting override, e.g. thr_db=1.0")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=")
        config.BLADE[k] = type(config.BLADE[k])(float(v)) if not isinstance(config.BLADE[k], bool) else v == "1"
    if a.set:
        print("settings:", ", ".join(a.set))

    bg = background_from_recording(a.bg) if a.bg else None
    fft = NumpyRangeFFT(config.FFT_SHIFT)
    print(f"{'file':44s} {'drone':>6s} {'person':>6s} {'none/unk':>8s} {'range':>6s} "
          f"{'excess':>7s} {'hi':>6s} {'p90 ex':>6s}")
    for path in a.files:
        tr, clf, bd = Tracker(), Classifier(), BladeDetector(config.BLADE, background=bg)
        labels, rngs, ex, exh = [], [], [], []
        for f in np.load(path)["frames"]:
            Rm, RD = range_doppler(fft(preprocess(f)))
            det = tr.update(RD)
            lab, conf, feats = clf.update(det, Rm)
            b = bd.update(RD, det)
            rng, _, _, lab, conf, feats, src = combine(det, lab, conf, feats, b, config.BLADE)
            if b.state != "running":
                continue
            labels.append(lab)
            ex.append(b.excess_db)
            exh.append(b.excess_hi_db)
            if lab == "drone":
                rngs.append(rng)
        n = max(len(labels), 1)
        pct = lambda *ks: 100 * sum(labels.count(k) for k in ks) / n
        r = f"{np.median(rngs):5.2f}" if rngs else "   - "
        name = os.path.basename(path).replace("20261125_S_system_", "")[:44]
        print(f"{name:44s} {pct('drone'):5.0f}% {pct('person'):5.0f}% {pct('none', 'unknown'):7.0f}% {r:>6s} "
              f"{np.median(ex):+6.1f} {np.median(exh):+6.1f} {np.percentile(ex, 90):+6.1f}"
              + (f"   range error {abs(np.median(rngs) - a.at):.2f} m" if a.at and rngs else ""))
    print("\nexcess / hi = median blade-band excess over the empty scene at the strongest bin (dB);"
          f" thresholds now: thr_db {config.BLADE['thr_db']}, thr_hi_db {config.BLADE['thr_hi_db']},"
          f" max_tilt_db {config.BLADE['max_tilt_db']}")


if __name__ == "__main__":
    main()
