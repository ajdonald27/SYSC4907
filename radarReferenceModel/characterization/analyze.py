"""
RATS radar characterization - analyze recordings made with record.py.

    python analyze.py show recordings/<file>.npz [--background <empty_room>.npz] [--gate 1 4]
        -> <file>.png with 6 panels + printed summary (peak range, SNR, speed, Doppler
           spread, blade-flash rate) + <file>.summary.json

    python analyze.py calib calib.csv
        calib.csv columns:  file,true_m[,background]   (one reflector recording per distance;
        background = empty-room recording with the same preset, strongly recommended)
        -> range error table, linear fit (scale/offset), RMS error, check against FR2 (+/-1 m)

The processing here is deliberately the same chain the FPGA will implement
(DC removal -> window -> range FFT -> clutter removal -> Doppler FFT), so these
recordings and outputs become test vectors for the hardware later.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")   # file output only; don't override the GUI backend when imported (live_view)
import matplotlib.pyplot as plt

C = 299_792_458.0


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z["frames"].astype(np.float64), z["t"], meta


def axes(meta):
    p = meta["params"]
    B = p["end_frequency_Hz"] - p["start_frequency_Hz"]
    fc = 0.5 * (p["end_frequency_Hz"] + p["start_frequency_Hz"])
    N, M, Tc = p["num_samples"], p["num_chirps"], p["chirp_repetition_time_s"]
    lam = C / fc
    rng = np.arange(N // 2) * C / (2 * B)
    vel = np.fft.fftshift(np.fft.fftfreq(M, Tc)) * lam / 2
    return rng, vel, lam, Tc


def range_fft(frames):
    """frames [F, M, N] real ADC codes -> complex range spectra [F, M, N/2]."""
    x = frames - frames.mean(axis=-1, keepdims=True)          # remove DC per chirp
    w = np.hanning(x.shape[-1])
    return np.fft.rfft(x * w, axis=-1)[..., : x.shape[-1] // 2] / w.sum()


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-12)


def peak_in_gate(profile_db, rng, lo, hi):
    """Peak bin inside [lo, hi] m with parabolic (sub-bin) interpolation."""
    idx = np.where((rng >= lo) & (rng <= hi))[0]
    k = idx[np.argmax(profile_db[idx])]
    if 0 < k < len(profile_db) - 1:
        a, b, c = profile_db[k - 1], profile_db[k], profile_db[k + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        frac = 0.0
    dR = rng[1] - rng[0]
    return k, (k + frac) * dR


def snr_db(profile_db, k, valid):
    mask = valid.copy()
    mask[max(0, k - 3): k + 4] = False
    return profile_db[k] - np.median(profile_db[mask])


def recordings(folder="recordings"):
    """All recordings, newest first."""
    return sorted(glob.glob(os.path.join(folder, "*.npz")), key=os.path.getmtime, reverse=True)


def resolve_file(name):
    """'last' -> newest recording that is not an empty-room file."""
    if name != "last":
        return name
    for f in recordings():
        if "empty" not in json.loads(str(np.load(f)["meta"])).get("label", "").lower():
            return f
    sys.exit("no recordings found in .\\recordings")


def resolve_background(name, meta):
    """'auto' -> newest empty-room recording made with the same preset settings."""
    if name != "auto":
        return name
    for f in recordings():
        m = json.loads(str(np.load(f)["meta"]))
        if "empty" in m.get("label", "").lower() and m["params"] == meta["params"]:
            return f
    print("  (no matching empty-room recording found: analyzing without background)")
    return None


def show(args):
    args.file = resolve_file(args.file)
    frames, t, meta = load(args.file)
    if args.background:
        args.background = resolve_background(args.background, meta)
        if args.background:
            print("Background:", args.background)
    rng, vel, lam, Tc = axes(meta)
    F, M, N = frames.shape
    X = range_fft(frames)                                    # [F, M, N/2]
    if args.background and args.background.lower() != "none":
        bf, _, bmeta = load(args.background)
        if bmeta["params"] != meta["params"]:
            sys.exit("background was recorded with a different preset")
        X = X - range_fft(bf).mean(axis=(0, 1), keepdims=True)   # complex static-scene subtraction
    Xmti = X - X.mean(axis=1, keepdims=True)                  # remove static clutter per frame

    lo = args.gate[0] if args.gate else 0.6
    hi = args.gate[1] if args.gate else rng[-1] * 0.95
    valid = rng >= 0.6

    prof = db(np.sqrt((np.abs(X) ** 2).mean(axis=(0, 1))))         # all returns
    prof_mov = db(np.sqrt((np.abs(Xmti) ** 2).mean(axis=(0, 1))))  # moving returns only
    use_moving = args.moving
    p_used = prof_mov if use_moving else prof
    k, r_est = peak_in_gate(p_used, rng, lo, hi)
    snr = snr_db(p_used, k, valid)

    # Range-Doppler (average power over frames)
    wd = np.hanning(M)[None, :, None]
    RD = np.fft.fftshift(np.fft.fft(Xmti * wd, axis=1), axes=1)      # [F, M, N/2]
    RDp = (np.abs(RD) ** 2).mean(axis=0)
    # Doppler spectrum at the target bin (+/-1 bin), per frame -> micro-Doppler spectrogram
    kk = slice(max(0, k - 1), k + 2)
    spec = (np.abs(RD[:, :, kk]) ** 2).sum(axis=2)                     # [F, M]
    spec_db = 10 * np.log10(spec + 1e-20)
    mean_spec_db = 10 * np.log10(spec.mean(axis=0) + 1e-20)
    v_peak = vel[np.argmax(mean_spec_db)]
    above = vel[mean_spec_db > mean_spec_db.max() - 20]
    spread = above.max() - above.min()

    # Blade-flash rate: envelope of the high-Doppler part of the slow-time signal
    s = Xmti[:, :, kk].sum(axis=2)                                    # [F, M] complex slow-time
    S = np.fft.fft(s, axis=1)
    fd = np.fft.fftfreq(M, Tc)
    S[:, np.abs(fd * lam / 2) < args.flash_vmin] = 0                  # keep only |v| > vmin
    env = np.abs(np.fft.ifft(S, axis=1)) ** 2
    env = env - env.mean(axis=1, keepdims=True)
    E = (np.abs(np.fft.rfft(env * np.hanning(M), axis=1)) ** 2).mean(axis=0)
    fe = np.fft.rfftfreq(M, Tc)
    band = (fe >= 20) & (fe <= 1000)
    kf = np.argmax(E[band])
    flash_hz = fe[band][kf]
    flash_prom = 10 * np.log10(E[band][kf] / (np.median(E[band]) + 1e-30))

    summary = dict(
        file=args.file, label=meta.get("label"), true_distance_m=meta.get("distance_m"),
        preset=meta["preset"], frames=F, achieved_fps=meta.get("achieved_fps"),
        sdk_errors=len(meta.get("sdk_errors", [])), peak_range_m=round(r_est, 3),
        peak_from="moving returns" if use_moving else "all returns",
        snr_db=round(float(snr), 1), dominant_speed_m_s=round(float(v_peak), 2),
        doppler_spread_20dB_m_s=round(float(spread), 2),
        flash_rate_hz=round(float(flash_hz), 1), flash_prominence_db=round(float(flash_prom), 1),
        flash_resolution_hz=round(1 / (M * Tc), 1),
        flash_detected=bool(flash_prom >= args.flash_threshold),
    )
    for kx, vx in summary.items():
        print(f"{kx:26s} {vx}")
    if flash_prom < args.flash_threshold:
        print("  (flash-rate peak below threshold: treat as 'no blade flash'. Tune --flash-threshold from real recordings)")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].plot(rng, prof, label="all returns")
    ax[0, 0].plot(rng, prof_mov, label="moving only", alpha=0.8)
    ax[0, 0].axvline(r_est, color="r", ls="--", label=f"peak {r_est:.2f} m, SNR {snr:.0f} dB")
    ax[0, 0].set(xlabel="range (m)", ylabel="dB", title="Range profile")
    ax[0, 0].legend(fontsize=8)

    rt = db(np.sqrt((np.abs(Xmti if use_moving else X) ** 2).mean(axis=1)))
    ax[0, 1].imshow(rt.T, aspect="auto", origin="lower", extent=[0, t[-1] if len(t) else F, 0, rng[-1]],
                    vmin=rt.max() - 50, vmax=rt.max(), cmap="viridis")
    ax[0, 1].set(xlabel="time (s)", ylabel="range (m)", title="Range vs time")

    rd_db = 10 * np.log10(RDp + 1e-20)
    ax[0, 2].imshow(rd_db, aspect="auto", origin="lower", extent=[0, rng[-1], vel[0], vel[-1]],
                    vmin=rd_db.max() - 50, vmax=rd_db.max(), cmap="viridis")
    ax[0, 2].set(xlabel="range (m)", ylabel="radial speed (m/s)", title="Range-Doppler (moving only)")

    ax[1, 0].imshow(spec_db.T, aspect="auto", origin="lower", extent=[0, t[-1] if len(t) else F, vel[0], vel[-1]],
                    vmin=spec_db.max() - 50, vmax=spec_db.max(), cmap="viridis")
    ax[1, 0].set(xlabel="time (s)", ylabel="radial speed (m/s)", title=f"Micro-Doppler at {rng[k]:.2f} m")

    ax[1, 1].plot(vel, mean_spec_db)
    ax[1, 1].set(xlabel="radial speed (m/s)", ylabel="dB", title=f"Mean Doppler spectrum (20 dB spread {spread:.1f} m/s)")

    ax[1, 2].plot(fe[band], 10 * np.log10(E[band] + 1e-30))
    ax[1, 2].axvline(flash_hz, color="r", ls="--", label=f"{flash_hz:.0f} Hz ({flash_prom:.0f} dB)")
    ax[1, 2].set(xlabel="Hz", ylabel="dB", title="Blade-flash envelope spectrum")
    ax[1, 2].legend(fontsize=8)

    fig.suptitle(f"{meta.get('label')}  |  {meta['preset']}  |  true {meta.get('distance_m')} m  |  {args.file}", fontsize=10)
    fig.tight_layout()
    out = args.file.rsplit(".", 1)[0]
    fig.savefig(out + ".png", dpi=110)
    with open(out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved", out + ".png")


def calib(args):
    rows = list(csv.DictReader(open(args.csv)))
    true, meas, snrs = [], [], []
    bg_cache = {}
    for r in rows:
        frames, _, meta = load(r["file"])
        rng, *_ = axes(meta)
        X = range_fft(frames)
        bg = (r.get("background") or args.background or "").strip()
        if bg:   # subtract the empty-room scene so walls/leakage cannot win the peak search
            if bg not in bg_cache:
                bf, _, bmeta = load(bg)
                if bmeta["params"] != meta["params"]:
                    sys.exit(f"{bg} was recorded with a different preset than {r['file']}")
                bg_cache[bg] = range_fft(bf).mean(axis=(0, 1), keepdims=True)
            X = X - bg_cache[bg]
        prof = db(np.sqrt((np.abs(X) ** 2).mean(axis=(0, 1))))
        tm = float(r["true_m"])
        k, r_est = peak_in_gate(prof, rng, max(0.6, tm - args.window), tm + args.window)
        true.append(tm); meas.append(r_est); snrs.append(snr_db(prof, k, rng >= 0.6))
    true, meas = np.array(true), np.array(meas)
    a, b = np.polyfit(true, meas, 1)
    err = meas - true
    print(f"{'true (m)':>9} {'meas (m)':>9} {'error (m)':>10} {'SNR (dB)':>9}")
    for tm, me, e, s in zip(true, meas, err, snrs):
        print(f"{tm:9.2f} {me:9.3f} {e:+10.3f} {s:9.1f}")
    print(f"\nfit: measured = {a:.4f} * true {b:+.3f} m")
    print(f"raw RMS error {np.sqrt(np.mean(err ** 2)):.3f} m, max |error| {np.abs(err).max():.3f} m")
    corr = (meas - b) / a - true
    print(f"after removing offset/scale: RMS {np.sqrt(np.mean(corr ** 2)):.3f} m")
    print("FR2 (+/-1 m):", "PASS" if np.abs(err).max() <= 1.0 else "FAIL")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(true, meas, "o", label="measured")
    ax[0].plot(true, true, "k--", label="ideal")
    ax[0].set(xlabel="tape distance (m)", ylabel="radar range (m)", title="Range calibration")
    ax[0].legend()
    ax[1].plot(true, snrs, "o-")
    if len(true) > 2:
        ref = snrs[0] - 40 * np.log10(true / true[0])
        ax[1].plot(true, ref, "k--", label="1/R^4 slope from first point")
        ax[1].legend()
    ax[1].set(xlabel="tape distance (m)", ylabel="SNR (dB)", title="SNR vs range")
    fig.tight_layout()
    out = args.csv.rsplit(".", 1)[0] + "_calib.png"
    fig.savefig(out, dpi=110)
    print("Saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("file", nargs="?", default="last", help="recording to analyze (default: the newest one)")
    s.add_argument("--background", default="auto", help="empty-room recording; default 'auto' = newest matching empty_* file; 'none' to skip")
    s.add_argument("--gate", nargs=2, type=float, metavar=("LO_M", "HI_M"), help="search for the target only in this range window")
    s.add_argument("--moving", action="store_true", help="pick the target from moving returns (fan, person, drone)")
    s.add_argument("--flash-threshold", type=float, default=12.0, help="dB above median to call a blade flash detected")
    s.add_argument("--flash-vmin", type=float, default=1.0, help="ignore |v| below this (m/s) for blade-flash analysis")
    c = sub.add_parser("calib")
    c.add_argument("csv")
    c.add_argument("--window", type=float, default=0.75, help="search +/- this many metres around the tape distance")
    c.add_argument("--background", help="empty-room recording (same preset) subtracted from every file; or add a 'background' column to the CSV")
    a = ap.parse_args()
    show(a) if a.cmd == "show" else calib(a)
