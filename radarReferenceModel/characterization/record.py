"""
RATS radar characterization - record raw frames from the DEMO BGT60UTR11AIP.

Runs anywhere the Infineon Radar SDK Python wrapper (ifxradarsdk) installs:
your Mac, Windows (Parallels), or the PYNQ-Z2. The same file is used on all three
so recordings are directly comparable.

Examples
    python record.py --preset A_range --label reflector --distance 2.0 --seconds 5
    python record.py --preset B_microdoppler --label fan_speed2 --distance 1.5 --seconds 10
    python record.py --preset B_microdoppler --label empty_room --seconds 10
    python record.py --preset B_microdoppler --benchmark --seconds 30     # PYNQ throughput test

Output: recordings/<date>_<preset>_<label>_<distance>m_<time>.npz containing
    frames  float32 [n_frames, n_chirps, n_samples]  (SDK-normalized samples, RX1)
    t       float64 [n_frames]  host time of each frame (s since start)
    meta    JSON string: preset parameters, label, distance, notes, SDK/sensor info,
            achieved frame rate, number of SDK errors (e.g. FIFO overflow = dropped data)
"""
import argparse
import datetime as dt
import json
import os
import platform
import sys
import time

import numpy as np

from presets import PRESETS, derived, to_sdk_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="B_microdoppler", choices=PRESETS.keys())
    ap.add_argument("--label", default="test", help="what is in front of the radar, e.g. reflector, fan_speed2, walk_toward")
    ap.add_argument("--distance", type=float, default=None, help="tape-measured distance to target in metres")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--notes", default="", help="free text: height, angle, room, anything odd")
    ap.add_argument("--outdir", default="recordings")
    ap.add_argument("--benchmark", action="store_true", help="do not keep frames, just measure frame rate and errors")
    ap.add_argument("--show", action="store_true", help="analyze right after recording (auto background, opens the PNG)")
    args = ap.parse_args()

    from ifxradarsdk import get_version_full
    from ifxradarsdk.fmcw import DeviceFmcw

    p = PRESETS[args.preset]
    d, problems = derived(p)
    for msg in problems:
        print("WARNING:", msg)

    meta = dict(
        preset=args.preset, params=p, derived=d, label=args.label, distance_m=args.distance,
        notes=args.notes, host=platform.platform(), machine=platform.machine(),
        python=sys.version.split()[0], sdk_version=get_version_full(),
        started=dt.datetime.now().isoformat(timespec="seconds"),
    )

    frames, stamps, errors = [], [], []
    with DeviceFmcw() as device:
        for attr in ("get_sensor_type", "get_board_uuid", "get_firmware_information"):
            try:
                meta[attr] = str(getattr(device, attr)())
            except Exception:
                pass
        print("Sensor:", meta.get("get_sensor_type", "?"), "| SDK:", meta["sdk_version"])
        sequence = device.create_simple_sequence(to_sdk_config(p))
        device.set_acquisition_sequence(sequence)

        print(f"Recording {args.seconds:.0f} s of '{args.label}' with {args.preset} ...")
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < args.seconds:
            try:
                frame_contents = device.get_next_frame()
            except Exception as e:  # FIFO overflow etc. -> data was lost
                errors.append(f"{time.perf_counter() - t0:.3f}s {type(e).__name__}: {e}")
                continue
            t = time.perf_counter() - t0
            n += 1
            if not args.benchmark:
                f = np.asarray(frame_contents[0])[0]          # [rx][chirps][samples] -> RX1
                frames.append(f.astype(np.float32))   # keep the SDK's floats as-is (they are signed, ~[-1, 1])
                stamps.append(t)
        elapsed = time.perf_counter() - t0

    meta["frames_received"] = n
    meta["achieved_fps"] = n / elapsed
    meta["expected_fps"] = d["frame_rate_Hz"]
    meta["sdk_errors"] = errors
    print(f"Frames: {n} in {elapsed:.1f} s -> {n / elapsed:.2f} fps (expected {d['frame_rate_Hz']:.1f}); "
          f"SDK errors: {len(errors)}")
    for e in errors[:5]:
        print("  ", e)

    if args.benchmark:
        return
    os.makedirs(args.outdir, exist_ok=True)
    dist = "na" if args.distance is None else f"{args.distance:g}"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.outdir, f"{stamp[:8]}_{args.preset}_{args.label}_{dist}m_{stamp[9:]}.npz")
    np.savez(path, frames=np.stack(frames), t=np.array(stamps), meta=json.dumps(meta))
    print("Saved", path)

    if args.show and "empty" not in args.label.lower():
        import types
        import analyze
        print()
        analyze.show(types.SimpleNamespace(
            file=path, background="auto", gate=None, moving=args.preset.startswith("B"),
            flash_vmin=1.0, flash_threshold=12.0))
        png = path.rsplit(".", 1)[0] + ".png"
        if hasattr(os, "startfile"):
            os.startfile(png)                      # Windows: open the plot
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", png])          # macOS: open in Preview


if __name__ == "__main__":
    main()
