"""
RATS radar node: acquire -> preprocess -> range FFT (numpy or PL) -> range-Doppler ->
detect/track -> classify -> UDP messages.

On the PYNQ (live radar):
    python3 node.py --source live --dest <listener IP>:5005
    python3 node.py --source live --fft pl --overlay ~/overlays/radar_fft.bit --dest <IP>:5005
On the Mac (replay a recording, fast or at 20 fps):
    python node.py --source replay recordings/<S_system file>.npz --fast
Logs one CSV row per frame with --log run.csv. Ctrl+C prints a summary.
"""
import argparse
import csv
import signal
import time

import numpy as np

from blade_det import BladeDetector, background_from_recording, combine
from classify import Classifier
from config import BLADE, FFT_SHIFT, NET, RADAR
from detect import Tracker
from dsp import NumpyRangeFFT, preprocess, range_doppler
from messages import Sender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["live", "replay"], default="live")
    ap.add_argument("files", nargs="*", help="recordings for --source replay")
    ap.add_argument("--fast", action="store_true", help="replay as fast as possible (no 20 fps pacing)")
    ap.add_argument("--loop", action="store_true", help="replay the files forever")
    ap.add_argument("--fft", choices=["numpy", "pl"], default="numpy")
    ap.add_argument("--overlay", default="/home/xilinx/overlays/radar_fft.bit")
    ap.add_argument("--per-chirp", action="store_true", help="PL FFT: one DMA transfer per chirp")
    ap.add_argument("--dest", default=f"{NET['dest_host']}:{NET['dest_port']}", help="listener host:port")
    ap.add_argument("--log", help="CSV file, one row per frame")
    ap.add_argument("--seconds", type=float, default=0, help="stop by itself after this many seconds (0 = run until Ctrl+C)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-blade", action="store_true", help="turn off the blade channel (small-drone detector)")
    ap.add_argument("--blade-bg", help="empty-scene profile for the blade channel: an empty .npz recording or a saved .npy "
                                       "(default: learn it from the first ~3.5 s, keep the area in front clear)")
    ap.add_argument("--blade-save", help="save the learned empty-scene profile to this .npy file")
    a = ap.parse_args()

    host, port = a.dest.rsplit(":", 1)
    tx = Sender(host, int(port), NET["source"])
    tx.send("radar_status", dict(state="initializing", fft_backend=a.fft, detail=a.source))

    # Load the FFT (and the FPGA overlay, ~10 s) BEFORE the radar starts streaming,
    # otherwise the reader thread drops frames while the bitstream loads
    if a.fft == "pl":
        from pl_fft import PLRangeFFT
        fft = PLRangeFFT(a.overlay, per_chirp=a.per_chirp)
    else:
        fft = NumpyRangeFFT(FFT_SHIFT)

    if a.source == "live":
        from sources import LiveSource
        src = LiveSource()
        # The radar library can take over Ctrl+C; give it back to Python so Ctrl+C stops cleanly
        signal.signal(signal.SIGINT, signal.default_int_handler)
    else:
        from sources import ReplaySource
        if not a.files:
            raise SystemExit("--source replay needs one or more .npz files")
        src = ReplaySource(a.files, realtime=not a.fast, loop=a.loop)

    tracker, clf = Tracker(), Classifier()
    blade = None
    if BLADE["enabled"] and not a.no_blade:
        bg = background_from_recording(a.blade_bg) if a.blade_bg else None
        blade = BladeDetector(background=bg)
        if not a.quiet:
            print("blade channel: " + (f"empty scene from {a.blade_bg}" if bg is not None else
                  "learning the empty scene for ~3.5 s, keep the area in front clear"))
    log = None
    if a.log:
        logf = open(a.log, "w", newline="")
        log = csv.writer(logf)
        log.writerow(["frame", "t_acq", "detected", "range_m", "speed_m_s", "snr_db", "spread_m_s",
                      "class", "confidence", "flash", "kurtosis", "flash_hz", "proc_ms", "latency_ms",
                      "source", "blade_excess_db", "blade_excess_hi_db", "blade_hits"])

    n, proc, t_start, t_status, last_cls = 0, [], time.time(), time.time(), None
    state = "running"
    try:
        for t_acq, frame in src:
            t0 = time.perf_counter()
            R = fft(preprocess(frame))
            Rm, RD = range_doppler(R)
            det = tracker.update(RD)
            label, conf, feats = clf.update(det, Rm)
            rng, spd, snr, source = det.range_m, det.speed_m_s, det.snr_db, "main"
            if blade is not None:
                was_learned = blade.learned
                b = blade.update(RD, det)
                rng, spd, snr, label, conf, feats, source = combine(det, label, conf, feats, b)
                if blade.learned and not was_learned:
                    if not a.quiet:
                        print("blade channel: empty scene learned, running")
                    if a.blade_save:
                        blade.save(a.blade_save)
            found = det.detected or source == "blade"
            proc_ms = (time.perf_counter() - t0) * 1e3
            proc.append(proc_ms)
            latency_ms = (time.time() - t_acq) * 1e3
            payload = dict(detected=found, range_m=round(rng, 2), speed_m_s=round(spd, 2),
                           snr_db=round(snr, 1), **{"class": label}, confidence=conf,
                           track_id=det.track_id if source != "blade" else 1000,
                           frame=n, latency_ms=round(latency_ms, 1), detector=source, features=feats)
            tx.send("radar_detection", payload, timestamp=t_acq)
            if log:
                log.writerow([n, f"{t_acq:.3f}", found, payload["range_m"], payload["speed_m_s"], payload["snr_db"],
                              round(det.spread_m_s, 2), label, conf, feats.get("flash"), feats.get("kurtosis"), feats.get("flash_hz"),
                              round(proc_ms, 2), round(latency_ms, 1), source, feats.get("blade_excess_db"),
                              feats.get("blade_excess_hi_db"), feats.get("blade_hits")])
            if not a.quiet and (n % 10 == 0 or label != last_cls):
                bx = feats.get("blade_excess_db")
                print(f"#{n:5d}  " + (f"{rng:5.2f} m  {spd:+5.2f} m/s  {snr:4.1f} dB  " if source != "blade" and found else
                                      f"{rng:5.2f} m  (blade)    {snr:4.1f} dB  " if source == "blade" else
                                      "   no target                     ")
                      + f"{label:8s} {conf:.2f}  {proc_ms:5.1f} ms"
                      + (f"  blade {bx:+.1f}/{feats.get('blade_excess_hi_db'):+.1f} dB {feats.get('blade_hits')}" if bx is not None else ""))
            last_cls = label
            n += 1
            if a.seconds and time.time() - t_start >= a.seconds:
                break
            if time.time() - t_status >= NET["status_period_s"]:
                el = time.time() - t_start
                tx.send("radar_status", dict(state="running", fps=round(n / el, 2), frames=n, sdk_errors=src.errors,
                                             skipped_frames=getattr(src, "skipped", 0),
                                             fft_backend=fft.name, proc_ms_mean=round(float(np.mean(proc[-40:])), 2),
                                             proc_ms_max=round(float(np.max(proc[-40:])), 2), detail=src.name))
                t_status = time.time()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        state = "disconnected" if "no frames" in str(e) else "fault"
        tx.send("radar_status", dict(state=state, frames=n, detail=f"{type(e).__name__}: {e}"))
        raise
    finally:
        el = max(time.time() - t_start, 1e-9)
        if state != "fault":
            tx.send("radar_status", dict(state="stopped", frames=n, fps=round(n / el, 2), sdk_errors=src.errors))
        if log:
            logf.close()
        if proc:
            print(f"\n{n} frames in {el:.1f} s ({n / el:.2f} fps), processing {np.mean(proc):.1f} ms mean / "
                  f"{np.max(proc):.1f} ms max per frame, sdk errors {src.errors}, "
                  f"skipped {getattr(src, 'skipped', 0)}, fft {fft.name}")
        if hasattr(src, "close"):
            src.close()


if __name__ == "__main__":
    main()
