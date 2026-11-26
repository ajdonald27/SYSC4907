"""
Self-test with synthetic data (no radar needed):  python selftest.py
Checks preprocessing against golden_fft.py, detection/tracking on a walking person,
person vs drone classification, latency per frame, and the message checks.
"""
import json
import os
import sys
import time

import numpy as np

import sim
from classify import Classifier
from config import FFT_SHIFT
from detect import Tracker
from dsp import NumpyRangeFFT, preprocess, range_doppler
from listener import Checker
from messages import Sender, seal, verify

ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


def run(frames):
    fft, tr, clf, out, ms = NumpyRangeFFT(FFT_SHIFT), Tracker(), Classifier(), [], []
    for f in frames:
        t0 = time.perf_counter()
        Rm, RD = range_doppler(fft(preprocess(f)))
        det = tr.update(RD)
        lab, conf, feats = clf.update(det, Rm)
        ms.append((time.perf_counter() - t0) * 1e3)
        out.append((det, lab, conf, feats))
    return out, ms


# 1. preprocessing identical to the golden model
here = os.path.dirname(os.path.abspath(__file__))
for cand in [os.path.join(here, "..", "characterization"), os.path.join(here, "..", "rats_radar_char")]:
    if os.path.exists(os.path.join(cand, "golden_fft.py")):
        sys.path.insert(0, cand)
        import golden_fft
        f = sim.frames("walker", 0.05)[0]
        check("preprocess matches golden_fft.py", np.array_equal(preprocess(f), golden_fft.preprocess(f)))
        break
else:
    print("[SKIP] golden_fft.py not found next to radar_node")

# 2. walking person: range, speed, class
seconds, start = 2.0, 3.5
res, ms = run(sim.frames("walker", seconds, distance=start))
dets = [(i, d) for i, (d, *_rest) in enumerate(res) if d.detected]
check("person detected in >= 90 % of frames", len(dets) >= 0.9 * len(res), f"({len(dets)}/{len(res)})")
err = [abs(d.range_m - (start - 1.2 * (i * 0.05 + 0.019))) for i, d in dets]
check("range within +/-1 m (FR2)", max(err) <= 1.0, f"(max error {max(err):.2f} m)")
v = np.median([d.speed_m_s for _, d in dets])
check("speed within +/-2 m/s (FR2), approaching negative", abs(v - (-1.2)) <= 2.0 and v < 0, f"(median {v:+.2f} m/s, true -1.20)")
labels = [lab for _, lab, *_ in res[10:]]
frac = labels.count("person") / len(labels)
check("classified person", frac >= 0.75, f"({frac:.0%} of frames after the first 0.5 s)")
check("no blade flash on the person", sum(f.get("flash", False) for *_x, f in res) <= 2)
check("processing time per frame", np.mean(ms) < 50, f"(mean {np.mean(ms):.1f} ms here; budget 50 ms per frame)")

# 3. spinning blades (drone stand-in)
res, _ = run(sim.frames("rotor", 1.5, distance=2.0, rps=40, blades=2))
labels = [lab for _, lab, *_ in res[5:]]
frac = labels.count("drone") / len(labels)
fl = [f.get("flash_hz") for *_x, f in res if f.get("flash_hz")]
check("rotor classified drone", frac >= 0.75, f"({frac:.0%}; flash-rate estimate ~{np.median(fl) if fl else float('nan'):.0f} Hz, true 2 blades x 40 rev/s = 80 Hz)")
r = np.median([d.range_m for d, *_ in res if d.detected])
check("rotor range", abs(r - 2.0) <= 0.5, f"({r:.2f} m)")

res, _ = run(sim.frames("rotor", 1.5, distance=5.0, rps=80, blades=2))
labels = [lab for _, lab, *_ in res[5:]]
check("rotor at 5 m classified drone", labels.count("drone") / len(labels) >= 0.75, f"({labels.count('drone') / len(labels):.0%})")

# 4. empty room: no target
res, _ = run(sim.frames("empty", 1.0))
check("empty room: no detections", sum(d.detected for d, *_ in res) <= 1)

# 4b. blade channel: weak drone (body far below the 15 dB detector), walker, empty scene
from blade_det import BladeDetector, combine


def run_blade(frames, bd):
    fft, tr, clf, out = NumpyRangeFFT(FFT_SHIFT), Tracker(), Classifier(), []
    for f in frames:
        Rm, RD = range_doppler(fft(preprocess(f)))
        det = tr.update(RD)
        lab, conf, feats = clf.update(det, Rm)
        b = bd.update(RD, det)
        out.append((det, b, combine(det, lab, conf, feats, b)))
    return out


def learned():
    bd = BladeDetector()
    run_blade(sim.frames("empty", 3.6, seed=1), bd)
    return bd


res = run_blade(sim.frames("rotor", 2.5, distance=2.5, rps=80, rotor_amp=0.01, seed=2), learned())
main_hits = sum(d.detected for d, *_ in res)
labels = [c[3] for *_x, c in res[20:]]
rr = [c[0] for *_x, c in res[20:] if c[3] == "drone"]
ex = np.median([b.excess_db for _, b, _ in res])
check("weak drone (~3 dB blade excess): main detector misses it", main_hits <= 2, f"({main_hits} frames)")
check("weak drone: blade channel says drone", labels.count("drone") / len(labels) >= 0.9,
      f"({labels.count('drone')}/{len(labels)} frames, excess {ex:.1f} dB)")
check("weak drone: blade range", bool(rr) and abs(np.median(rr) - 2.5) <= 0.3, f"({np.median(rr) if rr else float('nan'):.2f} m)")
res = run_blade(sim.frames("walker", 2.0, distance=3.5, seed=6), learned())
labels = [c[3] for *_x, c in res[10:]]
check("walker is never drone with the blade channel on", labels.count("drone") == 0 and labels.count("person") >= 0.75 * len(labels),
      str({k: labels.count(k) for k in set(labels)}))
fr = np.concatenate([sim.frames("walker", 2.0, distance=3.5, seed=6, walker_amp=0.05), sim.frames("empty", 3.0, seed=7)])
res = run_blade(fr, learned())
check("walker leaving the scene is never drone (Nov 25 5-min run)", sum(c[3] == "drone" for *_x, c in res) == 0,
      f"(max blade excess {max(b.excess_db for _, b, _ in res if b.state == 'running'):.1f} dB)")
res = run_blade(sim.frames("empty", 10.0, seed=5), learned())
check("empty scene: blade channel quiet for 10 s", sum(c[3] == "drone" for *_x, c in res) == 0,
      f"(max excess {max(b.excess_db for _, b, _ in res):.2f} dB)")

# 5. messages
m = seal(dict(type="radar_detection", source="radar-pynq", seq=7, timestamp=time.time(), payload={"range_m": 2.0}))
check("checksum verifies", verify(m))
m2 = dict(m, payload={"range_m": 3.0})
check("tampered message rejected", not verify(m2))
ck = Checker(window_s=1.0)
now = time.time()
for seq in (0, 1, 4):
    ck.check(json.dumps(seal(dict(type="t", source="s", seq=seq, timestamp=now, payload={}))), now)
ck.check(json.dumps(seal(dict(type="t", source="s", seq=5, timestamp=now - 5, payload={}))), now)
ck.check(b"not json", now)
check("listener counts lost / stale / malformed", ck.stats == dict(ok=3, malformed=1, lost=2, stale=1), str(ck.stats))
tx = Sender("127.0.0.1", 59999, "radar-pynq")
sent = tx.send("radar_detection", {"x": float("nan"), "y": np.float32(1.5)})
check("sender cleans NaN / numpy values", sent["payload"] == {"x": None, "y": 1.5} and verify(sent))

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
