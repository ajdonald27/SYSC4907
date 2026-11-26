"""
RATS golden model for the range FFT in the PL (S_system frame format, docs/radar_frame_format.md).

The SAME preprocess() runs on the PYNQ before DMA, so the model and the hardware see
identical int16 inputs. The reference output is an exact FFT of those inputs, scaled by
2**-shift like the Xilinx FFT IP in scaled mode; the PL output must match it within the
tolerance below (fixed-point rounding means "close", not bit-exact).

    python golden_fft.py vectors <S_system recording>.npz [--frames 3] [--out tv]
        -> tv/: DMA input words, expected outputs, a tone test frame, scaling.json, README.txt
    python golden_fft.py compare tv/expected_f0.npy <pl_output>.npy
        -> PASS/FAIL: peak bins agree, error SNR, worst error in LSBs
"""
import argparse
import json
import os
import sys

import numpy as np

N, M = 128, 256          # samples per chirp (range FFT size), chirps per frame
Q15 = 32767
WIN_Q15 = np.round(np.hanning(N) * Q15).astype(np.int16)   # exact window table used everywhere


def preprocess(frame):
    """SDK frame float [256,128] -> int16 [256,128]: DC removal, Hann window, Q1.15."""
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x * (WIN_Q15.astype(np.float64) / Q15)
    return np.clip(np.round(x * Q15), -32768, 32767).astype(np.int16)


def to_words(x16):
    """int16 [M,N] -> uint32 DMA words: real in [15:0], imag = 0 in [31:16]."""
    return (x16.astype(np.int64) & 0xFFFF).astype(np.uint32)


def reference(x16, shift):
    """Ideal scaled FFT of each chirp: FFT(x) / 2**shift (complex, float)."""
    return np.fft.fft(x16.astype(np.float64), axis=1) / (2 ** shift)


def pack_complex(X):
    """complex -> uint32 words: real int16 in [15:0], imag int16 in [31:16] (ideal rounding)."""
    re = np.clip(np.round(X.real), -32768, 32767).astype(np.int64) & 0xFFFF
    im = np.clip(np.round(X.imag), -32768, 32767).astype(np.int64) & 0xFFFF
    return ((im << 16) | re).astype(np.uint32)


def unpack_complex(w):
    w = np.asarray(w, dtype=np.uint32).astype(np.int64)
    re = ((w & 0xFFFF) ^ 0x8000) - 0x8000
    im = (((w >> 16) & 0xFFFF) ^ 0x8000) - 0x8000
    return re + 1j * im


def pick_shift(frames16):
    """Smallest total shift keeping the largest output below a quarter of full scale (12 dB headroom,
    room for targets stronger than the ones in this recording)."""
    peak = max(float(np.abs(np.fft.fft(f.astype(np.float64), axis=1)).max()) for f in frames16)
    shift = 0
    while peak / 2 ** shift > Q15 / 4 and shift < 7:
        shift += 1
    return shift, peak


def write_hex(path, words):
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v):08x}" for v in np.ravel(words)) + "\n")


def tone_frame(shift):
    """Chirp m holds a cosine at bin (m % 60) + 2; chirp 255 is an impulse. The amplitude is set so
    the tone peak (amplitude x 64) lands at about a quarter of full scale after the chosen shift."""
    amp = min(8000, int(Q15 / 4 * 2 ** shift / (N / 2)))
    n = np.arange(N)
    x = np.zeros((M, N))
    for m in range(M - 1):
        x[m] = amp * np.cos(2 * np.pi * ((m % 60) + 2) * n / N)
    x[M - 1, 0] = amp
    return np.round(x).astype(np.int16)


def vectors(a):
    z = np.load(a.recording, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    frames = z["frames"]
    if frames.shape[1:] != (M, N):
        sys.exit(f"expected S_system frames of shape (*, {M}, {N}); got {frames.shape} ({meta.get('preset')})")
    all16 = [preprocess(f) for f in frames]
    shift, peak = pick_shift(all16)
    os.makedirs(a.out, exist_ok=True)
    picks = np.linspace(0, len(frames) - 1, a.frames).round().astype(int)
    for k, i in enumerate(picks):
        x16 = all16[i]
        X = reference(x16, shift)
        np.save(f"{a.out}/input_f{k}.npy", x16)
        write_hex(f"{a.out}/input_f{k}.hex", to_words(x16))
        np.save(f"{a.out}/expected_f{k}.npy", X)
        write_hex(f"{a.out}/expected_f{k}.hex", pack_complex(X))
    t16 = tone_frame(shift)
    np.save(f"{a.out}/input_tone.npy", t16)
    write_hex(f"{a.out}/input_tone.hex", to_words(t16))
    np.save(f"{a.out}/expected_tone.npy", reference(t16, shift))
    write_hex(f"{a.out}/expected_tone.hex", pack_complex(reference(t16, shift)))
    np.save(f"{a.out}/window_q15.npy", WIN_Q15)
    write_hex(f"{a.out}/window_q15.hex", (WIN_Q15.astype(np.int64) & 0xFFFF).astype(np.uint32))
    info = dict(fft_size=N, chirps_per_frame=M, total_shift=shift,
                largest_unscaled_output=round(peak), headroom_dB=round(20 * np.log10(Q15 / (peak / 2 ** shift)), 1),
                source=os.path.basename(a.recording), frames_used=[int(i) for i in picks])
    json.dump(info, open(f"{a.out}/scaling.json", "w"), indent=2)
    open(f"{a.out}/README.txt", "w").write(README.format(**info))
    print(json.dumps(info, indent=2))
    print(f"Wrote {a.frames} recorded frames + the tone frame to {a.out}/")


README = """Range-FFT test vectors (S_system), from {source}

Each frame is {chirps_per_frame} chirps x {fft_size} samples, chirp-major (chirp 0 samples 0..127, then chirp 1 ...).
input_*.hex     one 32-bit DMA word per line: real int16 in [15:0], imag 0 in [31:16]
expected_*.hex  ideal FFT output words: real int16 in [15:0], imag int16 in [31:16]
*.npy           the same data for Python (input int16; expected complex float, before rounding)
input_tone      chirp m = cosine at bin (m % 60) + 2; chirp 255 = impulse.
                Expected peaks at bins (m % 60) + 2 and 128 - that; the impulse gives a flat spectrum.
window_q15      the 128-point Hann window table (Q1.15) applied before the FFT (on the ARM for now).

FFT IP: forward, 128-point, scaled, natural order. Total scaling = 2^-{total_shift}
(largest unscaled output in the data = {largest_unscaled_output}; headroom after scaling = {headroom_dB} dB).
Set SCALE_SCH so the per-stage shifts add up to {total_shift}, putting the shifts in the early stages first
(check the bit layout for Pipelined Streaming I/O in the FFT IP product guide, PG109).

Check the hardware output with:  python golden_fft.py compare expected_f0.npy <pl_output.npy>
(pl_output may be uint32 words or complex values, shape 256 x 128)
"""


def compare(a):
    E = np.load(a.expected)
    P = np.load(a.pl)
    if P.dtype.kind in "ui":
        P = unpack_complex(P)
    P = P.reshape(E.shape)
    err = P - E
    snr = 10 * np.log10(np.sum(np.abs(E) ** 2) / max(np.sum(np.abs(err) ** 2), 1e-30))
    worst = float(np.max(np.maximum(np.abs(err.real), np.abs(err.imag))))
    pe = np.argmax(np.abs(E[:, 1:N // 2]), axis=1)
    pp = np.argmax(np.abs(P[:, 1:N // 2]), axis=1)
    agree = float(np.mean(pe == pp)) * 100
    ok = snr >= a.min_snr and agree >= 99.0
    print(f"error SNR        {snr:6.1f} dB   (pass >= {a.min_snr} dB)")
    print(f"worst error      {worst:6.1f} LSB")
    print(f"peak bins agree  {agree:6.1f} %    (pass >= 99 %)")
    print("PASS" if ok else "FAIL")
    if not ok and agree < 50:
        print("Hint: most peaks differ. Check bit order (real in [15:0]), natural vs bit-reversed output, and scaling.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vectors")
    v.add_argument("recording")
    v.add_argument("--frames", type=int, default=3)
    v.add_argument("--out", default="tv")
    c = sub.add_parser("compare")
    c.add_argument("expected")
    c.add_argument("pl")
    c.add_argument("--min-snr", type=float, default=40.0)
    a = ap.parse_args()
    vectors(a) if a.cmd == "vectors" else compare(a)
