"""
RATS: test the FPGA overlay on the PYNQ against the golden model (run as root in a Jupyter terminal).

    python3 test_fft_overlay.py --bit ~/overlays/radar_loopback.bit          # bring-up design
    python3 test_fft_overlay.py --bit ~/overlays/radar_fft.bit --tv ~/tv      # FFT design

Loopback: sends a counting pattern and checks it comes back unchanged.
FFT: sets the FFT scaling over AXI GPIO, sends each tv/input_*.npy frame through
DMA -> FFT -> DMA, saves tv/pl_output_*.npy, and compares with tv/expected_*.npy
(PASS = error SNR >= 40 dB and the same peak bin in >= 99 % of chirps).
"""
import argparse
import glob
import json
import os
import time

import numpy as np

M, N = 256, 128


def fft_config(shift):
    """Xilinx FFT config word for a fixed 128-point, scaled, pipelined core:
    bit 0 = FWD_INV (1 = forward); then SCALE_SCH, 2 bits per radix-4 stage (4 stages, the
    last one radix-2 so at most 1). The shift is spread over the early stages first."""
    caps, s, left, i = [3, 3, 3, 1], [0, 0, 0, 0], shift, 0
    if not 0 <= shift <= sum(caps):
        raise ValueError(f"shift must be 0..{sum(caps)}")
    while left:
        if s[i % 4] < caps[i % 4]:
            s[i % 4] += 1
            left -= 1
        i += 1
    sch = sum(v << (2 * k) for k, v in enumerate(s))
    return 1 | (sch << 1), s


def unpack(words):
    w = np.asarray(words, dtype=np.uint32).astype(np.int64)
    re = ((w & 0xFFFF) ^ 0x8000) - 0x8000
    im = (((w >> 16) & 0xFFFF) ^ 0x8000) - 0x8000
    return re + 1j * im


def compare(E, P):
    err = P - E
    snr = 10 * np.log10(np.sum(np.abs(E) ** 2) / max(np.sum(np.abs(err) ** 2), 1e-30))
    worst = float(np.max(np.maximum(np.abs(err.real), np.abs(err.imag))))
    pe = np.argmax(np.abs(E[:, 1:N // 2]), axis=1)
    pp = np.argmax(np.abs(P[:, 1:N // 2]), axis=1)
    agree = float(np.mean(pe == pp)) * 100
    # how much the hardware actually scaled, relative to the model (0 = as expected)
    big = np.abs(E) > 0.25 * np.abs(E).max()
    ratio = np.median(np.abs(P[big]) / np.abs(E[big])) if big.any() else float("nan")
    return snr, worst, agree, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", required=True)
    ap.add_argument("--tv", default=os.path.expanduser("~/tv"))
    ap.add_argument("--shift", type=int, default=None, help="default: total_shift from tv/scaling.json")
    a = ap.parse_args()

    from pynq import Overlay, allocate
    ol = Overlay(os.path.expanduser(a.bit))
    ips = list(ol.ip_dict)
    print("IP blocks:", ips)
    dma = getattr(ol, [k for k in ips if "dma" in k.lower()][0])
    n = M * N
    inb = allocate(shape=(n,), dtype=np.uint32)
    outb = allocate(shape=(n,), dtype=np.uint32)

    def run():
        dma.sendchannel.transfer(inb)
        dma.recvchannel.transfer(outb)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

    def timing(label):
        t = time.time()
        for _ in range(50):
            run()
        print(f"{label}: {(time.time() - t) * 20:.2f} ms per frame round trip (a frame arrives every 50 ms)")

    # the FFT IP has no AXI-Lite port, so it is not in ip_dict; the FFT design is the one with the GPIO
    if not any(("fft" in k.lower() or "gpio" in k.lower()) for k in ips):
        inb[:] = np.arange(n, dtype=np.uint32)
        run()
        ok = np.array_equal(np.array(inb), np.array(outb))
        print("LOOPBACK", "PASS" if ok else "FAIL (data came back different)")
        timing("loopback")
        return

    shift = a.shift
    if shift is None:
        shift = json.load(open(os.path.join(a.tv, "scaling.json")))["total_shift"]
    cfg, stages = fft_config(shift)
    gpio = getattr(ol, [k for k in ips if "gpio" in k.lower()][0])
    gpio.mmio.write(0x0, cfg)            # GPIO channel 1 data register
    print(f"FFT config: forward, total shift {shift} (per stage {stages}) -> word 0x{cfg:04x}")

    names = sorted(os.path.basename(p)[6:-4] for p in glob.glob(os.path.join(a.tv, "input_*.npy")))
    names = ["tone"] + [x for x in names if x != "tone"]          # tone first: easiest to read
    first = True
    for name in names:
        x16 = np.load(os.path.join(a.tv, f"input_{name}.npy"))
        inb[:] = (x16.astype(np.int64) & 0xFFFF).astype(np.uint32).ravel()
        if first:                         # warm-up: the config is picked up at a frame boundary
            run()
            first = False
        run()
        words = np.array(outb).reshape(M, N)
        np.save(os.path.join(a.tv, f"pl_output_{name}.npy"), words)
        E = np.load(os.path.join(a.tv, f"expected_{name}.npy"))
        snr, worst, agree, ratio = compare(E, unpack(words))
        ok = snr >= 40 and agree >= 99
        print(f"{name:5s}  error SNR {snr:6.1f} dB  worst {worst:7.1f} LSB  peaks agree {agree:5.1f} %  "
              f"scale vs model x{ratio:.3g}  {'PASS' if ok else 'FAIL'}")
        if not ok and agree >= 99 and ratio == ratio and abs(np.log2(ratio)) >= 0.9:
            print(f"       peaks are right but the size is off by about 2^{np.log2(ratio):+.0f}: "
                  f"the scaling bits are being read differently; try --shift {shift + round(np.log2(ratio))}")
        elif not ok and agree < 50:
            print("       most peaks differ: check the word layout (real in [15:0]), natural order, forward FFT")
    timing("FFT")


if __name__ == "__main__":
    main()
