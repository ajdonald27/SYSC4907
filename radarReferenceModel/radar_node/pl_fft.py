"""
Range FFT in the PL (PYNQ overlay with AXI DMA + Xilinx FFT IP), same output as NumpyRangeFFT.

Built by fpga/build_radar_fft.tcl: DMA -> FFT -> axis_tlast_gen (TLAST every 32768 words) -> DMA,
so one transfer per frame. The FFT scaling (config.FFT_SHIFT) is written over AXI GPIO at start-up.
- per_chirp=True only works with a design WITHOUT axis_tlast_gen (TLAST every 128 words).
- Buffers are allocated once, and packing/unpacking uses int16 views (no int64 math: slow on the A9).

Timing check on the PYNQ (run from ~/radar_node, as root):
    python3 pl_fft.py --tv /home/xilinx/tv
"""
import numpy as np

from config import FFT_SHIFT, M, N, NB


def fft_config(shift):
    """Config word: bit 0 = forward; SCALE_SCH 2 bits per stage (4 stages, last at most 1)."""
    caps, s, left, i = [3, 3, 3, 1], [0, 0, 0, 0], shift, 0
    while left:
        if s[i % 4] < caps[i % 4]:
            s[i % 4] += 1
            left -= 1
        i += 1
    return 1 | (sum(v << (2 * k) for k, v in enumerate(s)) << 1)


def _alloc(allocate, n):
    try:                                    # cached buffers: the DMA driver flushes/invalidates them
        return allocate(shape=(n,), dtype=np.uint32, cacheable=True)
    except TypeError:
        return allocate(shape=(n,), dtype=np.uint32)


class PLRangeFFT:
    name = "pl"

    def __init__(self, bitfile, dma_name=None, per_chirp=False):
        from pynq import Overlay, allocate
        self.ol = Overlay(bitfile)
        if dma_name is None:
            names = [k for k in self.ol.ip_dict if "dma" in k.lower()]
            if not names:
                raise SystemExit(f"no DMA in {bitfile}; IP blocks: {list(self.ol.ip_dict)}")
            dma_name = names[0]
        self.dma = getattr(self.ol, dma_name)
        self.per_chirp = per_chirp
        n = N if per_chirp else M * N
        self.inb = _alloc(allocate, n)
        self.outb = _alloc(allocate, n)
        self.out16 = self.outb.view(np.int16).reshape(-1, N, 2)     # [.., bin, re/im]
        gpios = [k for k in self.ol.ip_dict if "gpio" in k.lower()]
        if gpios:                                   # FFT scaling + forward direction
            getattr(self.ol, gpios[0]).mmio.write(0x0, fft_config(FFT_SHIFT))
            self.inb[:] = 0
            self._run()                             # warm-up frame: config applies at a frame boundary

    def _run(self):
        self.dma.sendchannel.transfer(self.inb)
        self.dma.recvchannel.transfer(self.outb)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

    def pack(self, x16):
        # int16 -> uint16 -> uint32: real in [15:0], imaginary 0 in [31:16]
        return x16.view(np.uint16).reshape(-1)

    def unpack(self, rows=slice(None)):
        v = self.out16[rows, :NB]
        R = np.empty(v.shape[:2], dtype=np.complex64)
        R.real = v[..., 0]
        R.imag = v[..., 1]
        return R

    def __call__(self, x16):
        if self.per_chirp:
            R = np.empty((M, NB), dtype=np.complex64)
            for m in range(M):
                np.copyto(self.inb, self.pack(x16[m]))
                self._run()
                R[m] = self.unpack()[0]
            return R
        np.copyto(self.inb, self.pack(x16))
        self._run()
        return self.unpack()           # already scaled by the IP (2**-shift)


def _profile():
    import argparse
    import os
    import time

    from dsp import NumpyRangeFFT, range_doppler

    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="/home/xilinx/overlays/radar_fft.bit")
    ap.add_argument("--tv", default="/home/xilinx/tv")
    ap.add_argument("--frames", type=int, default=50)
    a = ap.parse_args()

    x16 = np.load(os.path.join(a.tv, "input_f0.npy")).astype(np.int16)
    E = np.load(os.path.join(a.tv, "expected_f0.npy"))[:, :NB]
    pl, sw = PLRangeFFT(a.overlay), NumpyRangeFFT(FFT_SHIFT)

    def ms(f):
        t = time.perf_counter()
        for _ in range(a.frames):
            f()
        return (time.perf_counter() - t) * 1e3 / a.frames

    t_pack = ms(lambda: np.copyto(pl.inb, pl.pack(x16)))
    t_dma = ms(pl._run)
    t_unpack = ms(pl.unpack)
    t_pl = ms(lambda: pl(x16))
    t_sw = ms(lambda: sw(x16))
    R = pl(x16)
    t_rd = ms(lambda: range_doppler(R))
    err = R - E
    snr = 10 * np.log10(np.sum(np.abs(E) ** 2) / max(np.sum(np.abs(err) ** 2), 1e-30))
    print(f"PL range FFT : {t_pl:6.2f} ms  (pack {t_pack:.2f} + DMA/FFT {t_dma:.2f} + unpack {t_unpack:.2f})")
    print(f"numpy rfft   : {t_sw:6.2f} ms")
    print(f"range-Doppler: {t_rd:6.2f} ms  (same for both)")
    print(f"PL vs golden : error SNR {snr:.1f} dB  {'PASS' if snr >= 40 else 'FAIL'}")


if __name__ == "__main__":
    _profile()
