"""
Signal processing: preprocessing (identical to golden_fft.py), range FFT backends, range-Doppler.

Range FFT backends return the same thing: complex [256 chirps, 64 range bins] equal to
FFT(x) / 2**FFT_SHIFT, i.e. what the Xilinx FFT IP outputs in scaled mode (first 64 bins).
"""
import numpy as np

from config import N, M, NB

Q15 = 32767
WIN_Q15 = np.round(np.hanning(N) * Q15).astype(np.int16)   # same table as golden_fft.py
_DOP_WIN = np.hanning(M)[:, None]


def preprocess(frame):
    """SDK frame float [256,128] -> int16 [256,128]: DC removal, Hann window, Q1.15."""
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x * (WIN_Q15.astype(np.float64) / Q15)
    return np.clip(np.round(x * Q15), -32768, 32767).astype(np.int16)


class NumpyRangeFFT:
    """Software range FFT on the ARM (used until the PL FFT passes the golden check)."""
    name = "numpy"

    def __init__(self, shift):
        self.scale = 2.0 ** -shift

    def __call__(self, x16):
        # rfft: real input, half the work of a full FFT; bins 0..63 are identical
        return np.fft.rfft(x16.astype(np.float64), axis=1)[:, :NB] * self.scale


def range_doppler(R):
    """R: complex [M, NB] range spectra -> (Rm, RD).
    Rm: R with the static scene removed (mean over chirps subtracted per range bin).
    RD: range-Doppler map, complex [M Doppler bins (fftshifted), NB range bins]."""
    Rm = R - R.mean(axis=0, keepdims=True)
    RD = np.fft.fftshift(np.fft.fft(Rm * _DOP_WIN, axis=0), axes=0)
    return Rm, RD
