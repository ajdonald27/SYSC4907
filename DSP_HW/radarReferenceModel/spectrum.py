#  API to compute spectrum of the FFT in Python Reference Model 

# Author : AJ Donald
# Initial Rev: July 21st, 2026 
# Last Rev: July 21st, 2026 

import numpy as np 

def apply_hamming_window(signal):
    """
    Apply a Hamming window before the FFT. 

    The window reduces spectral leakage when a target frequency doesn't perfectly fit into an FFT bin.
    """

    num_samples = len(signal) 

    window = np.hamming(num_samples)
    windowed_signal = signal * window 

    return windowed_signal 


def compute_spectrum(signal, sample_rate): 
    """
    Comput ethe one-sided FFT magnitude spectrum.

    Returns:
        frequencies: 
            Frequency represented by each FFT bin.

        magnitude:
            Magnitude of each FFT bin

        frequency_resolution:
            deltaF = Fs/N

    """
    num_samples = len(signal)


    fft_output = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(num_samples, d=1/sample_rate)

    magnitude = np.abs(fft_output)

    frequency_resolution = sample_rate / num_samples
    return (frequencies, magnitude, frequency_resolution)

