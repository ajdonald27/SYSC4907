#  FIR Filter API for Python Reference Model 

# Author : AJ Donald
# Initial Rev: July 21st, 2026 
# Last Rev: July 21st, 2026 

import numpy as np
from scipy.signal import firwin, lfilter 


def remove_dc(signal): 
    """
    Removes the average value from the radar signal. 

    A real radar singal may contain a large DC component caused by reflections, amplifier offsets or ADC offsets.
    """

    signal_mean = np.mean(signal)

    signal_without_dc = (signal - signal_mean)

    return signal_without_dc


def design_lpf(sample_rate, cutoff_freq, num_taps): 
    """
    Design a LP FIR filter 

    Frequencies below the cutoff_freq are preserved while higher-frequency noise is attenuated. 

    sample_rate : sampling_rate of the system
    cutoff_freq : cutoff frequency of the filter 
    num_taps : number of coefficients applied to the digital filter 
    """

    coeffs = firwin(numtaps = num_taps, cutoff = cutoff_freq, fs = sample_rate)

    return coeffs

def apply_fir_filter(signal, filter_coeffs): 
    """
    Apply an FIR filter to the signal
    """

    filtered_signal = lfilter(filter_coeffs, 1.0, signal)

    return filtered_signal 



