# Signal Generation API for Python Reference Model 

# Author : AJ Donald
# Initial Rev: July 21st, 2026 
# Last Rev: July 21st, 2026 


import numpy as np 

def generate_multi_target_signal(sample_rate, duration, target_freqs, target_amplitudes):
    """ 
    Generated a simulated radar signal which contains multiple targets

    Each target is represented by a sinusoid with its own Doppler freq + Ampltitude.

    Returns: 
        time: 
        Numpy array containing sample times. 

        clean_signal: 
        NumPy array containing the combined target signal
    """

    if len(target_freqs) != len(target_amplitudes):
        raise ValueError("target_freqs and target_ampltitudes must have the same length.")
    
    num_samples = int(sample_rate * duration)

    time = np.arange(num_samples) / sample_rate 

    # define the signal with at least zero at all points in time
    clean_signal = np.zeros(num_samples)

    #clean_signal = np.sin(2 * np.pi * target_freqs * am)
    for frequency, amplitude in zip(
        target_freqs, target_amplitudes
    ):
        target_signal = (amplitude * np.sin(2 * np.pi * frequency * time))

        clean_signal += target_signal
    
    return time, clean_signal

def compute_power(signal):
    """
    Calculates the RMS (average) power of the signal
    """

    power = np.mean(signal ** 2)

    return power 

def add_noise_snr(clean_signal, target_snr_db, random_generator=None):

    """
    Add Gaussian noise to a clean signal, at a requested SNR.

    returns: 
        noisy_signal 
        noise
    """

    if random_generator is None:
        random_generator = (np.random.default_rng())

    signal_power = compute_power(clean_signal)

    noise_power = (signal_power / (10 **(target_snr_db / 10)))

    noise_std = np.sqrt(noise_power)

    noise = random_generator.normal(loc = 0.0, scale=noise_std, size = len(clean_signal))

    noisy_signal = clean_signal + noise 

    return noisy_signal, noise

