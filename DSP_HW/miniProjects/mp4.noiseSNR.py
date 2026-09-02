import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import lfilter, firwin 

# simple signal 
sample_rate = 4000 
target_freq = 72 
target_amp = 1.0 
duration = 1 

# time axis 
number_samples =  int(duration * sample_rate)


# creates an array of time 

time = np.arange(number_samples) / sample_rate
print(time)


# clean target signal 

clean_signal = np.sin(2 * np.pi * target_freq * time)

# signalPower function 
def computePower(signal): 
    power = np.mean(signal**2) 
    return power 

signal_power = computePower(clean_signal)
target_snr_db = 10
print("Signal power: ", signal_power)

# SNR in dB 

noise_power = signal_power / 10 ** (target_snr_db / 10)

# convert noise power into standard deviation 

# Pnoise = sigma**2 
# sigma = sqrt(Pnoise)

noise_std = np.sqrt(noise_power)

# generate noise 

noise = np.random.normal(loc = 0.0, scale = noise_std, size=len(clean_signal))

noisy_signal = clean_signal + noise 

# put noise generation into a function 

def add_noise_at_snr(clean_signal, snr_db):
    signal_power = computePower(clean_signal)
    noise_power = signal_power / 10 ** (target_snr_db /10)

    noise_std = np.sqrt(noise_power)

    noise = np.random.normal(loc = 0.0, scale=noise_std, size=len(clean_signal))

    noisy_signal = clean_signal + noise 

    return noisy_signal, noise

measured_signal_power = computePower(clean_signal)
measured_noise_power = computePower(noise)

measured_snr_db = 10 * np.log10(measured_signal_power/measured_noise_power)

print("Requested SNR : ", target_snr_db)
print("Measured SNR : ", measured_snr_db)


plt.figure()
plt.plot(time, clean_signal, label='clean_signal')
plt.plot(time, noise, label ='noise')

#plt.plot(time, noisy_signal, label='noisy_signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f"Signals at {target_snr_db} dB SNR")
plt.legend()
plt.xlim(0,0.1)
plt.grid()
plt.show()

# apply FFT 

def compute_spectrum(signal, sample_rate, apply_window=True):
    N = len(signal)

    if apply_window:
        window = np.hamming(N)
        processed_signal = signal * window
    else:
        processed_signal = signal

    fft_output = np.fft.rfft(processed_signal)
    magnitude = np.abs(fft_output)

    freqs = np.fft.rfftfreq(
        N,
        d=1 / sample_rate
    )

    return freqs, magnitude


# example call to the function 
freqs, magnitude = compute_spectrum(noisy_signal, sample_rate, apply_window=True) 

plt.figure() 
plt.plot(freqs, magnitude)
plt.xlim(0,250)
plt.xlabel("Freq (Hz)")
plt.ylabel("Magnitude")
plt.title(f"Noisy signal spectrum - {target_snr_db} dB SNR")
plt.grid() 
plt.show() 


clean_freqs, clean_magnitude = compute_spectrum(clean_signal, sample_rate)

reference_magnitude = np.max(clean_magnitude)

# compare  SNR levels 
snr_levels_db = [30,20,10,0]
from scipy.signal import firwin, lfilter


def measure_peak_to_noise(
    freqs,
    magnitude,
    target_frequency,
    noise_min_frequency=30,
    noise_max_frequency=150,
    exclusion_width=10
):
    target_index = np.argmin(
        np.abs(freqs - target_frequency)
    )

    target_magnitude = magnitude[target_index]

    noise_region = (
        (freqs >= noise_min_frequency)
        & (freqs <= noise_max_frequency)
        & (
            np.abs(freqs - target_frequency)
            > exclusion_width
        )
    )

    noise_magnitudes = magnitude[noise_region]

    noise_floor = np.mean(noise_magnitudes)

    epsilon = 1e-12

    peak_to_noise_db = 20 * np.log10(
        (target_magnitude + epsilon)
        / (noise_floor + epsilon)
    )

    return peak_to_noise_db, target_magnitude, noise_floor


# Clean reference spectrum
clean_freqs, clean_magnitude = compute_spectrum(
    clean_signal,
    sample_rate,
    apply_window=True
)

reference_magnitude = np.max(clean_magnitude)

epsilon = 1e-12


# FIR filter design
num_taps = 64
cutoff_frequency = 300

filter_coefficients = firwin(
    numtaps=num_taps,
    cutoff=cutoff_frequency,
    fs=sample_rate
)


# Test several SNR levels
snr_levels_db = [30, 20, 10, 0]

for snr_db in snr_levels_db:

    noisy_signal, noise = add_noise_at_snr(
        clean_signal,
        snr_db
    )

    measured_signal_power = computePower(clean_signal)
    measured_noise_power = computePower(noise)

    measured_snr_db = 10 * np.log10(
        measured_signal_power / measured_noise_power
    )

    filtered_signal = lfilter(
        filter_coefficients,
        1.0,
        noisy_signal
    )

    freqs, noisy_magnitude = compute_spectrum(
        noisy_signal,
        sample_rate,
        apply_window=True
    )

    filtered_freqs, filtered_magnitude = compute_spectrum(
        filtered_signal,
        sample_rate,
        apply_window=True
    )

    noisy_magnitude_db = 20 * np.log10(
        noisy_magnitude / reference_magnitude + epsilon
    )

    filtered_magnitude_db = 20 * np.log10(
        filtered_magnitude / reference_magnitude + epsilon
    )

    before_peak_to_noise_db, _, _ = measure_peak_to_noise(
        freqs,
        noisy_magnitude,
        target_freq
    )

    after_peak_to_noise_db, _, _ = measure_peak_to_noise(
        filtered_freqs,
        filtered_magnitude,
        target_freq
    )

    print(f"Requested SNR: {snr_db:.2f} dB")
    print(f"Measured SNR:  {measured_snr_db:.2f} dB")
    print(
        f"Peak-to-noise before filtering: "
        f"{before_peak_to_noise_db:.2f} dB"
    )
    print(
        f"Peak-to-noise after filtering:  "
        f"{after_peak_to_noise_db:.2f} dB"
    )
    print()

    plt.figure()

    plt.plot(
        freqs,
        noisy_magnitude_db,
        label="Before filtering"
    )

    plt.plot(
        filtered_freqs,
        filtered_magnitude_db,
        label="After filtering"
    )

    plt.xlim(0, 1000)
    plt.ylim(-100, 10)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude relative to clean peak (dB)")
    plt.title(f"FIR Comparison — SNR = {snr_db} dB")

    plt.legend()
    plt.grid()
    plt.show()
# suggested structure 


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import firwin, lfilter


# def generate_target(...):
#     # Return time and clean sinusoidal target


# def calculate_power(...):
#     # Return average signal power


# def add_noise_at_snr(...):
#     # Calculate required noise power
#     # Generate Gaussian noise
#     # Return noisy signal and noise


# def design_low_pass_filter(...):
#     # Return FIR coefficients


# def apply_fir_filter(...):
#     # Apply the FIR filter


# def compute_spectrum(...):
#     # Apply window
#     # Compute rFFT
#     # Return frequency and magnitude arrays


# def measure_peak_to_noise(...):
#     # Locate target bin
#     # Estimate nearby noise floor
#     # Return ratio in dB


# def main():
#     # Configuration

#     # Generate clean target

#     # Compute clean reference spectrum

#     # Design filter

#     # Loop through SNR levels

#         # Add noise

#         # Verify measured SNR

#         # Filter noisy signal

#         # Compute before/after spectra

#         # Measure peak-to-noise ratio

#         # Plot results


# if __name__ == "__main__":
#     main()