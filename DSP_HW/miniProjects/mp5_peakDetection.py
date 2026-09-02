import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin


def generateSignal(sample_rate, duration, target_freq, target_amp=1.0):

    num_samples = int(sample_rate * duration)

    time = np.arange(num_samples) / sample_rate

    clean_signal = (
        target_amp
        * np.sin(2 * np.pi * target_freq * time)
    )

    return time, clean_signal


def computePower(signal):

    power = np.mean(signal ** 2)

    return power


def addNoiseSNR(clean_signal, snr_db):

    signal_power = computePower(clean_signal)

    noise_power = (
        signal_power
        / (10 ** (snr_db / 10))
    )

    noise_std = np.sqrt(noise_power)

    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=len(clean_signal)
    )

    noisy_signal = clean_signal + noise

    return noisy_signal, noise


def computeSpectrum(signal, sample_rate, apply_window=True):

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


def design_lpf(sample_rate, cutoff_freq, num_taps):

    coeffs = firwin(
        numtaps=num_taps,
        cutoff=cutoff_freq,
        fs=sample_rate
    )

    return coeffs


def apply_filter(signal, coeffs):

    filtered_signal = lfilter(
        coeffs,
        1.0,
        signal
    )

    return filtered_signal


def detect_peak(
    freqs,
    magnitude,
    threshold,
    min_frequency=5,
    max_frequency=250
):

    valid_region = (
        (freqs >= min_frequency)
        & (freqs <= max_frequency)
    )

    valid_freqs = freqs[valid_region]
    valid_magnitude = magnitude[valid_region]

    best_peak_frequency = 0.0
    best_peak_magnitude = 0.0
    detected = False

    for i in range(1, len(valid_magnitude) - 1):

        current = valid_magnitude[i]
        previous = valid_magnitude[i - 1]
        next_value = valid_magnitude[i + 1]

        is_local_maximum = (
            current > previous
            and current > next_value
        )

        is_above_threshold = current > threshold

        if is_local_maximum and is_above_threshold:

            if current > best_peak_magnitude:
                best_peak_magnitude = current
                best_peak_frequency = valid_freqs[i]
                detected = True

    return (
        best_peak_frequency,
        best_peak_magnitude,
        detected
    )


def main():

    # Simulation configuration
    sample_rate = 4000
    duration = 1.0

    target_freq = 72
    target_amp = 1.0
    target_snr_db = 10

    cutoff_freq = 325
    num_taps = 64

    threshold = 40.0

    # Generate the clean target signal
    time, clean_signal = generateSignal(
        sample_rate,
        duration,
        target_freq,
        target_amp
    )

    # Add Gaussian noise
    noisy_signal, noise = addNoiseSNR(
        clean_signal,
        target_snr_db
    )

    # Verify the measured SNR
    signal_power = computePower(clean_signal)
    noise_power = computePower(noise)

    measured_snr_db = 10 * np.log10(
        signal_power / noise_power
    )

    # Design and apply the FIR filter
    filter_coeffs = design_lpf(
        sample_rate,
        cutoff_freq,
        num_taps
    )

    filtered_signal = apply_filter(
        noisy_signal,
        filter_coeffs
    )

    # Compute the filtered signal spectrum
    freqs, magnitude = computeSpectrum(
        filtered_signal,
        sample_rate,
        apply_window=True
    )

    # Detect the strongest valid peak
    peak_freq, peak_magnitude, detected = detect_peak(
        freqs,
        magnitude,
        threshold
    )

    # Print results
    print("----------------------------------")
    print(f"Requested SNR: {target_snr_db:.2f} dB")
    print(f"Measured SNR:  {measured_snr_db:.2f} dB")
    print(f"Threshold:     {threshold:.2f}")

    if detected:
        print("Target detected!")
        print(f"Detected frequency: {peak_freq:.2f} Hz")
        print(f"Detected magnitude: {peak_magnitude:.2f}")
    else:
        print("No target detected.")

    print("----------------------------------")

    # Plot the spectrum
    plt.figure()

    plt.plot(
        freqs,
        magnitude,
        label="Filtered FFT magnitude"
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label="Detection threshold"
    )

    if detected:
        plt.scatter(
            peak_freq,
            peak_magnitude,
            label="Detected target"
        )

        plt.annotate(
            f"{peak_freq:.2f} Hz",
            xy=(peak_freq, peak_magnitude),
            xytext=(peak_freq + 15, peak_magnitude),
            arrowprops={"arrowstyle": "->"}
        )

    plt.xlim(0, 250)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Automatic Radar Peak Detection")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()