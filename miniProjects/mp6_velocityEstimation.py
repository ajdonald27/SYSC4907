import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin


def generateSignal(
    sample_rate,
    duration,
    target_freq,
    target_amp=1.0
):
    """
    Generate a clean sinusoidal Doppler signal.

    Returns:
        time: Time-axis array
        clean_signal: Generated sine wave
    """

    num_samples = int(sample_rate * duration)

    time = np.arange(num_samples) / sample_rate

    clean_signal = (
        target_amp
        * np.sin(2 * np.pi * target_freq * time)
    )

    return time, clean_signal


def computePower(signal):
    """
    Calculate the average power of a signal.
    """

    power = np.mean(signal ** 2)

    return power


def addNoiseSNR(clean_signal, snr_db):
    """
    Add Gaussian noise to a clean signal at a requested SNR.

    Returns:
        noisy_signal
        noise
    """

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


def computeSpectrum(
    signal,
    sample_rate,
    apply_window=True
):
    """
    Compute a one-sided FFT magnitude spectrum.

    Returns:
        freqs
        magnitude
    """

    number_of_samples = len(signal)

    if apply_window:
        window = np.hamming(number_of_samples)
        processed_signal = signal * window
    else:
        processed_signal = signal

    fft_output = np.fft.rfft(processed_signal)

    magnitude = np.abs(fft_output)

    freqs = np.fft.rfftfreq(
        number_of_samples,
        d=1 / sample_rate
    )

    return freqs, magnitude


def design_lpf(
    sample_rate,
    cutoff_freq,
    num_taps
):
    """
    Design a low-pass FIR filter.
    """

    coeffs = firwin(
        numtaps=num_taps,
        cutoff=cutoff_freq,
        fs=sample_rate
    )

    return coeffs


def apply_filter(signal, coeffs):
    """
    Apply an FIR filter to a signal.
    """

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
    """
    Find the strongest local FFT maximum above a threshold
    within the selected frequency range.

    Returns:
        best_peak_frequency
        best_peak_magnitude
        detected
    """

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

        previous_value = valid_magnitude[i - 1]
        current_value = valid_magnitude[i]
        next_value = valid_magnitude[i + 1]

        is_local_maximum = (
            current_value > previous_value
            and current_value > next_value
        )

        is_above_threshold = (
            current_value > threshold
        )

        if is_local_maximum and is_above_threshold:

            if current_value > best_peak_magnitude:

                best_peak_magnitude = current_value
                best_peak_frequency = valid_freqs[i]
                detected = True

    return (
        best_peak_frequency,
        best_peak_magnitude,
        detected
    )


def doppler_to_velocity(
    doppler_freq,
    carrier_freq=10.525e9
):
    """
    Convert Doppler frequency into radial velocity
    for a monostatic radar.

    Returns:
        velocity in metres per second
    """

    light_speed = 299_792_458.0

    wavelength = (
        light_speed
        / carrier_freq
    )

    velocity = (
        doppler_freq
        * wavelength
        / 2
    )

    return velocity


def main():

    # Use a fixed seed while debugging so that each run
    # produces the same random noise.
    np.random.seed(42)

    # -------------------------------------------------
    # Simulation configuration
    # -------------------------------------------------

    sample_rate = 4000
    duration = 1.0

    target_freq = 72
    target_amp = 1.0
    target_snr_db = 10

    carrier_freq = 10.525e9

    cutoff_freq = 325
    num_taps = 64

    threshold = 40.0

    minimum_detection_frequency = 5
    maximum_detection_frequency = 250

    # -------------------------------------------------
    # Generate the clean target signal
    # -------------------------------------------------

    time, clean_signal = generateSignal(
        sample_rate,
        duration,
        target_freq,
        target_amp
    )

    # -------------------------------------------------
    # Add Gaussian noise
    # -------------------------------------------------

    noisy_signal, noise = addNoiseSNR(
        clean_signal,
        target_snr_db
    )

    # -------------------------------------------------
    # Measure the generated SNR
    # -------------------------------------------------

    signal_power = computePower(clean_signal)
    noise_power = computePower(noise)

    measured_snr_db = 10 * np.log10(
        signal_power / noise_power
    )

    # -------------------------------------------------
    # Design and apply the FIR low-pass filter
    # -------------------------------------------------

    filter_coeffs = design_lpf(
        sample_rate,
        cutoff_freq,
        num_taps
    )

    filtered_signal = apply_filter(
        noisy_signal,
        filter_coeffs
    )

    # -------------------------------------------------
    # Compute the FFT spectrum
    # -------------------------------------------------

    freqs, magnitude = computeSpectrum(
        filtered_signal,
        sample_rate,
        apply_window=True
    )

    # -------------------------------------------------
    # Detect the strongest valid peak
    # -------------------------------------------------

    peak_freq, peak_magnitude, detected = detect_peak(
        freqs,
        magnitude,
        threshold,
        minimum_detection_frequency,
        maximum_detection_frequency
    )

    # -------------------------------------------------
    # Convert detected Doppler frequency to velocity
    # -------------------------------------------------

    if detected:

        velocity_mps = doppler_to_velocity(
            peak_freq,
            carrier_freq
        )

        velocity_kmh = velocity_mps * 3.6

    else:

        velocity_mps = 0.0
        velocity_kmh = 0.0

    # -------------------------------------------------
    # Print results
    # -------------------------------------------------

    print("=" * 50)

    print(f"Requested SNR:       {target_snr_db:.2f} dB")
    print(f"Measured SNR:        {measured_snr_db:.2f} dB")
    print(f"Detection threshold: {threshold:.2f}")

    print("-" * 50)

    if detected:

        print("Target detected:     TRUE")
        print(f"Detected frequency:  {peak_freq:.2f} Hz")
        print(f"Detected magnitude:  {peak_magnitude:.2f}")
        print(f"Radial velocity:     {velocity_mps:.3f} m/s")
        print(f"Radial velocity:     {velocity_kmh:.3f} km/h")

    else:

        print("Target detected:     FALSE")
        print("No valid peak was found above the threshold.")

    print("=" * 50)

    # -------------------------------------------------
    # Plot the time-domain signals
    # -------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        clean_signal,
        label="Clean signal"
    )

    plt.plot(
        time,
        noisy_signal,
        label="Noisy signal",
        alpha=0.7
    )

    plt.plot(
        time,
        filtered_signal,
        label="Filtered signal"
    )

    plt.xlim(0, 0.1)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Radar Signal in the Time Domain")
    plt.legend()
    plt.grid()

    # -------------------------------------------------
    # Plot the FFT spectrum and detection
    # -------------------------------------------------

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

        annotation_text = (
            f"{peak_freq:.2f} Hz\n"
            f"{velocity_mps:.3f} m/s\n"
            f"{velocity_kmh:.3f} km/h"
        )

        plt.annotate(
            annotation_text,
            xy=(peak_freq, peak_magnitude),
            xytext=(
                peak_freq + 20,
                peak_magnitude * 0.8
            ),
            arrowprops={
                "arrowstyle": "->"
            }
        )

    plt.xlim(
        0,
        maximum_detection_frequency
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Automatic Radar Velocity Estimation")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()