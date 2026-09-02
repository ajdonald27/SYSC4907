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
    Add Gaussian noise at the requested SNR.
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
    Compute the one-sided FFT magnitude spectrum.
    """

    num_samples = len(signal)

    if apply_window:
        window = np.hamming(num_samples)
        processed_signal = signal * window
    else:
        processed_signal = signal

    fft_output = np.fft.rfft(processed_signal)

    magnitude = np.abs(fft_output)

    freqs = np.fft.rfftfreq(
        num_samples,
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
    Find the strongest local FFT peak above a threshold.

    Returns:
        best_peak_frequency
        best_peak_magnitude
        best_peak_index
        detected
    """

    valid_region = (
        (freqs >= min_frequency)
        & (freqs <= max_frequency)
    )

    # Indices in the original FFT arrays
    valid_indices = np.where(valid_region)[0]

    valid_freqs = freqs[valid_indices]
    valid_magnitude = magnitude[valid_indices]

    best_peak_frequency = 0.0
    best_peak_magnitude = 0.0
    best_peak_index = -1
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

                # Save the position in the full FFT array
                best_peak_index = valid_indices[i]

                detected = True

    return (
        best_peak_frequency,
        best_peak_magnitude,
        best_peak_index,
        detected
    )


def quadratic_peak_interpolation(
    left_magnitude,
    center_magnitude,
    right_magnitude,
    center_frequency,
    frequency_resolution
):
    """
    Estimate the peak frequency between FFT bins using
    quadratic interpolation.

    Returns:
        refined_frequency
        bin_offset
    """

    denominator = (
        left_magnitude
        - 2 * center_magnitude
        + right_magnitude
    )

    # Avoid division by zero
    if abs(denominator) < 1e-12:
        return center_frequency, 0.0

    bin_offset = (
        0.5
        * (left_magnitude - right_magnitude)
        / denominator
    )

    # A valid three-point parabolic peak should normally
    # fall within half a bin of the center.
    bin_offset = np.clip(
        bin_offset,
        -0.5,
        0.5
    )

    refined_frequency = (
        center_frequency
        + bin_offset * frequency_resolution
    )

    return refined_frequency, bin_offset


def doppler_to_velocity(
    doppler_freq,
    carrier_freq=10.525e9
):
    """
    Convert Doppler frequency into radial velocity.
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

    # Fixed seed makes the noise repeatable while debugging
    np.random.seed(42)

    # -------------------------------------------------
    # Simulation configuration
    # -------------------------------------------------

    sample_rate = 4000
    duration = 1.0

    # Use a non-integer frequency to test interpolation
    target_freq = 72.35

    target_amp = 1.0

    # Begin at high SNR so interpolation is easy to observe
    target_snr_db = 30

    carrier_freq = 10.525e9

    cutoff_freq = 325
    num_taps = 64

    threshold = 40.0

    minimum_detection_frequency = 5
    maximum_detection_frequency = 250

    # -------------------------------------------------
    # Generate clean target signal
    # -------------------------------------------------

    time, clean_signal = generateSignal(
        sample_rate,
        duration,
        target_freq,
        target_amp
    )

    # -------------------------------------------------
    # Add noise
    # -------------------------------------------------

    noisy_signal, noise = addNoiseSNR(
        clean_signal,
        target_snr_db
    )

    # -------------------------------------------------
    # Measure generated SNR
    # -------------------------------------------------

    signal_power = computePower(clean_signal)
    noise_power = computePower(noise)

    measured_snr_db = 10 * np.log10(
        signal_power / noise_power
    )

    # -------------------------------------------------
    # Design and apply FIR filter
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
    # Compute FFT spectrum
    # -------------------------------------------------

    freqs, magnitude = computeSpectrum(
        filtered_signal,
        sample_rate,
        apply_window=True
    )

    frequency_resolution = (
        freqs[1] - freqs[0]
    )

    # -------------------------------------------------
    # Detect strongest valid FFT peak
    # -------------------------------------------------

    (
        peak_freq,
        peak_magnitude,
        peak_index,
        detected
    ) = detect_peak(
        freqs,
        magnitude,
        threshold,
        minimum_detection_frequency,
        maximum_detection_frequency
    )

    # Default refined estimate is the original FFT bin
    refined_peak_freq = peak_freq
    bin_offset = 0.0

    left_magnitude = 0.0
    center_magnitude = 0.0
    right_magnitude = 0.0

    # -------------------------------------------------
    # Perform sub-bin interpolation
    # -------------------------------------------------

    if (
        detected
        and 0 < peak_index < len(magnitude) - 1
    ):

        left_magnitude = magnitude[
            peak_index - 1
        ]

        center_magnitude = magnitude[
            peak_index
        ]

        right_magnitude = magnitude[
            peak_index + 1
        ]

        (
            refined_peak_freq,
            bin_offset
        ) = quadratic_peak_interpolation(
            left_magnitude,
            center_magnitude,
            right_magnitude,
            peak_freq,
            frequency_resolution
        )

    # -------------------------------------------------
    # Calculate velocity and frequency errors
    # -------------------------------------------------

    if detected:

        bin_velocity_mps = doppler_to_velocity(
            peak_freq,
            carrier_freq
        )

        refined_velocity_mps = doppler_to_velocity(
            refined_peak_freq,
            carrier_freq
        )

        refined_velocity_kmh = (
            refined_velocity_mps * 3.6
        )

        bin_frequency_error = abs(
            peak_freq - target_freq
        )

        refined_frequency_error = abs(
            refined_peak_freq - target_freq
        )

    else:

        bin_velocity_mps = 0.0
        refined_velocity_mps = 0.0
        refined_velocity_kmh = 0.0

        bin_frequency_error = 0.0
        refined_frequency_error = 0.0

    # -------------------------------------------------
    # Print results
    # -------------------------------------------------

    print("=" * 58)

    print(f"Requested SNR:          {target_snr_db:.2f} dB")
    print(f"Measured SNR:           {measured_snr_db:.2f} dB")
    print(f"FFT resolution:         {frequency_resolution:.3f} Hz")
    print(f"Detection threshold:    {threshold:.2f}")

    print("-" * 58)

    if detected:

        print("Target detected:        TRUE")
        print()
        print(f"True frequency:         {target_freq:.3f} Hz")
        print(f"FFT-bin frequency:      {peak_freq:.3f} Hz")
        print(f"Refined frequency:      {refined_peak_freq:.3f} Hz")
        print()
        print(f"Bin offset:             {bin_offset:.4f} bins")
        print()
        print(f"FFT-bin error:          {bin_frequency_error:.3f} Hz")
        print(f"Refined error:          {refined_frequency_error:.3f} Hz")
        print()
        print(f"Peak magnitude:         {peak_magnitude:.2f}")
        print(f"Left magnitude:         {left_magnitude:.2f}")
        print(f"Center magnitude:       {center_magnitude:.2f}")
        print(f"Right magnitude:        {right_magnitude:.2f}")
        print()
        print(f"FFT-bin velocity:       {bin_velocity_mps:.4f} m/s")
        print(f"Refined velocity:       {refined_velocity_mps:.4f} m/s")
        print(f"Refined velocity:       {refined_velocity_kmh:.4f} km/h")

    else:

        print("Target detected:        FALSE")
        print("No valid peak was found above the threshold.")

    print("=" * 58)

    # -------------------------------------------------
    # Plot time-domain signals
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
    # Plot FFT spectrum
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

        # Discrete FFT point
        plt.scatter(
            peak_freq,
            peak_magnitude,
            label="Detected FFT bin"
        )

        # Refined sub-bin estimate
        plt.axvline(
            refined_peak_freq,
            linestyle=":",
            label="Refined frequency"
        )

        # Known simulated target frequency
        plt.axvline(
            target_freq,
            linestyle="-.",
            label="True frequency"
        )

        annotation_text = (
            f"FFT bin: {peak_freq:.2f} Hz\n"
            f"Refined: {refined_peak_freq:.3f} Hz\n"
            f"True: {target_freq:.3f} Hz\n"
            f"Velocity: {refined_velocity_mps:.3f} m/s"
        )

        plt.annotate(
            annotation_text,
            xy=(
                peak_freq,
                peak_magnitude
            ),
            xytext=(
                peak_freq + 15,
                peak_magnitude * 0.75
            ),
            arrowprops={
                "arrowstyle": "->"
            }
        )

    plt.xlim(
        60,
        85
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Sub-bin Doppler Frequency Estimation")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()