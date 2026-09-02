import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin


def generateMultiTargetSignal(
    sample_rate,
    duration,
    target_frequencies,
    target_amplitudes
):
    """
    Generate a signal containing multiple Doppler targets.

    Each target is represented by a sinusoid with its own
    frequency and amplitude.
    """

    if len(target_frequencies) != len(target_amplitudes):
        raise ValueError(
            "target_frequencies and target_amplitudes "
            "must have the same length."
        )

    num_samples = int(sample_rate * duration)

    time = np.arange(num_samples) / sample_rate

    clean_signal = np.zeros(num_samples)

    for frequency, amplitude in zip(
        target_frequencies,
        target_amplitudes
    ):
        clean_signal += (
            amplitude
            * np.sin(
                2 * np.pi * frequency * time
            )
        )

    return time, clean_signal


def computePower(signal):
    """
    Calculate average signal power.
    """

    return np.mean(signal ** 2)


def addNoiseSNR(clean_signal, snr_db):
    """
    Add zero-mean Gaussian noise at the requested SNR.
    """

    signal_power = computePower(clean_signal)

    noise_power = (
        signal_power
        / (10 ** (snr_db / 10))
    )

    noise_standard_deviation = np.sqrt(
        noise_power
    )

    noise = np.random.normal(
        loc=0.0,
        scale=noise_standard_deviation,
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

    fft_output = np.fft.rfft(
        processed_signal
    )

    magnitude = np.abs(
        fft_output
    )

    frequencies = np.fft.rfftfreq(
        num_samples,
        d=1 / sample_rate
    )

    return frequencies, magnitude


def design_lpf(
    sample_rate,
    cutoff_frequency,
    num_taps
):
    """
    Design a low-pass FIR filter.
    """

    coefficients = firwin(
        numtaps=num_taps,
        cutoff=cutoff_frequency,
        fs=sample_rate
    )

    return coefficients


def apply_filter(
    signal,
    coefficients
):
    """
    Apply an FIR filter.
    """

    filtered_signal = lfilter(
        coefficients,
        1.0,
        signal
    )

    return filtered_signal


def quadratic_peak_interpolation(
    left_magnitude,
    center_magnitude,
    right_magnitude,
    center_frequency,
    frequency_resolution
):
    """
    Estimate the sub-bin peak location using three-point
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

    if abs(denominator) < 1e-12:
        return center_frequency, 0.0

    bin_offset = (
        0.5
        * (left_magnitude - right_magnitude)
        / denominator
    )

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
    doppler_frequency,
    carrier_frequency=10.525e9
):
    """
    Convert Doppler frequency to radial velocity.
    """

    speed_of_light = 299_792_458.0

    wavelength = (
        speed_of_light
        / carrier_frequency
    )

    velocity = (
        doppler_frequency
        * wavelength
        / 2
    )

    return velocity


def detect_multiple_peaks(
    frequencies,
    magnitude,
    threshold,
    minimum_frequency=5,
    maximum_frequency=250,
    minimum_peak_separation_hz=5,
    maximum_targets=5
):
    """
    Detect multiple local FFT maxima above a threshold.

    The strongest candidates are selected first while
    enforcing a minimum frequency separation.

    Returns:
        List of dictionaries containing:
            frequency
            magnitude
            index
    """

    valid_region = (
        (frequencies >= minimum_frequency)
        & (frequencies <= maximum_frequency)
    )

    valid_indices = np.where(
        valid_region
    )[0]

    candidate_peaks = []

    for full_index in valid_indices:

        if (
            full_index <= 0
            or full_index >= len(magnitude) - 1
        ):
            continue

        previous_value = magnitude[
            full_index - 1
        ]

        current_value = magnitude[
            full_index
        ]

        next_value = magnitude[
            full_index + 1
        ]

        is_local_maximum = (
            current_value > previous_value
            and current_value > next_value
        )

        is_above_threshold = (
            current_value > threshold
        )

        if (
            is_local_maximum
            and is_above_threshold
        ):
            candidate_peaks.append(
                {
                    "frequency": frequencies[
                        full_index
                    ],
                    "magnitude": current_value,
                    "index": full_index
                }
            )

    # Process strongest candidates first
    candidate_peaks.sort(
        key=lambda peak: peak["magnitude"],
        reverse=True
    )

    selected_peaks = []

    for candidate in candidate_peaks:

        separated_from_existing_peaks = all(
            abs(
                candidate["frequency"]
                - selected["frequency"]
            )
            >= minimum_peak_separation_hz
            for selected in selected_peaks
        )

        if separated_from_existing_peaks:
            selected_peaks.append(
                candidate
            )

        if len(selected_peaks) >= maximum_targets:
            break

    # Sort final output from lowest frequency to highest
    selected_peaks.sort(
        key=lambda peak: peak["frequency"]
    )

    return selected_peaks


def refine_detected_peaks(
    detected_peaks,
    frequencies,
    magnitude,
    carrier_frequency
):
    """
    Apply sub-bin interpolation and velocity conversion
    to every detected peak.
    """

    refined_targets = []

    frequency_resolution = (
        frequencies[1]
        - frequencies[0]
    )

    for peak in detected_peaks:

        peak_index = peak["index"]

        fft_bin_frequency = peak["frequency"]
        peak_magnitude = peak["magnitude"]

        refined_frequency = (
            fft_bin_frequency
        )

        bin_offset = 0.0

        if (
            0 < peak_index
            < len(magnitude) - 1
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
                refined_frequency,
                bin_offset
            ) = quadratic_peak_interpolation(
                left_magnitude,
                center_magnitude,
                right_magnitude,
                fft_bin_frequency,
                frequency_resolution
            )

        velocity_mps = doppler_to_velocity(
            refined_frequency,
            carrier_frequency
        )

        velocity_kmh = velocity_mps * 3.6

        refined_targets.append(
            {
                "fft_bin_frequency": (
                    fft_bin_frequency
                ),
                "refined_frequency": (
                    refined_frequency
                ),
                "magnitude": peak_magnitude,
                "index": peak_index,
                "bin_offset": bin_offset,
                "velocity_mps": velocity_mps,
                "velocity_kmh": velocity_kmh
            }
        )

    return refined_targets


def match_true_frequency(
    refined_frequency,
    target_frequencies
):
    """
    Find the closest simulated true target frequency.

    This is used only to evaluate the simulation.
    A real radar would not know the true frequency.
    """

    closest_frequency = min(
        target_frequencies,
        key=lambda target_frequency: abs(
            target_frequency
            - refined_frequency
        )
    )

    error = abs(
        closest_frequency
        - refined_frequency
    )

    return closest_frequency, error


def main():

    # Repeatable noise while testing
    np.random.seed(42)

    # -------------------------------------------------
    # Simulation configuration
    # -------------------------------------------------

    sample_rate = 4000
    duration = 1.0

    target_frequencies = [
        45.3,
        92.6,
        145.2
    ]

    target_amplitudes = [
        1.0,
        0.65,
        0.35
    ]

    target_snr_db = 20

    carrier_frequency = 10.525e9

    cutoff_frequency = 325
    num_taps = 64

    threshold = 40.0

    minimum_detection_frequency = 5
    maximum_detection_frequency = 250

    minimum_peak_separation_hz = 5
    maximum_targets = 5

    # -------------------------------------------------
    # Generate multiple targets
    # -------------------------------------------------

    time, clean_signal = generateMultiTargetSignal(
        sample_rate,
        duration,
        target_frequencies,
        target_amplitudes
    )

    # -------------------------------------------------
    # Add Gaussian noise
    # -------------------------------------------------

    noisy_signal, noise = addNoiseSNR(
        clean_signal,
        target_snr_db
    )

    signal_power = computePower(
        clean_signal
    )

    noise_power = computePower(
        noise
    )

    measured_snr_db = 10 * np.log10(
        signal_power / noise_power
    )

    # -------------------------------------------------
    # FIR filtering
    # -------------------------------------------------

    filter_coefficients = design_lpf(
        sample_rate,
        cutoff_frequency,
        num_taps
    )

    filtered_signal = apply_filter(
        noisy_signal,
        filter_coefficients
    )

    # -------------------------------------------------
    # FFT
    # -------------------------------------------------

    frequencies, magnitude = computeSpectrum(
        filtered_signal,
        sample_rate,
        apply_window=True
    )

    frequency_resolution = (
        frequencies[1]
        - frequencies[0]
    )

    # -------------------------------------------------
    # Multi-target peak detection
    # -------------------------------------------------

    detected_peaks = detect_multiple_peaks(
        frequencies,
        magnitude,
        threshold,
        minimum_detection_frequency,
        maximum_detection_frequency,
        minimum_peak_separation_hz,
        maximum_targets
    )

    # -------------------------------------------------
    # Interpolation and velocity conversion
    # -------------------------------------------------

    refined_targets = refine_detected_peaks(
        detected_peaks,
        frequencies,
        magnitude,
        carrier_frequency
    )

    # -------------------------------------------------
    # Console output
    # -------------------------------------------------

    print("=" * 68)
    print("MULTI-TARGET DOPPLER RADAR DETECTION")
    print("=" * 68)

    print(
        f"Requested SNR:              "
        f"{target_snr_db:.2f} dB"
    )

    print(
        f"Measured SNR:               "
        f"{measured_snr_db:.2f} dB"
    )

    print(
        f"FFT resolution:             "
        f"{frequency_resolution:.3f} Hz"
    )

    print(
        f"Detection threshold:        "
        f"{threshold:.2f}"
    )

    print(
        f"Minimum peak separation:    "
        f"{minimum_peak_separation_hz:.2f} Hz"
    )

    print(
        f"Number of true targets:     "
        f"{len(target_frequencies)}"
    )

    print(
        f"Number of detected targets: "
        f"{len(refined_targets)}"
    )

    print("-" * 68)

    if len(refined_targets) == 0:

        print(
            "No targets were detected above "
            "the threshold."
        )

    else:

        for target_number, target in enumerate(
            refined_targets,
            start=1
        ):
            (
                closest_true_frequency,
                frequency_error
            ) = match_true_frequency(
                target["refined_frequency"],
                target_frequencies
            )

            print(f"Target {target_number}")
            print(
                f"  Closest true frequency: "
                f"{closest_true_frequency:.3f} Hz"
            )

            print(
                f"  FFT-bin frequency:      "
                f"{target['fft_bin_frequency']:.3f} Hz"
            )

            print(
                f"  Refined frequency:      "
                f"{target['refined_frequency']:.3f} Hz"
            )

            print(
                f"  Frequency error:        "
                f"{frequency_error:.3f} Hz"
            )

            print(
                f"  Bin offset:             "
                f"{target['bin_offset']:.4f} bins"
            )

            print(
                f"  Peak magnitude:         "
                f"{target['magnitude']:.2f}"
            )

            print(
                f"  Radial velocity:        "
                f"{target['velocity_mps']:.4f} m/s"
            )

            print(
                f"  Radial velocity:        "
                f"{target['velocity_kmh']:.4f} km/h"
            )

            print("-" * 68)

    print("=" * 68)

    # -------------------------------------------------
    # Time-domain plot
    # -------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        clean_signal,
        label="Clean multi-target signal"
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
    plt.title("Multi-Target Radar Signal")
    plt.legend()
    plt.grid()

    # -------------------------------------------------
    # FFT plot
    # -------------------------------------------------

    plt.figure()

    plt.plot(
        frequencies,
        magnitude,
        label="Filtered FFT magnitude"
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label="Detection threshold"
    )

    for target_number, target in enumerate(
        refined_targets,
        start=1
    ):
        fft_bin_frequency = target[
            "fft_bin_frequency"
        ]

        refined_frequency = target[
            "refined_frequency"
        ]

        peak_magnitude = target[
            "magnitude"
        ]

        plt.scatter(
            fft_bin_frequency,
            peak_magnitude,
            label=f"Detected target {target_number}"
        )

        plt.axvline(
            refined_frequency,
            linestyle=":"
        )

        annotation_text = (
            f"Target {target_number}\n"
            f"{refined_frequency:.2f} Hz\n"
            f"{target['velocity_mps']:.3f} m/s"
        )

        plt.annotate(
            annotation_text,
            xy=(
                fft_bin_frequency,
                peak_magnitude
            ),
            xytext=(
                fft_bin_frequency + 6,
                peak_magnitude * 1.05
            ),
            arrowprops={
                "arrowstyle": "->"
            }
        )

    # True frequencies are shown only because this
    # is a controlled simulation.
    for index, true_frequency in enumerate(
        target_frequencies
    ):
        label = (
            "True target frequencies"
            if index == 0
            else None
        )

        plt.axvline(
            true_frequency,
            linestyle="-.",
            label=label
        )

    plt.xlim(
        0,
        maximum_detection_frequency
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(
        "Multi-Target Doppler Detection "
        "and Velocity Estimation"
    )

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()