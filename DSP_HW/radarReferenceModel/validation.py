# Validation testing for Python Radar Reference Model

# Author : AJ Donald
# Initial Rev: July 21st, 2026
# Last Rev: July 21st, 2026

import os
import numpy as np

from signal_generation import (
    generate_multi_target_signal,
    add_noise_snr
)

from data_input import (
    save_signal_to_csv,
    load_signal_from_csv
)

from filtering import *
from spectrum import *
from detection import *
from velocity import *


####################################################
# Shared validation settings
####################################################

SAMPLE_RATE = 4000
DURATION = 1.0

# HB100 carrier frequency
CARRIER_FREQ = 10.525e9

# Digital filter settings
CUTOFF_FREQ = 300
NUMBER_TAPS = 64

# Detector settings
DETECTION_THRESHOLD = 40

MIN_DETECTION_FREQ = 5
MAX_DETECTION_FREQ = 250

MIN_PEAK_SEPERATE = 5
MAX_NUM_TARGETS = 5

# Maximum acceptable error between the expected
# and detected Doppler frequency.
FREQUENCY_TOLERANCE_HZ = 0.75


####################################################
# Golden-model processing function
####################################################

def process_radar_signal(
    input_signal,
    sample_rate=SAMPLE_RATE,
    detection_threshold=DETECTION_THRESHOLD,
    min_peak_seperate=MIN_PEAK_SEPERATE
):
    """
    Run an input signal through the complete radar
    golden-reference processing chain.

    Processing chain:

        Input signal
        -> remove DC
        -> FIR filter
        -> Hamming window
        -> FFT
        -> magnitude
        -> peak detection
        -> quadratic interpolation
        -> radial velocity calculation

    Returns:
        refined_peaks:
            List of detected targets.

        frequencies:
            FFT frequency axis.

        magnitude:
            FFT magnitude spectrum.
    """

    ####################################################
    # Filtering
    ####################################################

    signal_without_dc = remove_dc(
        input_signal
    )

    filter_coeffs = design_lpf(
        sample_rate,
        CUTOFF_FREQ,
        NUMBER_TAPS
    )

    filtered_signal = apply_fir_filter(
        signal_without_dc,
        filter_coeffs
    )

    ####################################################
    # FFT processing
    ####################################################

    windowed_signal = apply_hamming_window(
        filtered_signal
    )

    frequencies, magnitude, freq_resolution = compute_spectrum(
        windowed_signal,
        sample_rate
    )

    ####################################################
    # Peak detection
    ####################################################

    detected_peaks = detect_multiple_peaks(
        frequencies,
        magnitude,
        detection_threshold,
        MIN_DETECTION_FREQ,
        MAX_DETECTION_FREQ,
        min_peak_seperate,
        MAX_NUM_TARGETS
    )

    refined_peaks = refine_detected_peaks(
        detected_peaks,
        frequencies,
        magnitude,
        freq_resolution
    )

    ####################################################
    # Velocity calculation
    ####################################################

    for peak in refined_peaks:

        velocity_mps = doppler_freq_to_velocity(
            peak["refined_frequency"],
            CARRIER_FREQ
        )

        velocity_kmh = convert_kmh(
            velocity_mps
        )

        peak["velocity_mps"] = velocity_mps
        peak["velocity_kmh"] = velocity_kmh

    return refined_peaks, frequencies, magnitude


####################################################
# Validation helper functions
####################################################

def frequencies_match(
    expected_frequencies,
    detected_peaks,
    tolerance_hz=FREQUENCY_TOLERANCE_HZ
):
    """
    Check whether every expected target frequency has
    a matching detected frequency within the tolerance.
    """

    detected_frequencies = [
        peak["refined_frequency"]
        for peak in detected_peaks
    ]

    if len(expected_frequencies) != len(detected_frequencies):

        return False

    expected_frequencies = sorted(
        expected_frequencies
    )

    detected_frequencies = sorted(
        detected_frequencies
    )

    for expected, detected in zip(
        expected_frequencies,
        detected_frequencies
    ):

        frequency_error = abs(
            expected - detected
        )

        if frequency_error > tolerance_hz:

            return False

    return True


def print_detected_targets(
    detected_peaks
):
    """
    Print all detected targets for a validation test.
    """

    if len(detected_peaks) == 0:

        print("  No targets detected.")
        return

    for target_number, peak in enumerate(
        detected_peaks,
        start=1
    ):

        print(
            f"  Target #{target_number}: "
            f"{peak['refined_frequency']:.3f} Hz, "
            f"{peak['velocity_mps']:.3f} m/s"
        )


def print_test_result(
    test_name,
    passed
):
    """
    Print a clear PASS or FAIL message.
    """

    if passed:

        print(
            f"[PASS] {test_name}"
        )

    else:

        print(
            f"[FAIL] {test_name}"
        )

    print()


####################################################
# Test 1: Single target
####################################################

def test_single_target():
    """
    Verify that one simulated target is detected at
    the correct Doppler frequency.
    """

    test_name = "Single-target detection"

    target_frequencies = [
        75.4
    ]

    target_amplitudes = [
        1.0
    ]

    time, clean_signal = generate_multi_target_signal(
        SAMPLE_RATE,
        DURATION,
        target_frequencies,
        target_amplitudes
    )

    random_generator = np.random.default_rng(
        101
    )

    input_signal, noise = add_noise_snr(
        clean_signal,
        20,
        random_generator
    )

    detected_peaks, frequencies, magnitude = process_radar_signal(
        input_signal
    )

    print(
        f"Test: {test_name}"
    )

    print(
        f"  Expected frequencies: "
        f"{target_frequencies}"
    )

    print_detected_targets(
        detected_peaks
    )

    passed = frequencies_match(
        target_frequencies,
        detected_peaks
    )

    print_test_result(
        test_name,
        passed
    )

    return passed


####################################################
# Test 2: Multiple targets
####################################################

def test_multiple_targets():
    """
    Verify that the standard three-target signal is
    detected correctly.
    """

    test_name = "Multiple-target detection"

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

    time, clean_signal = generate_multi_target_signal(
        SAMPLE_RATE,
        DURATION,
        target_frequencies,
        target_amplitudes
    )

    random_generator = np.random.default_rng(
        42
    )

    input_signal, noise = add_noise_snr(
        clean_signal,
        20,
        random_generator
    )

    detected_peaks, frequencies, magnitude = process_radar_signal(
        input_signal
    )

    print(
        f"Test: {test_name}"
    )

    print(
        f"  Expected frequencies: "
        f"{target_frequencies}"
    )

    print_detected_targets(
        detected_peaks
    )

    passed = frequencies_match(
        target_frequencies,
        detected_peaks
    )

    print_test_result(
        test_name,
        passed
    )

    return passed


####################################################
# Test 3: No target
####################################################

def test_no_target():
    """
    Verify that a zero-valued input does not produce
    any false target detections.
    """

    test_name = "No-target detection"

    number_samples = int(
        SAMPLE_RATE * DURATION
    )

    input_signal = np.zeros(
        number_samples
    )

    detected_peaks, frequencies, magnitude = process_radar_signal(
        input_signal
    )

    print(
        f"Test: {test_name}"
    )

    print(
        "  Expected number of targets: 0"
    )

    print_detected_targets(
        detected_peaks
    )

    passed = (
        len(detected_peaks) == 0
    )

    print_test_result(
        test_name,
        passed
    )

    return passed


####################################################
# Test 4: Low SNR
####################################################

def test_low_snr():
    """
    Verify that the detector can still detect strong
    targets when the signal contains substantial noise.
    """

    test_name = "Low-SNR detection"

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

    time, clean_signal = generate_multi_target_signal(
        SAMPLE_RATE,
        DURATION,
        target_frequencies,
        target_amplitudes
    )

    random_generator = np.random.default_rng(
        202
    )

    # Lower SNR than the normal model.
    input_signal, noise = add_noise_snr(
        clean_signal,
        5,
        random_generator
    )

    detected_peaks, frequencies, magnitude = process_radar_signal(
        input_signal
    )

    print(
        f"Test: {test_name}"
    )

    print(
        "  Input SNR: 5 dB"
    )

    print(
        f"  Expected frequencies: "
        f"{target_frequencies}"
    )

    print_detected_targets(
        detected_peaks
    )

    passed = frequencies_match(
        target_frequencies,
        detected_peaks,
        tolerance_hz=1.1
    )

    print_test_result(
        test_name,
        passed
    )

    return passed


####################################################
# Test 5: Closely spaced targets
####################################################

def test_close_targets():
    """
    Verify that two nearby targets can be detected when
    the minimum peak-separation setting is reduced.

    The normal minimum separation is 5 Hz. This test uses
    targets separated by 4 Hz, so the detector setting is
    reduced to 2 Hz for this controlled case.
    """

    test_name = "Closely spaced target detection"

    target_frequencies = [
        90.0,
        94.0
    ]

    target_amplitudes = [
        1.0,
        0.8
    ]

    time, clean_signal = generate_multi_target_signal(
        SAMPLE_RATE,
        DURATION,
        target_frequencies,
        target_amplitudes
    )

    random_generator = np.random.default_rng(
        303
    )

    input_signal, noise = add_noise_snr(
        clean_signal,
        25,
        random_generator
    )

    detected_peaks, frequencies, magnitude = process_radar_signal(
        input_signal,
        min_peak_seperate=2
    )

    print(
        f"Test: {test_name}"
    )

    print(
        f"  Expected frequencies: "
        f"{target_frequencies}"
    )

    print_detected_targets(
        detected_peaks
    )

    passed = frequencies_match(
        target_frequencies,
        detected_peaks,
        tolerance_hz=0.75
    )

    print_test_result(
        test_name,
        passed
    )

    return passed


####################################################
# Test 6: Target below detection threshold
####################################################

def test_below_threshold():
    """
    Verify that a very weak target below the configured
    FFT-magnitude threshold is not detected.
    """

    test_name = "Below-threshold rejection"

    target_frequencies = [
        120.0
    ]

    # This amplitude is intentionally very small.
    target_amplitudes = [
        0.01
    ]

    time, input_signal = generate_multi_target_signal(
        SAMPLE_RATE,
        DURATION,
        target_frequencies,
        target_amplitudes
    )

    detected_peaks, frequencies, magnitude = process_radar_signal(
        input_signal
    )

    print(
        f"Test: {test_name}"
    )

    print(
        "  Expected number of detected targets: 0"
    )

    print_detected_targets(
        detected_peaks
    )

    passed = (
        len(detected_peaks) == 0
    )

    print_test_result(
        test_name,
        passed
    )

    return passed


####################################################
# Test 7: CSV save/load round trip
####################################################

def test_csv_round_trip():
    """
    Verify that saving and reloading radar samples does
    not alter the input signal.
    """

    test_name = "CSV save/load round trip"

    target_frequencies = [
        60.5
    ]

    target_amplitudes = [
        1.0
    ]

    time, original_signal = generate_multi_target_signal(
        SAMPLE_RATE,
        DURATION,
        target_frequencies,
        target_amplitudes
    )

    test_filename = (
        "data/validation_test_signal.csv"
    )

    save_signal_to_csv(
        test_filename,
        original_signal
    )

    loaded_time, loaded_signal = load_signal_from_csv(
        test_filename,
        SAMPLE_RATE
    )

    samples_match = np.allclose(
        original_signal,
        loaded_signal,
        rtol=1e-12,
        atol=1e-12
    )

    time_matches = np.allclose(
        time,
        loaded_time,
        rtol=1e-12,
        atol=1e-12
    )

    passed = (
        samples_match
        and time_matches
    )

    print(
        f"Test: {test_name}"
    )

    print(
        f"  Original sample count: "
        f"{len(original_signal)}"
    )

    print(
        f"  Loaded sample count: "
        f"{len(loaded_signal)}"
    )

    print(
        f"  Samples match: "
        f"{samples_match}"
    )

    print(
        f"  Time axes match: "
        f"{time_matches}"
    )

    print_test_result(
        test_name,
        passed
    )

    # Remove the temporary validation file after the test.
    if os.path.exists(
        test_filename
    ):

        os.remove(
            test_filename
        )

    return passed


####################################################
# Run all validation tests
####################################################

def main():

    print()
    print(
        "=============================================="
    )
    print(
        "Radar Golden Reference Model Validation"
    )
    print(
        "=============================================="
    )
    print()

    test_results = []

    test_results.append(
        test_single_target()
    )

    test_results.append(
        test_multiple_targets()
    )

    test_results.append(
        test_no_target()
    )

    test_results.append(
        test_low_snr()
    )

    test_results.append(
        test_close_targets()
    )

    test_results.append(
        test_below_threshold()
    )

    test_results.append(
        test_csv_round_trip()
    )

    number_passed = sum(
        test_results
    )

    number_tests = len(
        test_results
    )

    number_failed = (
        number_tests
        - number_passed
    )

    print(
        "=============================================="
    )
    print(
        "Validation Summary"
    )
    print(
        "=============================================="
    )

    print(
        f"Tests passed: "
        f"{number_passed}/{number_tests}"
    )

    print(
        f"Tests failed: "
        f"{number_failed}/{number_tests}"
    )

    if number_failed == 0:

        print()
        print(
            "RESULT: GOLDEN REFERENCE MODEL VALIDATION PASSED"
        )

    else:

        print()
        print(
            "RESULT: GOLDEN REFERENCE MODEL VALIDATION FAILED"
        )

        print(
            "Review the failed test output before proceeding."
        )

    print()


if __name__ == "__main__":

    main()