#  Detection API for Python Reference Model 

# Author : AJ Donald
# Initial Rev: July 21st, 2026 
# Last Rev: July 21st, 2026 


import numpy as np 

def quadratic_peak_interpolation(
    left_magnitude,
    center_magnitude,
    right_magnitude,
    center_frequency,
    frequency_resolution
):
    """
    Estimate the peak frequency between FFT bins.

    The FFT only reports discrete frequency bins. This interpolation
    uses the peak bin and its two neighbours to estimate a more precise
    frequency.
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


def detect_multiple_peaks(
    frequencies,
    magnitude,
    threshold,
    minimum_frequency,
    maximum_frequency,
    minimum_peak_separation,
    maximum_number_of_targets
):
    """
    Detect multiple local maxima in the FFT magnitude spectrum.

    Returns a list of dictionaries. Each dictionary describes one
    detected target peak.
    """

    candidate_peaks = []

    for index in range(
        1,
        len(magnitude) - 1
    ):

        frequency = frequencies[index]

        if frequency < minimum_frequency:
            continue

        if frequency > maximum_frequency:
            continue

        left_magnitude = magnitude[index - 1]
        center_magnitude = magnitude[index]
        right_magnitude = magnitude[index + 1]

        is_local_maximum = (
            center_magnitude > left_magnitude
            and center_magnitude > right_magnitude
        )

        is_above_threshold = (
            center_magnitude > threshold
        )

        if (
            is_local_maximum
            and is_above_threshold
        ):
            candidate_peaks.append(
                {
                    "index": index,
                    "frequency": frequency,
                    "magnitude": center_magnitude
                }
            )

    candidate_peaks.sort(
        key=lambda peak: peak["magnitude"],
        reverse=True
    )

    selected_peaks = []

    for candidate in candidate_peaks:

        candidate_frequency = (
            candidate["frequency"]
        )

        far_enough_from_existing_peaks = True

        for selected_peak in selected_peaks:

            selected_frequency = (
                selected_peak["frequency"]
            )

            frequency_difference = abs(
                candidate_frequency
                - selected_frequency
            )

            if (
                frequency_difference
                < minimum_peak_separation
            ):
                far_enough_from_existing_peaks = False
                break

        if far_enough_from_existing_peaks:
            selected_peaks.append(
                candidate
            )

        if (
            len(selected_peaks)
            >= maximum_number_of_targets
        ):
            break

    selected_peaks.sort(
        key=lambda peak: peak["frequency"]
    )

    return selected_peaks


def refine_detected_peaks(
    detected_peaks,
    frequencies,
    magnitude,
    frequency_resolution
):
    """
    Apply quadratic interpolation to every detected peak.
    """

    refined_peaks = []

    for peak in detected_peaks:

        peak_index = peak["index"]

        left_magnitude = (
            magnitude[peak_index - 1]
        )

        center_magnitude = (
            magnitude[peak_index]
        )

        right_magnitude = (
            magnitude[peak_index + 1]
        )

        center_frequency = (
            frequencies[peak_index]
        )

        (
            refined_frequency,
            bin_offset
        ) = quadratic_peak_interpolation(
            left_magnitude,
            center_magnitude,
            right_magnitude,
            center_frequency,
            frequency_resolution
        )

        refined_peak = {
            "index": peak_index,
            "fft_bin_frequency": center_frequency,
            "refined_frequency": refined_frequency,
            "magnitude": center_magnitude,
            "bin_offset": bin_offset
        }

        refined_peaks.append(
            refined_peak
        )

    return refined_peaks