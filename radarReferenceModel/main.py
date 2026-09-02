# Main file for Python Reference Model

# Author : AJ Donald
# Initial Rev: July 21st, 2026
# Last Rev: July 21st, 2026

import numpy as np

from signal_generation import (
    generate_multi_target_signal,
    compute_power,
    add_noise_snr
)

from data_input import *
from filtering import *
from spectrum import *
from detection import *
from velocity import *
from visualization import *


def main():

    sample_rate = 4000
    duration = 1.0

    ####################################################
    # Data source settings
    ####################################################

    # Options:
    # "simulation" generates a new simulated radar signal.
    # "csv" loads previously recorded or generated samples.
    input_mode = "csv"

    csv_filename = "data/radar_samples.csv"

    # Set this to True for one simulation run when you want
    # to generate or replace radar_samples.csv.
    #
    # Set it back to False afterward so the CSV is not
    # overwritten every time the model runs.
    save_simulation_to_csv = False

    # CSV loading settings
    csv_column_index = 0
    csv_skip_header = 0

    ####################################################
    # Radar and signal settings
    ####################################################

    # HB100 carrier frequency
    carrier_freq = 10.525e9

    radar_wavelength = calculate_radar_wavelength(
        carrier_freq
    )

    target_frequencies = [
        45.3,
        92.6,
        145.2
    ]

    target_ampltiudes = [
        1.0,
        0.65,
        0.35
    ]

    # Hope for 20 dB SNR
    target_snr_db = 20

    ####################################################
    # Filter settings
    ####################################################

    # Digital filter keeps Doppler components <= 300 Hz
    cutoff_freq = 300
    number_taps = 64

    ####################################################
    # Detector settings
    ####################################################

    detection_threshold = 40

    min_detection_freq = 5
    max_detection_freq = 250

    min_peak_seperate = 5
    max_num_targets = 5

    random_generator = np.random.default_rng(
        42
    )

    ####################################################
    # Generate or load input signal
    ####################################################

    if input_mode == "simulation":

        time, clean_signal = generate_multi_target_signal(
            sample_rate,
            duration,
            target_frequencies,
            target_ampltiudes
        )

        noisy_signal, noise = add_noise_snr(
            clean_signal,
            target_snr_db,
            random_generator
        )

        input_signal = noisy_signal

        if save_simulation_to_csv:

            save_signal_to_csv(
                csv_filename,
                input_signal
            )

            print(
                f"Saved simulated radar samples to: "
                f"{csv_filename}"
            )

    elif input_mode == "csv":

        time, input_signal = load_signal_from_csv(
            csv_filename,
            sample_rate,
            column_index=csv_column_index,
            skip_header=csv_skip_header
        )

        clean_signal = None
        noise = None

        print(
            f"Loaded radar samples from: "
            f"{csv_filename}"
        )

    else:

        raise ValueError(
            f"Unknown input mode: {input_mode}"
        )

    ####################################################
    # Filtering
    ####################################################

    # Process the noisy/input signal:
    # remove DC offset, calculate coefficients and
    # apply the FIR filter.

    signal_without_dc = remove_dc(
        input_signal
    )

    filter_coeffs = design_lpf(
        sample_rate,
        cutoff_freq,
        number_taps
    )

    filtered_signal = apply_fir_filter(
        signal_without_dc,
        filter_coeffs
    )

    ####################################################
    # FFT processing stage
    ####################################################

    windowed_signal = apply_hamming_window(
        filtered_signal
    )

    frequencies, magnitude, freq_resolution = compute_spectrum(
        windowed_signal,
        sample_rate
    )

    print(
        f"The FFT frequency resolution is "
        f"{freq_resolution:.2f} Hz/bin"
    )

    ####################################################
    # Detect peaks after FFT
    ####################################################

    detected_peaks = detect_multiple_peaks(
        frequencies,
        magnitude,
        detection_threshold,
        min_detection_freq,
        max_detection_freq,
        min_peak_seperate,
        max_num_targets
    )

    refined_peaks = refine_detected_peaks(
        detected_peaks,
        frequencies,
        magnitude,
        freq_resolution
    )

    ####################################################
    # Add velocity information to each peak
    ####################################################

    for peak in refined_peaks:

        velocity_mps = doppler_freq_to_velocity(
            peak["refined_frequency"],
            carrier_freq
        )

        velocity_kmh = convert_kmh(
            velocity_mps
        )

        peak["velocity_mps"] = velocity_mps
        peak["velocity_kmh"] = velocity_kmh

    ####################################################
    # Console output
    ####################################################

    print(
        f"Radar wavelength: "
        f"{radar_wavelength:.5f} m"
    )

    print(
        f"Input mode: "
        f"{input_mode}"
    )

    print(
        f"Number of samples: "
        f"{len(input_signal)}"
    )

    if input_mode == "simulation":

        measured_snr_db = (
            10
            * np.log10(
                compute_power(clean_signal)
                / compute_power(noise)
            )
        )

        print(
            f"Requested SNR: "
            f"{target_snr_db} dB"
        )

        print(
            f"Measured SNR: "
            f"{measured_snr_db:.2f} dB"
        )

    else:

        print(
            f"Loaded CSV: "
            f"{csv_filename}"
        )

    ####################################################
    # Print out detection output
    ####################################################

    print(
        f"Number of detected targets: "
        f"{len(refined_peaks)}"
    )

    for target_number, peak in enumerate(
        refined_peaks,
        start=1
    ):

        print(
            f"Target #{target_number}"
        )

        print(
            f"  FFT-bin frequency: "
            f"{peak['fft_bin_frequency']:.2f} Hz"
        )

        print(
            f"  Refined frequency: "
            f"{peak['refined_frequency']:.2f} Hz"
        )

        print(
            f"  Magnitude: "
            f"{peak['magnitude']:.2f}"
        )

        print(
            f"  Bin offset: "
            f"{peak['bin_offset']:.2f}"
        )

        print(
            f"  Velocity: "
            f"{peak['velocity_mps']:.3f} m/s"
        )

        print(
            f"  Velocity: "
            f"{peak['velocity_kmh']:.3f} km/h"
        )

    ####################################################
    # Visualization
    ####################################################

    plot_time_domain_signals(
        time,
        input_signal,
        filtered_signal,
        display_duration=0.1
    )

    plot_doppler_spectrum(
        frequencies,
        magnitude,
        refined_peaks,
        detection_threshold,
        maximum_display_frequency=250
    )

    # CURRENT CHAIN IS:
    #
    # Generate or load input
    # -> remove DC
    # -> FIR filter
    # -> Hamming window
    # -> FFT
    # -> magnitude
    # -> local peak detection
    # -> minimum-distance check
    # -> sub-bin interpolation (quadratic)
    # -> radial velocity calculation


if __name__ == "__main__":

    main()