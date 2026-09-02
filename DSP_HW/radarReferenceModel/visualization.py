import matplotlib.pyplot as plt


def plot_time_domain_signals(
    time,
    noisy_signal,
    filtered_signal,
    display_duration=0.1
):
    """
    Plot the noisy input signal and filtered output signal.

    Parameters:
        time:
            Time values in seconds.

        noisy_signal:
            Signal before filtering.

        filtered_signal:
            Signal after DC removal and FIR filtering.

        display_duration:
            Amount of time to display on the x-axis.
    """

    plt.figure()

    plt.plot(
        time,
        noisy_signal,
        label="Noisy signal",
        alpha=0.6
    )

    plt.plot(
        time,
        filtered_signal,
        label="Filtered signal"
    )

    plt.xlim(
        0,
        display_duration
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Radar Signal Before and After FIR Filtering")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_doppler_spectrum(
    frequencies,
    magnitude,
    detected_peaks,
    detection_threshold,
    maximum_display_frequency=250
):
    """
    Plot the FFT magnitude spectrum and mark detected targets.
    """

    plt.figure()

    plt.plot(
        frequencies,
        magnitude,
        label="FFT magnitude"
    )

    plt.axhline(
        detection_threshold,
        linestyle="--",
        label="Detection threshold"
    )

    for target_number, peak in enumerate(
        detected_peaks,
        start=1
    ):
        plt.scatter(
            peak["fft_bin_frequency"],
            peak["magnitude"],
            label=f"Target {target_number}"
        )

        plt.axvline(
            peak["refined_frequency"],
            linestyle=":"
        )

        plt.annotate(
            (
                f"{peak['refined_frequency']:.2f} Hz\n"
                f"{peak['velocity_mps']:.3f} m/s"
            ),
            xy=(
                peak["fft_bin_frequency"],
                peak["magnitude"]
            ),
            xytext=(
                peak["fft_bin_frequency"] + 5,
                peak["magnitude"] * 1.05
            ),
            arrowprops={
                "arrowstyle": "->"
            }
        )

    plt.xlim(
        0,
        maximum_display_frequency
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Radar Doppler Spectrum")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()