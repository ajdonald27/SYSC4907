# Functions for loading and saving radar sample data

# Author : AJ Donald
# Initial Rev: July 21st, 2026
# Last Rev: July 21st, 2026

import os
import numpy as np


def load_signal_from_csv(
    filename,
    sample_rate,
    column_index=0,
    skip_header=0
):
    """
    Load radar samples from a CSV file.

    Parameters:
        filename:
            Path to the CSV file.

        sample_rate:
            ADC sample rate in samples per second.

        column_index:
            Column containing the radar samples.

        skip_header:
            Number of header rows to skip.

    Returns:
        time:
            Time axis in seconds.

        signal:
            Radar sample values.
    """

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"CSV input file was not found: {filename}"
        )

    csv_data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=skip_header
    )

    # If the CSV contains one column, NumPy returns a 1D array.
    if csv_data.ndim == 1:

        signal = csv_data

    # If the CSV contains multiple columns, select the requested one.
    else:

        if column_index >= csv_data.shape[1]:

            raise ValueError(
                f"Requested CSV column {column_index}, "
                f"but the file only contains "
                f"{csv_data.shape[1]} columns."
            )

        signal = csv_data[:, column_index]

    signal = np.asarray(
        signal,
        dtype=float
    )

    if len(signal) == 0:

        raise ValueError(
            f"The CSV input file contains no samples: "
            f"{filename}"
        )

    number_samples = len(signal)

    time = (
        np.arange(number_samples)
        / sample_rate
    )

    return time, signal


def save_signal_to_csv(
    filename,
    signal
):
    """
    Save a one-dimensional radar signal to a CSV file.

    Parameters:
        filename:
            Output CSV path.

        signal:
            Radar samples to save.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if signal.ndim != 1:

        raise ValueError(
            "save_signal_to_csv expects a one-dimensional signal."
        )

    # Find the folder portion of the filename.
    output_directory = os.path.dirname(
        filename
    )

    # Create the data directory if it does not already exist.
    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    np.savetxt(
        filename,
        signal,
        delimiter=","
    )