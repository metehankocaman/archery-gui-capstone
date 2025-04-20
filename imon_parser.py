# imon_parser.py (Updated)
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import datetime


def read_imon_file(filepath, max_datapoints=None, channel=0):
    """
    Reads an iMON data file (TXT) and returns time and data arrays.

    Args:
        filepath (str): Path to the iMON data file
        max_datapoints (int, optional): Maximum number of datapoints to return (most recent)
        channel (int, optional): Which strain channel to use (default 0)

    Returns:
        tuple: (timestamps, values_array, raw_datetimes) or (None, None, None) on error
             timestamps are relative seconds, values_array is the strain data,
             raw_datetimes are datetime objects
    """
    try:
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return None, None, None

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find the header row (contains "Date" and "Time")
        header_row = -1
        for i, line in enumerate(lines):
            if "Date" in line and "Time" in line:
                header_row = i
                break

        if header_row == -1:
            logging.warning("Could not find header row in iMON file")
            return None, None, None

        # Extract data rows (skip header)
        raw_datetimes = []
        relative_timestamps = []
        values_list = []

        for i in range(header_row + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 3:  # Need date, time, and at least one value
                continue

            # Parse timestamp
            try:
                date_str = parts[0]
                time_str = parts[1]
                timestamp = datetime.datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %I:%M:%S %p")

                # Add to datetime list
                raw_datetimes.append(timestamp)

                # Convert to seconds since first entry
                if not relative_timestamps:
                    first_timestamp = timestamp
                    relative_timestamps.append(0.0)
                else:
                    delta = (timestamp - first_timestamp).total_seconds()
                    relative_timestamps.append(delta)

                # Extract strain value for the specified channel
                channel_idx = min(channel, len(parts) - 3)  # Ensure in range
                channel_value = float(parts[channel_idx + 2])  # +2 to skip date/time
                values_list.append(channel_value)

            except Exception as e:
                logging.warning(f"Error parsing line {i}: {e}")

        if not relative_timestamps:
            logging.warning("No valid data rows found in iMON file")
            return None, None, None

        # Limit to most recent data points if requested
        if max_datapoints and len(relative_timestamps) > max_datapoints:
            start_idx = len(relative_timestamps) - max_datapoints
            relative_timestamps = relative_timestamps[start_idx:]
            values_list = values_list[start_idx:]
            raw_datetimes = raw_datetimes[start_idx:]

        time_array = np.array(relative_timestamps)
        values_array = np.array(values_list)

        return time_array, values_array, raw_datetimes

    except Exception as e:
        logging.error(f"Error reading iMON file {filepath}: {e}")
        return None, None, None


def get_calibration_coefficients(filepath):
    """
    Extract calibration coefficients from iMON file header

    Args:
        filepath (str): Path to the iMON data file

    Returns:
        dict: Dictionary of calibration coefficients or None on error
    """
    try:
        with open(filepath, 'r') as f:
            # Read first few lines to find coefficients
            for _ in range(10):  # Check first 10 lines
                line = f.readline().strip()
                # Check if this is the calibration line (contains scientific notation)
                if "E+" in line or "E-" in line:
                    parts = line.split('\t')
                    if len(parts) >= 6:
                        # Extract coefficients (A1, B1-B5)
                        coeffs = {}
                        labels = ['A1', 'B1', 'B2', 'B3', 'B4', 'B5']

                        for i, label in enumerate(labels):
                            if i < len(parts):
                                try:
                                    coeffs[label] = float(parts[i])
                                except ValueError:
                                    coeffs[label] = None

                        return coeffs

        logging.warning("Could not find calibration coefficients in iMON file")
        return None

    except Exception as e:
        logging.error(f"Error reading calibration data: {e}")
        return None


def plot_imon_data(time, values, title="iMON Strain Data"):
    """
    Creates a matplotlib figure from the provided time and values data.

    Args:
        time (numpy.ndarray): Array of timestamps
        values (numpy.ndarray): Array of strain values
        title (str, optional): Plot title

    Returns:
        matplotlib.figure.Figure: Figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the strain data
    ax.plot(time, values, label="Strain")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    return fig