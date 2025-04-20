# imon_file_monitor.py (Updated for Spectral Data)
from PyQt5.QtCore import QThread, pyqtSignal
import time
import os
import logging
import numpy as np
import datetime
from imon_spectral import IMONSpectralAnalyzer


class IMONFileMonitor(QThread):
    """
    A thread that monitors an I-MON data file for changes and processes the full
    spectral data to extract wavelength shifts and calculate strain.
    """
    newData = pyqtSignal(object)  # Signal emitting a dict with processed spectral data

    def __init__(self, file_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._running = True
        self.last_modified_time = 0
        self.last_file_size = 0
        self.last_line_count = 0

        # Initialize spectral analyzer
        self.analyzer = IMONSpectralAnalyzer()
        self.reference_set = False

    def set_file_path(self, file_path):
        """Set or update the file path to monitor"""
        self.file_path = file_path
        # Reset tracking variables when file path changes
        self.last_modified_time = 0
        self.last_file_size = 0
        self.last_line_count = 0

        # Load calibration from the file
        if file_path and os.path.exists(file_path):
            self.analyzer.load_calibration(file_path)
            self.reference_set = False  # Need to set a new reference

        logging.info(f"Set I-MON file path to: {file_path}")

    def set_strain_conversion_factor(self, factor):
        """Set the strain conversion factor used for calculations"""
        self.analyzer.set_strain_conversion_factor(factor)

    def reset_reference(self):
        """Reset the reference spectrum"""
        self.analyzer.reset_reference()
        self.reference_set = False
        logging.info("Reset spectral reference")

    def run(self):
        """Monitor the file for changes and process new spectral data when detected"""
        logging.info("I-MON file monitor started")
        while self._running:
            if not self.file_path or not os.path.exists(self.file_path):
                time.sleep(0.5)  # Sleep longer if file doesn't exist
                continue

            try:
                # Check if file has been modified
                current_mtime = os.path.getmtime(self.file_path)
                current_size = os.path.getsize(self.file_path)

                if (current_mtime > self.last_modified_time or
                        current_size != self.last_file_size):
                    # File has changed, read new data
                    self._process_file()
                    self.last_modified_time = current_mtime
                    self.last_file_size = current_size

            except Exception as e:
                logging.error(f"Error monitoring I-MON file: {e}")

            # Sleep briefly to avoid high CPU usage
            time.sleep(0.1)  # Check for updates 10 times per second

    def _process_file(self):
        """Read the file and extract new spectral data"""
        try:
            with open(self.file_path, 'r') as f:
                lines = f.readlines()

            # Find the header row index (contains "Date" and "Time")
            header_row = -1
            for i, line in enumerate(lines):
                if "Date" in line and "Time" in line:
                    header_row = i
                    break

            if header_row == -1:
                logging.warning("Could not find header row in I-MON file")
                return

            # Skip processing if no new lines
            if len(lines) <= self.last_line_count:
                return

            # Process only new lines
            start_line = max(self.last_line_count, header_row + 1)

            for i in range(start_line, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 2 + 200:  # Need date, time, and spectral data (200 channels)
                    continue

                try:
                    # Parse timestamp
                    date_str = parts[0]
                    time_str = parts[1]
                    timestamp = datetime.datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %I:%M:%S %p")

                    # Parse full spectral data
                    spectrum = np.array([float(v) for v in parts[2:2 + 200]])

                    # Set reference spectrum if not already set
                    if not self.reference_set:
                        if self.analyzer.set_reference_spectrum(spectrum):
                            self.reference_set = True

                    # Process the spectrum to find peaks and calculate strain
                    result = self.analyzer.process_spectrum(spectrum)

                    if result and result['main_peak_wavelength'] is not None:
                        # Create data package to emit
                        data = {
                            'timestamp': timestamp,
                            'wavelength': result['main_peak_wavelength'],
                            'strain': result['strain'] if result['strain'] is not None else 0,
                            'peaks_found': result['peaks_found'],
                            'spectrum': spectrum,
                            'analysis': result
                        }

                        # Emit the processed data
                        self.newData.emit(data)

                except Exception as e:
                    logging.warning(f"Error processing I-MON spectrum at line {i}: {e}")

            # Update line count
            self.last_line_count = len(lines)

        except Exception as e:
            logging.error(f"Error reading I-MON file: {e}")

    def stop(self):
        """Stop the monitoring thread"""
        logging.info("Stopping I-MON file monitor")
        self._running = False
        self.wait()