# imon_spectral.py - FBG Spectral Analysis Module
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.signal import find_peaks, savgol_filter
import datetime
import os


class IMONSpectralAnalyzer:
    """
    Analyzer for I-MON spectral data from FBG sensors.
    Processes full spectral data to identify and track FBG peaks.
    """

    def __init__(self):
        self.calibration_coeffs = None
        self.wavelengths = None
        self.num_channels = 200  # Default channel count
        self.reference_peak = None
        self.reference_wavelength = None
        self.strain_conversion_factor = 0.78  # Default strain conversion (µε/pm), typical for 1550nm FBGs

    def load_calibration(self, file_path):
        """
        Load wavelength calibration coefficients from the I-MON data file.

        Args:
            file_path (str): Path to I-MON data file

        Returns:
            bool: True if calibration was successful
        """
        try:
            with open(file_path, 'r') as f:
                for _ in range(10):  # Check first 10 lines
                    line = f.readline().strip()
                    # Check if this is the calibration line (contains scientific notation)
                    if "E+" in line or "E-" in line:
                        parts = line.split('\t')

                        # Extract calibration coefficients
                        coeffs = {}
                        if len(parts) >= 6:  # Expect at least A1, B1-B5
                            try:
                                # Common coefficient names in I-MON files
                                coeff_names = ['A1', 'B1', 'B2', 'B3', 'B4', 'B5']
                                coeffs = {name: float(val) for name, val in zip(coeff_names, parts[:6])}

                                # Store coefficients
                                self.calibration_coeffs = coeffs

                                # Generate wavelength array
                                self._generate_wavelength_array()

                                logging.info("Loaded I-MON calibration coefficients")
                                return True
                            except ValueError:
                                logging.warning("Could not parse calibration coefficients")

            logging.warning("No calibration coefficients found in file")
            return False

        except Exception as e:
            logging.error(f"Error loading calibration: {e}")
            return False

    def _generate_wavelength_array(self):
        """Generate wavelength array from calibration coefficients"""
        if not self.calibration_coeffs:
            logging.warning("No calibration coefficients available")
            return

        try:
            # Use polynomial to convert pixel indices to wavelengths
            # Typical formula: wavelength = A1 + B1*x + B2*x^2 + B3*x^3 + ...
            x = np.arange(self.num_channels)

            # Apply polynomial formula
            wavelengths = np.zeros(self.num_channels)

            # First term is A1
            if 'A1' in self.calibration_coeffs:
                wavelengths += self.calibration_coeffs['A1']

            # Add B terms
            for i in range(1, 6):
                coeff_name = f'B{i}'
                if coeff_name in self.calibration_coeffs:
                    wavelengths += self.calibration_coeffs[coeff_name] * (x ** i)

            self.wavelengths = wavelengths
            logging.info(f"Generated wavelength array: {wavelengths[0]:.2f}nm - {wavelengths[-1]:.2f}nm")

        except Exception as e:
            logging.error(f"Error generating wavelength array: {e}")

    def process_spectrum(self, spectrum_data):
        """
        Process a single spectrum to find FBG peaks and calculate strain.

        Args:
            spectrum_data (np.ndarray): Array of intensity values (200 channels)

        Returns:
            dict: Dictionary containing peak information and strain data
        """
        if self.wavelengths is None:
            logging.warning("Wavelength calibration not available")
            return None

        if len(spectrum_data) != len(self.wavelengths):
            logging.warning(
                f"Spectrum length ({len(spectrum_data)}) doesn't match wavelength array ({len(self.wavelengths)})")
            return None

        try:
            # Apply smoothing filter to reduce noise
            smoothed_spectrum = savgol_filter(spectrum_data, 11, 3)

            # Find peaks in the spectrum
            peaks, properties = find_peaks(
                smoothed_spectrum,
                height=np.median(smoothed_spectrum) * 1.5,  # Peak must be 50% above median
                distance=5  # Minimum distance between peaks
            )

            # If no peaks found, return None
            if len(peaks) == 0:
                return {
                    'peaks_found': 0,
                    'main_peak_idx': None,
                    'main_peak_wavelength': None,
                    'strain': None
                }

            # Find the most prominent peak (highest)
            peak_heights = properties['peak_heights']
            main_peak_idx = peaks[np.argmax(peak_heights)]
            main_peak_wavelength = self.wavelengths[main_peak_idx]

            # Calculate strain if we have a reference peak
            strain = None
            if self.reference_wavelength is not None:
                # Convert wavelength shift to strain
                wavelength_shift = main_peak_wavelength - self.reference_wavelength  # in nm
                wavelength_shift_pm = wavelength_shift * 1000  # convert to picometers
                strain = wavelength_shift_pm * self.strain_conversion_factor  # µε (microstrain)

            return {
                'peaks_found': len(peaks),
                'all_peaks': peaks,
                'peak_heights': peak_heights,
                'main_peak_idx': main_peak_idx,
                'main_peak_wavelength': main_peak_wavelength,
                'strain': strain,
                'smoothed_spectrum': smoothed_spectrum
            }

        except Exception as e:
            logging.error(f"Error processing spectrum: {e}")
            return None

    def set_reference_spectrum(self, spectrum_data):
        """
        Set a reference spectrum for strain calculations.

        Args:
            spectrum_data (np.ndarray): Array of intensity values

        Returns:
            bool: True if reference was set successfully
        """
        try:
            # Process the spectrum
            result = self.process_spectrum(spectrum_data)

            if result and result['main_peak_wavelength'] is not None:
                self.reference_peak = result['main_peak_idx']
                self.reference_wavelength = result['main_peak_wavelength']
                logging.info(f"Set reference peak at {self.reference_wavelength:.4f} nm")
                return True
            else:
                logging.warning("Could not set reference - no valid peaks found")
                return False

        except Exception as e:
            logging.error(f"Error setting reference spectrum: {e}")
            return False

    def set_strain_conversion_factor(self, factor):
        """Set the strain conversion factor (µε/pm)"""
        self.strain_conversion_factor = factor
        logging.info(f"Set strain conversion factor to {factor} µε/pm")

    def reset_reference(self):
        """Reset the reference peak"""
        self.reference_peak = None
        self.reference_wavelength = None
        logging.info("Reset reference peak")

    def plot_spectrum(self, spectrum_data, analysis_result=None, show=True):
        """
        Create a plot of the spectrum with peak identification.

        Args:
            spectrum_data (np.ndarray): Array of intensity values
            analysis_result (dict, optional): Result from process_spectrum
            show (bool): Whether to display the plot immediately

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.wavelengths is None:
            logging.warning("Wavelength calibration not available")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot raw spectrum
        ax.plot(self.wavelengths, spectrum_data, 'b-', alpha=0.5, label='Raw spectrum')

        # If we have analysis results, add details
        if analysis_result:
            # Plot smoothed spectrum
            if 'smoothed_spectrum' in analysis_result:
                ax.plot(self.wavelengths, analysis_result['smoothed_spectrum'],
                        'g-', linewidth=1.5, label='Smoothed spectrum')

            # Mark peaks
            if 'all_peaks' in analysis_result and len(analysis_result['all_peaks']) > 0:
                peak_indices = analysis_result['all_peaks']
                peak_wavelengths = self.wavelengths[peak_indices]
                peak_intensities = spectrum_data[peak_indices]
                ax.plot(peak_wavelengths, peak_intensities, 'ro', label='Detected peaks')

                # Highlight main peak
                if analysis_result['main_peak_idx'] is not None:
                    main_idx = analysis_result['main_peak_idx']
                    main_wavelength = self.wavelengths[main_idx]
                    main_intensity = spectrum_data[main_idx]
                    ax.plot(main_wavelength, main_intensity, 'go', markersize=10,
                            label=f'Main peak: {main_wavelength:.4f}nm')

                    # Add strain information if available
                    if analysis_result['strain'] is not None:
                        ax.text(0.05, 0.95, f"Strain: {analysis_result['strain']:.2f} µε",
                                transform=ax.transAxes, fontsize=12,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Mark reference wavelength if available
            if self.reference_wavelength is not None:
                ax.axvline(x=self.reference_wavelength, color='k', linestyle='--',
                           label=f'Reference: {self.reference_wavelength:.4f}nm')

        # Add labels and legend
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title('FBG Reflection Spectrum')
        ax.legend(loc='best')
        ax.grid(True)

        # Adjust axes for better visibility
        if analysis_result and 'main_peak_idx' in analysis_result and analysis_result['main_peak_idx'] is not None:
            main_idx = analysis_result['main_peak_idx']
            main_wavelength = self.wavelengths[main_idx]
            # Zoom to region around main peak
            window = 2.0  # nm window around peak
            ax.set_xlim(main_wavelength - window / 2, main_wavelength + window / 2)

            # Adjust y-range
            buffer = 0.1 * (max(spectrum_data) - min(spectrum_data))
            ax.set_ylim(min(spectrum_data) - buffer, max(spectrum_data) + buffer)

        fig.tight_layout()

        if show:
            plt.show()

        return fig


# Function to read and process I-MON data file
def read_imon_spectral_data(filepath, analyzer=None, max_rows=None):
    """
    Read and process I-MON spectral data file, extracting timestamps and spectral data.

    Args:
        filepath (str): Path to I-MON data file
        analyzer (IMONSpectralAnalyzer, optional): Analyzer to use for processing
        max_rows (int, optional): Maximum number of rows to read

    Returns:
        tuple: (timestamps, strain_values, wavelength_values, raw_spectra)
               or (None, None, None, None) on error
    """
    try:
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return None, None, None, None

        # Create analyzer if not provided
        if analyzer is None:
            analyzer = IMONSpectralAnalyzer()
            if not analyzer.load_calibration(filepath):
                logging.warning("Could not load calibration, using default wavelength values")
                # Create default wavelength array from 1510 to 1590 nm
                analyzer.wavelengths = np.linspace(1590, 1510, 200)

        # Initialize data containers
        timestamps = []
        strain_values = []
        wavelength_values = []
        raw_spectra = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find header row (contains "Date" and "Time")
        header_row = -1
        for i, line in enumerate(lines):
            if "Date" in line and "Time" in line:
                header_row = i
                break

        if header_row == -1:
            logging.warning("Could not find header row in I-MON file")
            return None, None, None, None

        # Process data rows
        row_count = 0
        reference_set = False

        for i in range(header_row + 1, len(lines)):
            if max_rows is not None and row_count >= max_rows:
                break

            line = lines[i].strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 200 + 2:  # Need date, time, and spectral data
                continue

            try:
                # Parse timestamp
                date_str = parts[0]
                time_str = parts[1]
                timestamp = datetime.datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %I:%M:%S %p")

                # Parse spectral data
                spectrum = np.array([float(v) for v in parts[2:2 + 200] if v.strip()])

                # Set first spectrum as reference if not already set
                if not reference_set:
                    analyzer.set_reference_spectrum(spectrum)
                    reference_set = True

                # Process spectrum
                result = analyzer.process_spectrum(spectrum)

                if result and result['main_peak_wavelength'] is not None:
                    # Store the data
                    timestamps.append(timestamp)
                    wavelength_values.append(result['main_peak_wavelength'])
                    strain_values.append(result['strain'] if result['strain'] is not None else 0)
                    raw_spectra.append(spectrum)
                    row_count += 1

            except Exception as e:
                logging.warning(f"Error parsing line {i}: {e}")

        if not timestamps:
            logging.warning("No valid data rows found in I-MON file")
            return None, None, None, None

        # Convert to numpy arrays
        timestamps_array = np.array(timestamps)
        strain_array = np.array(strain_values)
        wavelength_array = np.array(wavelength_values)
        spectra_array = np.array(raw_spectra)

        return timestamps_array, strain_array, wavelength_array, spectra_array

    except Exception as e:
        logging.error(f"Error reading I-MON file {filepath}: {e}")
        return None, None, None, None