# imon_xlsx_parser.py - XLSX File Parser for I-MON Spectral Data
import pandas as pd
import numpy as np
import logging
import os
import datetime
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import xlrd  # For handling older XLS files
import openpyxl  # For handling newer XLSX files

# Import custom tab-delimited reader
from tab_delimited_reader import read_tab_delimited_file

class IMONXLSXParser:
    """
    Parser for I-MON spectral data in XLSX format.
    Reads calibration coefficients and spectral data from an XLSX file.
    Processes spectral data to extract reflected peaks and calculate strain.
    """
    
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.calibration_coeffs = None
        self.wavelengths = None
        self.pixel_numbers = None
        self.last_modified_time = 0
        self.last_column_count = 0
        self.reference_peak = None
        self.reference_wavelength = None
        self.strain_conversion_factor = 0.22  # Default strain conversion as per requirements
        
        # Initialize if file path is provided
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """
        Load an Excel file (XLS or XLSX) and extract calibration coefficients.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            bool: True if file was loaded successfully
        """
        self.file_path = file_path
        
        try:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return False
                
            # Check file size for debugging
            file_size = os.path.getsize(file_path)
            logging.info(f"File size: {file_size} bytes")
            
            # Determine file format based on extension
            file_ext = os.path.splitext(file_path)[-1].lower()
            
            coeffs = {}
            
            # Try different approaches to read the file
            try:
                # First attempt: Use openpyxl for XLSX
                logging.info("Attempting to read file with openpyxl...")
                wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                ws = wb.active
                
                # Extract calibration coefficients from A1, B1-B5
                coeff_cells = ['A1', 'B1', 'B2', 'B3', 'B4', 'B5']
                
                for cell in coeff_cells:
                    try:
                        cell_value = ws[cell].value
                        if cell_value is not None:
                            coeffs[cell] = float(cell_value)
                        else:
                            coeffs[cell] = None
                    except Exception as e:
                        logging.warning(f"Could not read coefficient {cell}: {e}")
                        coeffs[cell] = None
                
                logging.info("Successfully read file with openpyxl")
                
            except Exception as e:
                logging.warning(f"Could not read with openpyxl: {e}")
                
                try:
                    # Second attempt: Use xlrd for XLS
                    logging.info("Attempting to read file with xlrd...")
                    wb = xlrd.open_workbook(file_path)
                    ws = wb.sheet_by_index(0)
                    
                    # Map cell references to row, col coordinates
                    cell_coords = {
                        'A1': (0, 0),
                        'B1': (0, 1),
                        'B2': (1, 1),
                        'B3': (2, 1),
                        'B4': (3, 1),
                        'B5': (4, 1)
                    }
                    
                    for cell, (row, col) in cell_coords.items():
                        try:
                            if row < ws.nrows and col < ws.ncols:
                                cell_value = ws.cell_value(row, col)
                                if cell_value:
                                    coeffs[cell] = float(cell_value)
                                else:
                                    coeffs[cell] = None
                            else:
                                coeffs[cell] = None
                        except Exception as e:
                            logging.warning(f"Could not read coefficient {cell}: {e}")
                            coeffs[cell] = None
                    
                    logging.info("Successfully read file with xlrd")
                    
                except Exception as e:
                    logging.warning(f"Could not read with xlrd: {e}")
                    
                    # Third attempt: Use pandas
                    logging.info("Attempting to read file with pandas...")
                    # Read first few rows to extract calibration coefficients
                    df_coeffs = pd.read_excel(file_path, header=None, nrows=5, engine='auto')
                    
                    if not df_coeffs.empty:
                        # Try to extract A1, B1-B5
                        if df_coeffs.shape[1] > 1:
                            coeffs['A1'] = df_coeffs.iloc[0, 0] if not pd.isna(df_coeffs.iloc[0, 0]) else None
                            coeffs['B1'] = df_coeffs.iloc[0, 1] if not pd.isna(df_coeffs.iloc[0, 1]) else None
                            
                            if df_coeffs.shape[0] > 1:
                                coeffs['B2'] = df_coeffs.iloc[1, 1] if not pd.isna(df_coeffs.iloc[1, 1]) else None
                            if df_coeffs.shape[0] > 2:
                                coeffs['B3'] = df_coeffs.iloc[2, 1] if not pd.isna(df_coeffs.iloc[2, 1]) else None
                            if df_coeffs.shape[0] > 3:
                                coeffs['B4'] = df_coeffs.iloc[3, 1] if not pd.isna(df_coeffs.iloc[3, 1]) else None
                            if df_coeffs.shape[0] > 4:
                                coeffs['B5'] = df_coeffs.iloc[4, 1] if not pd.isna(df_coeffs.iloc[4, 1]) else None
                    
                    logging.info("Attempted to read with pandas")
            
            # If we couldn't get any coefficients, provide default values
            if not coeffs or all(v is None for v in coeffs.values()):
                logging.warning("Could not read coefficients, using default values")
                # Provide default calibration coefficients similar to example
                coeffs = {
                    'A1': 1.60E+03,
                    'B1': -1.37E-01,
                    'B2': -6.30E-05,
                    'B3': 7.95E-09,
                    'B4': -2.84E-11,
                    'B5': 2.28E-14
                }
            
            self.calibration_coeffs = coeffs
            
            # Determine the last modified time
            self.last_modified_time = os.path.getmtime(file_path)
            
            logging.info(f"Loaded Excel file: {file_path}")
            logging.info(f"Calibration coefficients: {coeffs}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading Excel file {file_path}: {e}")
            return False
    
    def check_for_updates(self):
        """
        Check if the XLSX file has been updated.
        
        Returns:
            bool: True if file has been updated
        """
        if not self.file_path or not os.path.exists(self.file_path):
            return False
            
        try:
            current_mtime = os.path.getmtime(self.file_path)
            if current_mtime > self.last_modified_time:
                return True
            return False
        except Exception as e:
            logging.error(f"Error checking file modification time: {e}")
            return False
    
    def read_latest_spectrum(self):
        """
        Read the latest spectrum data from the Excel file (XLS or XLSX).
        
        Returns:
            dict: Dictionary containing the latest spectrum data or None on error
        """
        if not self.file_path or not os.path.exists(self.file_path):
            logging.error("No valid file path provided")
            return None
            
        try:
            # File exists, proceed with reading
            logging.info(f"Attempting to read spectrum data from {self.file_path}")
            
            # Try standard Excel readers first
            excel_result = self._try_excel_readers()
            
            if excel_result:
                return excel_result
            
            # If standard readers failed, try custom tab-delimited reader
            logging.info("Standard Excel readers failed, trying custom tab-delimited reader")
            from tab_delimited_reader import read_tab_delimited_file
            tab_result = read_tab_delimited_file(self.file_path)
            
            if tab_result:
                # Store wavelengths and pixel numbers for future use
                self.wavelengths = tab_result['wavelengths']
                self.pixel_numbers = tab_result['pixel_numbers']
                
                # Update calibration coefficients if available
                if 'calibration' in tab_result and tab_result['calibration']:
                    self.calibration_coeffs = tab_result['calibration']
                
                # Update last modified time
                self.last_modified_time = os.path.getmtime(self.file_path)
                
                return tab_result
            
            # If all readers failed
            logging.error("All file reading methods failed")
            return None
            
        except Exception as e:
            logging.error(f"Error reading latest spectrum: {e}")
            return None
            
    def _try_excel_readers(self):
        """Try standard Excel readers"""
        try:
            # Try with auto engine first
            df = pd.read_excel(self.file_path, header=None, engine='auto')
        except Exception as e:
            logging.warning(f"Auto engine failed: {e}, trying openpyxl...")
            try:
                # Try with openpyxl explicitly
                df = pd.read_excel(self.file_path, header=None, engine='openpyxl')
            except Exception as e:
                logging.warning(f"openpyxl engine failed: {e}, trying xlrd...")
                try:
                    # Try with xlrd explicitly
                    df = pd.read_excel(self.file_path, header=None, engine='xlrd')
                except Exception as e:
                    logging.error(f"All Excel engines failed: {e}")
                    return None
        
        logging.info(f"Successfully loaded file with Excel reader, data shape: {df.shape}")
        
        # Share a sample of the data for debugging
        if not df.empty:
            logging.info(f"First few rows sample: {df.iloc[:5, :5]}")
        
        # Find the row containing 'Pixel No.' and 'Wavelength'
        header_row = None
        for i, row in df.iterrows():
            if 'Pixel No.' in row.values or 'Pixel' in row.values:
                pixel_found = True
                if 'Wavelength' in row.values:
                    header_row = i
                    logging.info(f"Found header row at index {i}")
                    break
        
        # If we couldn't find the exact header, try a more flexible approach
        if header_row is None:
            logging.warning("Could not find standard header row, attempting alternative detection")
            
            # Look for any row that might contain relevant column names
            for i, row in df.iterrows():
                row_str = ' '.join([str(x).lower() for x in row.values if x is not None])
                if 'pixel' in row_str and ('wavelength' in row_str or 'wave' in row_str):
                    header_row = i
                    logging.info(f"Found approximate header row at index {i}")
                    break
        
        # If still no header row, use a default location
        if header_row is None:
            logging.warning("Using default header row location (row 11)")
            header_row = 11  # 0-based index for row 12 in the file
            
        # Find the date and time rows (should be above the header row)
        date_row = max(0, header_row - 2)
        time_row = max(0, header_row - 1)
        
        # Extract column indices for Pixel No. and Wavelength
        pixel_col = None
        wavelength_col = None
        
        # Check each column in the header row
        for col in range(df.shape[1]):
            cell_value = str(df.iloc[header_row, col]).lower() if df.iloc[header_row, col] is not None else ""
            if 'pixel' in cell_value:
                pixel_col = col
                logging.info(f"Found Pixel column at index {col}")
            elif 'wave' in cell_value:
                wavelength_col = col
                logging.info(f"Found Wavelength column at index {col}")
        
        # If we couldn't find the columns, use default values
        if pixel_col is None:
            logging.warning("Could not find Pixel column, using default (column 0)")
            pixel_col = 0
        
        if wavelength_col is None:
            logging.warning("Could not find Wavelength column, using default (column 1)")
            wavelength_col = 1
        
        # Count how many 'Counts' columns there are (should be all columns after Wavelength)
        counts_cols = []
        for col in range(wavelength_col + 1, df.shape[1]):
            # Check if this column contains 'Counts' or is after known Counts columns
            if col < df.shape[1] and df.iloc[header_row, col] is not None:
                cell_value = str(df.iloc[header_row, col]).lower()
                if 'count' in cell_value or (counts_cols and col == counts_cols[-1] + 1):
                    counts_cols.append(col)
        
        if not counts_cols:
            logging.warning("No 'Counts' columns found, assuming all columns after wavelength are data")
            # Assume all columns after wavelength are data columns
            counts_cols = list(range(wavelength_col + 1, df.shape[1]))
        
        if not counts_cols:
            logging.error("Still no data columns found")
            return None
        
        # Get the latest 'Counts' column (the last one)
        latest_counts_col = counts_cols[-1]
        logging.info(f"Using data from column {latest_counts_col}")
        
        # Try to get the corresponding date and time
        try:
            if date_row < df.shape[0] and latest_counts_col < df.shape[1]:
                date_str = str(df.iloc[date_row, latest_counts_col])
                time_str = str(df.iloc[time_row, latest_counts_col])
                
                # Try multiple timestamp formats
                timestamp = None
                for fmt in ["%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"]:
                    try:
                        timestamp = datetime.datetime.strptime(f"{date_str} {time_str}", fmt)
                        break
                    except ValueError:
                        continue
                
                if timestamp is None:
                    logging.warning(f"Could not parse timestamp: {date_str} {time_str}, using current time")
                    timestamp = datetime.datetime.now()
            else:
                logging.warning("Date/time row or column out of bounds, using current time")
                timestamp = datetime.datetime.now()
        except Exception as e:
            logging.warning(f"Error parsing timestamp: {e}, using current time")
            timestamp = datetime.datetime.now()
        
        # Extract data - try to get 512 rows of data as specified in requirements
        start_row = header_row + 1
        
        # Try to determine how many data rows we actually have
        data_rows = 0
        for i in range(start_row, min(start_row + 600, df.shape[0])):
            if i < df.shape[0] and df.iloc[i, pixel_col] is not None:
                # Check if this still looks like data and not some footer
                try:
                    # Try to convert to number - if it fails, we've probably hit non-data
                    float(df.iloc[i, pixel_col])
                    data_rows += 1
                except:
                    break
        
        logging.info(f"Found {data_rows} data rows")
        
        # If we found no data rows, use the default 512
        if data_rows == 0:
            data_rows = min(512, df.shape[0] - start_row)
            logging.warning(f"Could not detect data rows, using default: {data_rows}")
        
        end_row = start_row + data_rows
        
        # Ensure we don't go beyond the DataFrame bounds
        end_row = min(end_row, df.shape[0])
        
        # Extract pixel numbers, wavelengths, and counts
        try:
            pixel_numbers = df.iloc[start_row:end_row, pixel_col].values
            wavelengths = df.iloc[start_row:end_row, wavelength_col].values
            counts = df.iloc[start_row:end_row, latest_counts_col].values
            
            # Convert to numeric types if needed
            pixel_numbers = pd.to_numeric(pixel_numbers, errors='coerce').fillna(0).astype(int)
            wavelengths = pd.to_numeric(wavelengths, errors='coerce').fillna(0)
            counts = pd.to_numeric(counts, errors='coerce').fillna(0)
            
            # Check if we have valid data
            if len(wavelengths) == 0 or len(counts) == 0:
                logging.error("No valid data extracted")
                return None
            
            # Log some statistics for debugging
            logging.info(f"Data statistics: {len(wavelengths)} wavelengths, {len(counts)} counts")
            logging.info(f"Wavelength range: {min(wavelengths)} to {max(wavelengths)}")
            logging.info(f"Counts range: {min(counts)} to {max(counts)}")
            
            # Store wavelengths and pixel numbers for future use
            self.wavelengths = wavelengths
            self.pixel_numbers = pixel_numbers
            
            # Check if this column count is different from last time (new data)
            is_new_data = len(counts_cols) != self.last_column_count
            self.last_column_count = len(counts_cols)
            
            # Update last modified time
            self.last_modified_time = os.path.getmtime(self.file_path)
            
            # Create result dictionary
            result = {
                'timestamp': timestamp,
                'pixel_numbers': pixel_numbers,
                'wavelengths': wavelengths,
                'counts': counts,
                'is_new_data': is_new_data,
                'column_index': latest_counts_col
            }
            
            logging.info(f"Successfully read spectrum data from column {latest_counts_col}, timestamp: {timestamp}")
            
            return result
        except Exception as e:
            logging.error(f"Error extracting data: {e}")
            return None
    
    def process_spectrum(self, spectrum_data):
        """
        Process spectrum data to extract reflected peaks and calculate strain.
        
        Args:
            spectrum_data (dict): Dictionary containing spectrum data
            
        Returns:
            dict: Dictionary containing processed spectrum data or None on error
        """
        try:
            wavelengths = spectrum_data['wavelengths']
            counts = spectrum_data['counts']
            
            # Apply smoothing filter to reduce noise
            smoothed_counts = savgol_filter(counts, 11, 3)
            
            # 1. Fit a Gaussian curve to the transmission spectrum
            def gaussian(x, a, b, c):
                return a * np.exp(-(x - b)**2 / (2 * c**2))
            
            # Initial guess for the Gaussian parameters
            initial_guess = [np.max(smoothed_counts), np.mean(wavelengths), 20]
            
            try:
                # Fit the Gaussian curve
                params, _ = curve_fit(gaussian, wavelengths, smoothed_counts, p0=initial_guess)
                
                # Generate the Gaussian curve with the fitted parameters
                fitted_gaussian = gaussian(wavelengths, *params)
                
                # 2. Subtract the Gaussian to flatten the curve
                flattened = smoothed_counts - fitted_gaussian
                
                # 3. Invert the resulting data to transform troughs into reflected peaks
                reflected = -flattened
                
                # 4. Identify the wavelength positions of these reflected peaks
                peaks, properties = find_peaks(
                    reflected,
                    height=np.median(reflected) * 1.5,  # Peak must be 50% above median
                    distance=5  # Minimum distance between peaks
                )
                
                # If no peaks found, return with limited data
                if len(peaks) == 0:
                    return {
                        'peaks_found': 0,
                        'main_peak_idx': None,
                        'main_peak_wavelength': None,
                        'strain': None,
                        'reflection_data': reflected,
                        'gaussian_fit': fitted_gaussian,
                        'flattened_data': flattened
                    }
                
                # Find the most prominent peak (highest)
                peak_heights = properties['peak_heights']
                main_peak_idx = peaks[np.argmax(peak_heights)]
                main_peak_wavelength = wavelengths[main_peak_idx]
                
                # Calculate strain if we have a reference peak
                strain = None
                if self.reference_wavelength is None:
                    # Set this as the reference if we don't have one yet
                    self.reference_wavelength = main_peak_wavelength
                    strain = 0.0
                else:
                    # Calculate strain using the formula: ε = Δλ/k = (1550 - λpeak)/0.22
                    # As specified in the requirements
                    wavelength_shift = 1550 - main_peak_wavelength  # in nm
                    strain = wavelength_shift / self.strain_conversion_factor
                
                return {
                    'peaks_found': len(peaks),
                    'all_peaks': peaks,
                    'peak_heights': peak_heights,
                    'main_peak_idx': main_peak_idx,
                    'main_peak_wavelength': main_peak_wavelength,
                    'strain': strain,
                    'gaussian_fit': fitted_gaussian,
                    'flattened_data': flattened,
                    'reflection_data': reflected
                }
                
            except RuntimeError as e:
                logging.error(f"Error fitting Gaussian: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Error processing spectrum: {e}")
            return None
    
    def reset_reference(self):
        """Reset the reference wavelength for strain calculation"""
        self.reference_wavelength = None
        logging.info("Reset reference wavelength")
    
    def set_strain_conversion_factor(self, factor):
        """Set the strain conversion factor"""
        self.strain_conversion_factor = factor
        logging.info(f"Set strain conversion factor to {factor}")
    
    def plot_spectrum(self, spectrum_data, processed_data=None, show=True):
        """
        Create a plot of the spectrum with Gaussian fit and reflected peaks.
        
        Args:
            spectrum_data (dict): Dictionary containing spectrum data
            processed_data (dict, optional): Dictionary containing processed spectrum data
            show (bool): Whether to display the plot immediately
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot original spectrum (transmission)
        wavelengths = spectrum_data['wavelengths']
        counts = spectrum_data['counts']
        
        axs[0].plot(wavelengths, counts, 'b-', label='Raw spectrum')
        axs[0].set_xlabel('Wavelength (nm)')
        axs[0].set_ylabel('Counts')
        axs[0].set_title('Transmission Spectrum')
        axs[0].grid(True)
        
        # If we have processed data, add Gaussian fit
        if processed_data and 'gaussian_fit' in processed_data:
            axs[0].plot(wavelengths, processed_data['gaussian_fit'], 'r--', 
                       label='Gaussian fit')
        
        axs[0].legend()
        
        # Plot reflected spectrum (processing result)
        if processed_data and 'reflection_data' in processed_data:
            reflected = processed_data['reflection_data']
            
            axs[1].plot(wavelengths, reflected, 'g-', label='Reflected spectrum')
            axs[1].set_xlabel('Wavelength (nm)')
            axs[1].set_ylabel('Reflected Intensity')
            axs[1].set_title('Reflected Spectrum')
            axs[1].grid(True)
            
            # Mark peaks
            if 'all_peaks' in processed_data and len(processed_data['all_peaks']) > 0:
                peak_indices = processed_data['all_peaks']
                peak_wavelengths = wavelengths[peak_indices]
                peak_intensities = reflected[peak_indices]
                axs[1].plot(peak_wavelengths, peak_intensities, 'ro', label='Detected peaks')
                
                # Highlight main peak
                if processed_data['main_peak_idx'] is not None:
                    main_idx = processed_data['main_peak_idx']
                    main_wavelength = wavelengths[main_idx]
                    main_intensity = reflected[main_idx]
                    axs[1].plot(main_wavelength, main_intensity, 'go', markersize=10,
                               label=f'Main peak: {main_wavelength:.4f}nm')
                    
                    # Add strain information if available
                    if processed_data['strain'] is not None:
                        axs[1].text(0.05, 0.95, f"Strain: {processed_data['strain']:.2f} µε",
                                   transform=axs[1].transAxes, fontsize=12,
                                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            axs[1].legend()
        
        fig.tight_layout()
        
        if show:
            plt.show()
        
        return fig