# tab_delimited_reader.py - Reader for tab-delimited files with XLSX extension
import os
import logging
import numpy as np
import pandas as pd
import datetime

def read_tab_delimited_file(file_path):
    """
    Read a tab-delimited file that has an XLSX extension but is actually a text file.
    
    Args:
        file_path (str): Path to the tab-delimited file
        
    Returns:
        dict: Dictionary containing the file data or None on error
    """
    try:
        logging.info(f"Reading file as tab-delimited text: {file_path}")
        
        # Read the entire file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
        
        # Display first few lines for debugging
        for i in range(min(15, len(all_lines))):
            logging.info(f"Line {i}: {all_lines[i][:50]}...")
        
        # Extract calibration coefficients from the first line
        coeffs = {}
        if all_lines and '\t' in all_lines[0]:
            coeff_values = all_lines[0].split('\t')
            coeff_names = ['A1', 'B1', 'B2', 'B3', 'B4', 'B5']
            
            for i, name in enumerate(coeff_names):
                if i < len(coeff_values):
                    try:
                        coeffs[name] = float(coeff_values[i])
                    except (ValueError, TypeError):
                        coeffs[name] = None
        
        logging.info(f"Extracted coefficients: {coeffs}")
        
        # Find date/time row
        date_row_idx = -1
        for i, line in enumerate(all_lines):
            if 'Date' in line and 'Time' in line:
                date_row_idx = i
                logging.info(f"Found date row at line {i}")
                break
        
        if date_row_idx == -1:
            logging.error("Could not find date row")
            return None
        
        # The wavelength row appears to be right after the date row
        # And the data starts on the row after that
        wavelength_row_idx = date_row_idx + 1
        data_start_idx = date_row_idx + 2
        
        # Extract wavelength values from the wavelength row
        if wavelength_row_idx < len(all_lines):
            wavelength_values = all_lines[wavelength_row_idx].strip().split('\t')
            # Skip any non-numeric values that might be at the beginning
            wavelengths = []
            for val in wavelength_values:
                try:
                    wavelengths.append(float(val))
                except (ValueError, TypeError):
                    pass
            
            if wavelengths:
                logging.info(f"Found {len(wavelengths)} wavelength values from {wavelength_row_idx}")
            else:
                logging.warning(f"No valid wavelength values found in row {wavelength_row_idx}")
                return None
        else:
            logging.error("Wavelength row is beyond file length")
            return None
        
        # Get the most recent data row (last row with timestamps)
        latest_data_row_idx = -1
        latest_timestamp = None
        
        for i in range(data_start_idx, len(all_lines)):
            line = all_lines[i].strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 3:  # Need at least date, time, and one value
                continue
                
            try:
                # Extract date and time from the row
                date_str = parts[0]
                time_str = parts[1]
                
                # Parse timestamp
                try:
                    timestamp = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %I:%M:%S %p")
                except ValueError:
                    try:
                        timestamp = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        logging.warning(f"Could not parse timestamp: {date_str} {time_str}")
                        continue
                
                # Keep track of the latest timestamp
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_data_row_idx = i
            except Exception as e:
                logging.warning(f"Error parsing row {i}: {e}")
        
        if latest_data_row_idx == -1:
            logging.error("No valid data rows found")
            return None
        
        logging.info(f"Found latest data at row {latest_data_row_idx}, timestamp: {latest_timestamp}")
        
        # Extract the data from the latest row
        latest_data = all_lines[latest_data_row_idx].strip().split('\t')
        
        # Skip date and time columns and extract counts
        counts = []
        for i in range(2, min(len(latest_data), len(wavelengths) + 2)):
            try:
                counts.append(float(latest_data[i]))
            except (ValueError, TypeError, IndexError):
                counts.append(0)
        
        # Make sure counts and wavelengths have the same length
        if len(counts) < len(wavelengths):
            # Pad counts with zeros if needed
            counts.extend([0] * (len(wavelengths) - len(counts)))
        elif len(counts) > len(wavelengths):
            # Trim counts if needed
            counts = counts[:len(wavelengths)]
        
        logging.info(f"Extracted {len(counts)} counts from latest data row")
        
        # Create pixel numbers
        pixel_numbers = np.arange(len(wavelengths))
        
        # Create result dictionary
        result = {
            'file_type': 'tab_delimited',
            'timestamp': latest_timestamp,
            'pixel_numbers': pixel_numbers,
            'wavelengths': np.array(wavelengths),
            'counts': np.array(counts),
            'calibration': coeffs,
            'is_new_data': True,
            'column_index': 2  # Starting column for data
        }
        
        logging.info(f"Successfully read tab-delimited file: {len(wavelengths)} wavelengths, {len(counts)} counts")
        logging.info(f"Wavelength range: {min(wavelengths)}-{max(wavelengths)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error reading tab-delimited file: {e}")
        return None