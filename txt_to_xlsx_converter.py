#!/usr/bin/env python
# txt_to_xlsx_converter.py - Convert IMON TXT data to XLSX format
import os
import sys
import pandas as pd
import logging
import argparse
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class IMONConverter:
    """Converts I-MON data from TXT format to XLSX format"""
    
    def __init__(self):
        self.calibration_coeffs = None
        self.header_row_idx = None
        self.wavelengths = None
        self.data = []
        
    def read_txt_file(self, input_file):
        """Read an I-MON data file in TXT format"""
        logging.info(f"Reading TXT file: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                
            # Extract calibration coefficients from first line
            if lines and '\t' in lines[0]:
                coeff_values = lines[0].strip().split('\t')
                if len(coeff_values) >= 6:
                    self.calibration_coeffs = {
                        'A1': coeff_values[0],
                        'B1': coeff_values[1],
                        'B2': coeff_values[2],
                        'B3': coeff_values[3],
                        'B4': coeff_values[4],
                        'B5': coeff_values[5]
                    }
                    logging.info(f"Extracted calibration coefficients: {self.calibration_coeffs}")
                
            # Find header row (containing "Date" and "Time")
            for i, line in enumerate(lines):
                if "Date" in line and "Time" in line:
                    self.header_row_idx = i
                    logging.info(f"Found header row at line {i+1}")
                    break
                    
            if self.header_row_idx is None:
                raise ValueError("Could not find header row with 'Date' and 'Time'")
                
            # Extract wavelength values (row after header)
            wavelength_row = self.header_row_idx + 1
            if wavelength_row < len(lines):
                wavelength_values = lines[wavelength_row].strip().split('\t')
                # Skip any non-numeric values that might be at the beginning (column labels)
                self.wavelengths = []
                for val in wavelength_values:
                    try:
                        self.wavelengths.append(float(val))
                    except (ValueError, TypeError):
                        pass
                
                logging.info(f"Extracted {len(self.wavelengths)} wavelength values")
            
            # Extract data rows
            data_start_idx = self.header_row_idx + 2
            for i in range(data_start_idx, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) < 3:  # Need at least date, time, and one value
                    continue
                    
                try:
                    # Process date and time
                    date_str = parts[0]
                    time_str = parts[1]
                    
                    # Extract spectral data values
                    values = []
                    for j in range(2, min(len(parts), 2 + len(self.wavelengths))):
                        try:
                            values.append(float(parts[j]))
                        except (ValueError, TypeError):
                            values.append(0)  # Use zero for invalid values
                    
                    # Add to data array
                    self.data.append({
                        'date': date_str,
                        'time': time_str,
                        'values': values
                    })
                    
                except Exception as e:
                    logging.warning(f"Error parsing row {i+1}: {e}")
            
            logging.info(f"Extracted {len(self.data)} data rows")
            return True
            
        except Exception as e:
            logging.error(f"Error reading TXT file: {e}")
            return False
    
    def write_xlsx_file(self, output_file):
        """Write data to XLSX format"""
        if not self.calibration_coeffs or not self.wavelengths or not self.data:
            logging.error("No data to write")
            return False
            
        try:
            logging.info(f"Writing XLSX file: {output_file}")
            
            # Create a pandas DataFrame for the XLSX file
            # First, create the calibration coefficients rows
            rows = []
            
            # Row 1: Calibration coefficient labels
            rows.append(["A1", "B1", "B2", "B3", "B4", "B5"])
            
            # Row 2: Calibration coefficient values
            rows.append([
                self.calibration_coeffs['A1'],
                self.calibration_coeffs['B1'],
                self.calibration_coeffs['B2'],
                self.calibration_coeffs['B3'],
                self.calibration_coeffs['B4'],
                self.calibration_coeffs['B5']
            ])
            
            # Add some blank rows to match the TXT format structure
            for _ in range(7):
                rows.append([])
            
            # Create the header row (Date, Time, followed by column indices)
            header_row = ["Date", "Time"]
            for i in range(len(self.wavelengths)):
                header_row.append(str(i))
            rows.append(header_row)
            
            # Add the wavelength row (empty for Date and Time, then wavelengths)
            wavelength_row = ["", ""]
            wavelength_row.extend(self.wavelengths)
            rows.append(wavelength_row)
            
            # Add data rows
            for data_point in self.data:
                data_row = [data_point['date'], data_point['time']]
                data_row.extend(data_point['values'])
                rows.append(data_row)
                
            # Create DataFrame and write to Excel
            df = pd.DataFrame(rows)
            
            # Add sheet name to match IMON format
            writer = pd.ExcelWriter(output_file, engine='openpyxl')
            df.to_excel(writer, sheet_name='Data', header=False, index=False)
            writer.save()
            
            logging.info(f"Successfully wrote {len(rows)} rows to XLSX file")
            return True
            
        except Exception as e:
            logging.error(f"Error writing XLSX file: {e}")
            return False

def convert_file(input_file, output_file=None):
    """Convert a single file from TXT to XLSX format"""
    # Generate output filename if not provided
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.xlsx"
    
    # Create converter and process file
    converter = IMONConverter()
    if converter.read_txt_file(input_file):
        return converter.write_xlsx_file(output_file)
    return False

def convert_directory(input_dir, output_dir=None):
    """Convert all TXT files in a directory to XLSX format"""
    # Use same directory if output_dir not specified
    if not output_dir:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process all TXT files
    success_count = 0
    fail_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.xlsx')
            
            logging.info(f"Converting {input_path} to {output_path}")
            if convert_file(input_path, output_path):
                success_count += 1
            else:
                fail_count += 1
    
    logging.info(f"Conversion complete. Successfully converted: {success_count}, Failed: {fail_count}")
    return success_count, fail_count

def main():
    """Main entry point for the converter application"""
    parser = argparse.ArgumentParser(description='Convert I-MON data from TXT to XLSX format')
    parser.add_argument('-i', '--input', required=True, help='Input TXT file or directory')
    parser.add_argument('-o', '--output', help='Output XLSX file or directory (optional)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Process directories recursively')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Convert a single file
        if convert_file(args.input, args.output):
            logging.info("File conversion successful")
            return 0
        else:
            logging.error("File conversion failed")
            return 1
    
    elif os.path.isdir(args.input):
        # Convert all files in directory
        success, fail = convert_directory(args.input, args.output)
        
        if args.recursive:
            # Process subdirectories recursively
            for root, dirs, files in os.walk(args.input):
                if root == args.input:
                    continue  # Skip the top directory, already processed
                
                # Determine corresponding output directory
                if args.output:
                    rel_path = os.path.relpath(root, args.input)
                    sub_output_dir = os.path.join(args.output, rel_path)
                else:
                    sub_output_dir = root
                
                # Process this subdirectory
                s, f = convert_directory(root, sub_output_dir)
                success += s
                fail += f
        
        logging.info(f"Total converted: {success}, Total failed: {fail}")
        return 0 if fail == 0 else 1
    
    else:
        logging.error(f"Input path does not exist: {args.input}")
        return 1

if __name__ == "__main__":
    sys.exit(main())