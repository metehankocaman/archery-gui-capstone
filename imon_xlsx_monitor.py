# imon_xlsx_monitor.py - Monitor for XLSX Spectral Data Files
from PyQt5.QtCore import QThread, pyqtSignal
import time
import os
import logging
import numpy as np
import datetime
from imon_xlsx_parser import IMONXLSXParser


class IMONXLSXMonitor(QThread):
    """
    A thread that monitors an I-MON data file (XLSX format) for changes and processes 
    the full spectral data to extract wavelength shifts and calculate strain.
    """
    newData = pyqtSignal(object)  # Signal emitting a dict with processed spectral data

    def __init__(self, file_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._running = True
        
        # Initialize the parser
        self.parser = IMONXLSXParser()
        if file_path:
            self.parser.load_file(file_path)

    def set_file_path(self, file_path):
        """Set or update the file path to monitor"""
        self.file_path = file_path
        
        # Load file in the parser
        if file_path and os.path.exists(file_path):
            self.parser.load_file(file_path)
            
        logging.info(f"Set I-MON XLSX file path to: {file_path}")

    def set_strain_conversion_factor(self, factor):
        """Set the strain conversion factor used for calculations"""
        self.parser.set_strain_conversion_factor(factor)

    def reset_reference(self):
        """Reset the reference spectrum"""
        self.parser.reset_reference()
        logging.info("Reset spectral reference")

    def run(self):
        """Monitor the file for changes and process new spectral data when detected"""
        logging.info("I-MON XLSX file monitor started")
        while self._running:
            if not self.file_path or not os.path.exists(self.file_path):
                time.sleep(0.5)  # Sleep longer if file doesn't exist
                continue

            try:
                # Check if file has been updated
                if self.parser.check_for_updates():
                    # File has changed, read new data
                    self._process_file()
                
            except Exception as e:
                logging.error(f"Error monitoring I-MON XLSX file: {e}")

            # Sleep briefly to avoid high CPU usage
            time.sleep(0.1)  # Check for updates 10 times per second

    def _process_file(self):
        """Read the file and extract new spectral data"""
        try:
            # Read latest spectrum
            spectrum_data = self.parser.read_latest_spectrum()
            
            if spectrum_data:
                # Process spectrum to extract reflected peaks and calculate strain
                processed_data = self.parser.process_spectrum(spectrum_data)
                
                if processed_data:
                    # Create data package to emit
                    data = {
                        'timestamp': spectrum_data['timestamp'],
                        'wavelength': processed_data['main_peak_wavelength'] if processed_data['main_peak_wavelength'] is not None else 0,
                        'strain': processed_data['strain'] if processed_data['strain'] is not None else 0,
                        'peaks_found': processed_data['peaks_found'],
                        'spectrum': spectrum_data['counts'],
                        'wavelengths': spectrum_data['wavelengths'],
                        'analysis': processed_data
                    }
                    
                    # Emit the processed data
                    self.newData.emit(data)
                    
        except Exception as e:
            logging.error(f"Error processing I-MON XLSX file: {e}")

    def stop(self):
        """Stop the monitoring thread"""
        logging.info("Stopping I-MON XLSX file monitor")
        self._running = False
        self.wait()