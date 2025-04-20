# archery_gui_xlsx.py - Extended Archery GUI with XLSX Support
import sys
import time
import logging
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QFileDialog, QLabel,
                             QStyleFactory, QPushButton, QGroupBox, QFormLayout,
                             QLineEdit, QComboBox, QCheckBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPalette, QColor
import pyqtgraph as pg
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Import our custom modules
from serial_reader_thread import SerialReaderThread
from orientation_cube import OrientationCube
from data_processor import SensorDataProcessor

# Import original IMON modules
from imon_file_monitor import IMONFileMonitor
from imon_spectral import IMONSpectralAnalyzer, read_imon_spectral_data

# Import new XLSX modules
from imon_xlsx_parser import IMONXLSXParser
from imon_xlsx_monitor import IMONXLSXMonitor
from enhanced_spectrum_dialog import EnhancedSpectrumViewDialog

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Main GUI Application ---
class ArcheryPerformanceGUIXLSX(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Archery Performance Analyzer (XLSX Support)")

        # Initialize data processor
        self.data_processor = SensorDataProcessor(buffer_size=2000)

        # Initialize I-MON data buffers
        self.imon_timestamps = []
        self.imon_wavelengths = []
        self.imon_strains = []
        self.current_spectrum = None
        self.current_wavelengths = None
        self.current_processed_data = None
        self.imon_active = False
        
        # Choose between TXT and XLSX parsers
        self.use_xlsx = True  # Default to XLSX mode
        
        # Initialize analyzers for both modes
        self.spectral_analyzer = IMONSpectralAnalyzer()
        self.xlsx_parser = IMONXLSXParser()

        # Set up the UI
        self._setup_ui()

        # Start the serial reader thread
        self.serialThread = SerialReaderThread(port='COM5', baud=115200)
        self.serialThread.newData.connect(self.handle_new_data)
        self.serialThread.start()

        # Create the appropriate file monitor (will be started when path is set)
        if self.use_xlsx:
            self.imon_monitor = IMONXLSXMonitor()
        else:
            self.imon_monitor = IMONFileMonitor()
            
        self.imon_monitor.newData.connect(self.handle_imon_data)

        # Use a timer to update plots at a faster rate (60fps)
        self.plotTimer = QTimer()
        self.plotTimer.setInterval(16)  # ~60fps
        self.plotTimer.timeout.connect(self.update_plots)
        self.plotTimer.start()

    def _setup_ui(self):
        """Set up the user interface"""
        # Create tabs for different views
        self.tabs = QTabWidget()

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        # Add tabs
        self.tabs.addTab(self._create_realtime_tab(), "Real-time View")
        self.tabs.addTab(self._create_analysis_tab(), "Shot Analysis")
        self.tabs.addTab(self._create_spectrum_tab(), "Spectrum Analysis")

        # Apply performance optimizations
        self._optimize_plot_settings()

        # Set window size
        self.resize(1200, 800)

    def _create_realtime_tab(self):
        """Create the real-time visualization tab"""
        tab = QWidget()
        layout = QHBoxLayout()
        tab.setLayout(layout)

        # Left side - plots
        plots_widget = QWidget()
        plots_layout = QVBoxLayout()
        plots_widget.setLayout(plots_layout)

        # 1. Acceleration plot
        self.accel_plot = pg.PlotWidget(title="Acceleration Magnitude")
        self.accel_plot.setLabel('left', "Acceleration", units='m/s²')
        self.accel_plot.setLabel('bottom', "Time", units='s')
        self.accel_plot.showGrid(x=True, y=True)
        self.accel_curve = self.accel_plot.plot(pen=pg.mkPen('r', width=2))
        plots_layout.addWidget(self.accel_plot)

        # 2. Gyro plot
        self.gyro_plot = pg.PlotWidget(title="Gyro Z (Yaw)")
        self.gyro_plot.setLabel('left', "Angular Velocity", units='deg/s')
        self.gyro_plot.setLabel('bottom', "Time", units='s')
        self.gyro_plot.showGrid(x=True, y=True)
        self.gyro_curve = self.gyro_plot.plot(pen=pg.mkPen('g', width=2))
        plots_layout.addWidget(self.gyro_plot)

        # 3. Strain plot
        self.strain_plot = pg.PlotWidget(title="Strain vs Time")
        self.strain_plot.setLabel('left', "Strain (µε)")
        self.strain_plot.setLabel('bottom', "Time", units='s')
        self.strain_plot.showGrid(x=True, y=True)
        self.strain_curve = self.strain_plot.plot(pen=pg.mkPen('b', width=2))
        plots_layout.addWidget(self.strain_plot)

        # Add plots to the left side of the layout
        layout.addWidget(plots_widget, stretch=2)

        # Right side - 3D visualization and metrics
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # 3D orientation visualization
        self.orientation_cube = OrientationCube()
        right_layout.addWidget(self.orientation_cube, stretch=3)

        # Metrics display
        metrics_group = QGroupBox("Shot Metrics")
        metrics_layout = QFormLayout()
        metrics_group.setLayout(metrics_layout)

        self.draw_duration_label = QLabel("--")
        self.release_impulse_label = QLabel("--")
        self.recovery_time_label = QLabel("--")
        self.peak_strain_label = QLabel("--")

        metrics_layout.addRow("Draw Duration (s):", self.draw_duration_label)
        metrics_layout.addRow("Release Impulse (m/s²):", self.release_impulse_label)
        metrics_layout.addRow("Recovery Time (s):", self.recovery_time_label)
        metrics_layout.addRow("Peak Strain (µε):", self.peak_strain_label)

        right_layout.addWidget(metrics_group)

        # I-MON Configuration
        imon_config_group = QGroupBox("I-MON Spectral Configuration")
        imon_config_layout = QFormLayout()
        imon_config_group.setLayout(imon_config_layout)

        # File format selector
        file_format_layout = QHBoxLayout()
        
        # Use radio buttons for better UX
        self.file_format_group = QButtonGroup()
        self.txt_format_radio = QRadioButton("TXT Format")
        self.xlsx_format_radio = QRadioButton("XLSX Format")
        
        self.file_format_group.addButton(self.txt_format_radio)
        self.file_format_group.addButton(self.xlsx_format_radio)
        
        # Set initial state
        self.txt_format_radio.setChecked(not self.use_xlsx)
        self.xlsx_format_radio.setChecked(self.use_xlsx)
        
        # Connect signals
        self.txt_format_radio.toggled.connect(self.toggle_file_format)
        
        file_format_layout.addWidget(self.txt_format_radio)
        file_format_layout.addWidget(self.xlsx_format_radio)
        
        # File path controls
        file_path_layout = QHBoxLayout()
        self.imon_path_edit = QLineEdit()
        if self.use_xlsx:
            self.imon_path_edit.setPlaceholderText("Path to I-MON XLSX file")
        else:
            self.imon_path_edit.setPlaceholderText("Path to I-MON data file")

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_imon_file)

        file_path_layout.addWidget(self.imon_path_edit, stretch=3)
        file_path_layout.addWidget(self.browse_button, stretch=1)

        # Strain conversion factor
        self.strain_factor_edit = QLineEdit()
        if self.use_xlsx:
            self.strain_factor_edit.setText("0.22")  # Default for XLSX as per requirements
            self.strain_factor_edit.setToolTip("Strain conversion factor for XLSX data")
        else:
            self.strain_factor_edit.setText("0.78")  # Default for TXT
            self.strain_factor_edit.setToolTip("Strain conversion factor (µε/pm)")

        # Monitor controls
        monitor_controls_layout = QHBoxLayout()
        self.start_monitor_button = QPushButton("Start Monitoring")
        self.start_monitor_button.clicked.connect(self.toggle_imon_monitor)

        self.reset_ref_button = QPushButton("Reset Reference")
        self.reset_ref_button.clicked.connect(self.reset_reference_spectrum)

        self.view_spectrum_button = QPushButton("View Spectrum")
        self.view_spectrum_button.clicked.connect(self.view_current_spectrum)
        self.view_spectrum_button.setEnabled(False)

        monitor_controls_layout.addWidget(self.start_monitor_button)
        monitor_controls_layout.addWidget(self.reset_ref_button)

        # Add to layout
        imon_config_layout.addRow("File Format:", file_format_layout)
        imon_config_layout.addRow("Data File:", file_path_layout)
        imon_config_layout.addRow("Strain Factor:", self.strain_factor_edit)
        imon_config_layout.addRow(monitor_controls_layout)
        imon_config_layout.addRow(self.view_spectrum_button)

        right_layout.addWidget(imon_config_group)

        # Control buttons
        controls_layout = QHBoxLayout()

        self.load_imon_button = QPushButton("Load I-MON Data")
        self.load_imon_button.clicked.connect(self.load_imon_data)
        controls_layout.addWidget(self.load_imon_button)

        self.export_button = QPushButton("Export Session")
        self.export_button.clicked.connect(self.export_session)
        controls_layout.addWidget(self.export_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_session)
        controls_layout.addWidget(self.reset_button)

        right_layout.addLayout(controls_layout)

        # Add the right panel to the layout
        layout.addWidget(right_panel, stretch=1)

        return tab

    def _create_analysis_tab(self):
        """Create the shot analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Create instruction label
        layout.addWidget(QLabel("Shot analysis features - this tab can be used to analyze individual shots."))
        
        # Add placeholder content (would be expanded in a full implementation)
        self.shot_list = QComboBox()
        self.shot_list.addItem("No shots recorded yet")
        
        layout.addWidget(QLabel("Select Shot:"))
        layout.addWidget(self.shot_list)
        
        layout.addStretch()

        return tab

    def _create_spectrum_tab(self):
        """Create the spectrum analysis tab for FBG data visualization"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Create instruction label
        layout.addWidget(QLabel("Spectrum analysis - view and analyze FBG spectral data here."))
        
        # Create live spectrum plotting area
        spectrum_group = QGroupBox("Live Spectrum Analysis")
        spectrum_layout = QVBoxLayout()
        spectrum_group.setLayout(spectrum_layout)
        
        # Use pyqtgraph for real-time plotting
        self.live_wavelength_plot = pg.PlotWidget(title="FBG Spectrum - Transmission")
        self.live_wavelength_plot.setLabel('left', "Intensity", units='counts')
        self.live_wavelength_plot.setLabel('bottom', "Wavelength", units='nm')
        self.live_wavelength_plot.showGrid(x=True, y=True)
        self.live_transmission_curve = self.live_wavelength_plot.plot(pen=pg.mkPen('b', width=2))
        spectrum_layout.addWidget(self.live_wavelength_plot)
        
        # Add reflected spectrum plot
        self.live_reflection_plot = pg.PlotWidget(title="FBG Spectrum - Reflection")
        self.live_reflection_plot.setLabel('left', "Reflected Intensity")
        self.live_reflection_plot.setLabel('bottom', "Wavelength", units='nm')
        self.live_reflection_plot.showGrid(x=True, y=True)
        self.live_reflection_curve = self.live_reflection_plot.plot(pen=pg.mkPen('g', width=2))
        spectrum_layout.addWidget(self.live_reflection_plot)
        
        # Controls for spectrum analysis
        controls_layout = QHBoxLayout()
        
        self.reprocess_spectrum_button = QPushButton("Reprocess Current Spectrum")
        self.reprocess_spectrum_button.clicked.connect(self.reprocess_current_spectrum)
        
        self.save_spectrum_button = QPushButton("Save Spectrum")
        self.save_spectrum_button.clicked.connect(self.save_current_spectrum)
        
        # Initially disable buttons until data is available
        self.reprocess_spectrum_button.setEnabled(False)
        self.save_spectrum_button.setEnabled(False)
        
        controls_layout.addWidget(self.reprocess_spectrum_button)
        controls_layout.addWidget(self.save_spectrum_button)
        spectrum_layout.addLayout(controls_layout)
        
        layout.addWidget(spectrum_group)

        return tab

    def _optimize_plot_settings(self):
        """Apply performance optimizations to all plots"""
        # Apply to all plot widgets
        for plot in [self.accel_plot, self.gyro_plot, self.strain_plot, 
                    self.live_wavelength_plot, self.live_reflection_plot]:
            # Disable autoRange for better performance
            plot.enableAutoRange(x=False, y=False)

            # Optimize rendering
            plot.setDownsampling(auto=True, mode='peak')
            plot.setClipToView(True)

            # Reduce update overhead
            plot.setMouseEnabled(x=True, y=True)  # Allow zoom/pan

        # Set fixed ranges based on expected data
        self.accel_plot.setXRange(0, 10)  # 10 seconds of data
        self.gyro_plot.setXRange(0, 10)  # 10 seconds of data
        self.strain_plot.setXRange(0, 10)  # 10 seconds of data

        # Apply optimizations after creating plots - with better ranges
        self.accel_plot.setYRange(0, 35)  # Increased range to handle up to 35 m/s²
        self.gyro_plot.setYRange(-150, 150)  # Wider gyro range (-150 to 150 deg/s)
        self.strain_plot.setYRange(0, 1000)  # Initial strain range (0-1000 µε)
        
        # Set appropriate ranges for wavelength plots
        self.live_wavelength_plot.setYRange(0, 5000)  # Typical intensity range
        self.live_reflection_plot.setYRange(-500, 500)  # Typical reflected range
        
        # Wavelength ranges will be adjusted once data is loaded

        # Add a reference line for Earth's gravity
        gravity_line = pg.InfiniteLine(
            pos=9.81,  # Earth's gravity in m/s²
            angle=0,  # Horizontal line
            pen=pg.mkPen((100, 100, 100), width=1, style=pg.QtCore.Qt.DashLine)
        )
        self.accel_plot.addItem(gravity_line)

        # Re-enable autorange for initial display then disable after 2 seconds
        for plot in [self.accel_plot, self.gyro_plot, self.strain_plot]:
            plot.enableAutoRange(x=True, y=True)

        # Create a timer to disable autorange after data starts flowing
        self.autorange_timer = QTimer()
        self.autorange_timer.setSingleShot(True)
        self.autorange_timer.timeout.connect(self._disable_autorange)
        self.autorange_timer.start(2000)  # 2 seconds

    def _disable_autorange(self):
        """Disable autorange after initial data display"""
        for plot in [self.accel_plot, self.gyro_plot, self.strain_plot]:
            plot.enableAutoRange(x=False, y=False)
            # Use fixed range that matches the data we've seen
            if plot == self.accel_plot:
                plot.setYRange(0, 35)  # Increased range to handle up to 35 m/s²
            elif plot == self.gyro_plot:
                plot.setYRange(-150, 150)

    def toggle_file_format(self, checked):
        """Toggle between TXT and XLSX file formats"""
        # Only process if the TXT radio button was toggled (this avoids double processing)
        if self.sender() == self.txt_format_radio:
            self.use_xlsx = not checked  # If TXT is checked, XLSX is not
            
            # Update strain factor default based on format
            if self.use_xlsx:
                self.strain_factor_edit.setText("0.22")  # Default for XLSX
                self.imon_path_edit.setPlaceholderText("Path to I-MON XLSX file")
            else:
                self.strain_factor_edit.setText("0.78")  # Default for TXT
                self.imon_path_edit.setPlaceholderText("Path to I-MON data file")
            
            # Stop the current monitor if running
            if hasattr(self, 'imon_monitor') and self.imon_monitor.isRunning():
                self.imon_monitor.stop()
                self.imon_active = False
                self.start_monitor_button.setText("Start Monitoring")
            
            # Create the appropriate monitor
            if self.use_xlsx:
                self.imon_monitor = IMONXLSXMonitor()
            else:
                self.imon_monitor = IMONFileMonitor()
                
            self.imon_monitor.newData.connect(self.handle_imon_data)
            
            # Reset data buffers
            self.clear_imon_data()
            
            logging.info(f"Switched to {'XLSX' if self.use_xlsx else 'TXT'} file format")

    def handle_new_data(self, data):
        """Process new sensor data from the serial thread with timestamp"""
        try:
            # Unpack timestamped data
            timestamp, ax, ay, az, gx, gy, gz = data

            # Process data through the sensor processor
            processed_data = self.data_processor.process_data(
                timestamp, (ax, ay, az), (gx, gy, gz))

            # Update metrics display if available
            metrics = processed_data['metrics']
            if metrics['draw_duration'] is not None:
                self.draw_duration_label.setText(f"{metrics['draw_duration']:.2f}")
            if metrics['release_impulse'] is not None:
                self.release_impulse_label.setText(f"{metrics['release_impulse']:.2f}")
            if metrics['recovery_time'] is not None:
                self.recovery_time_label.setText(f"{metrics['recovery_time']:.2f}")
            if metrics['peak_strain'] is not None:
                self.peak_strain_label.setText(f"{metrics['peak_strain']:.2f}")

        except Exception as e:
            logging.exception("Error in handle_new_data")

    def handle_imon_data(self, data):
        """Process new I-MON spectral data"""
        try:
            # Extract processed data
            timestamp = data['timestamp']
            wavelength = data['wavelength']
            strain = data['strain']
            
            if self.use_xlsx:
                # XLSX format includes wavelengths and full spectrum
                spectrum = {
                    'wavelengths': data['wavelengths'],
                    'counts': data['spectrum']
                }
                self.current_wavelengths = data['wavelengths']
                self.current_processed_data = data['analysis']
                
                # Update the live spectrum plot with new data
                self.update_spectrum_plots(data['wavelengths'], data['spectrum'], data['analysis'])
                
            else:
                # TXT format just has the spectrum
                spectrum = data['spectrum']
                self.current_processed_data = None
                
                # Cannot update spectrum plots without wavelength data

            # Store the data
            self.imon_timestamps.append(timestamp)
            self.imon_wavelengths.append(wavelength)
            self.imon_strains.append(strain)
            self.current_spectrum = spectrum  # Store most recent spectrum

            # Enable spectrum viewer button and spectrum analysis buttons
            self.view_spectrum_button.setEnabled(True)
            self.reprocess_spectrum_button.setEnabled(True)
            self.save_spectrum_button.setEnabled(True)

            # Handle very large datasets
            max_points = 3600  # 1 hour at 1Hz
            if len(self.imon_timestamps) > max_points:
                self.imon_timestamps = self.imon_timestamps[-max_points:]
                self.imon_wavelengths = self.imon_wavelengths[-max_points:]
                self.imon_strains = self.imon_strains[-max_points:]

            # Update peak strain metric
            if strain > self.data_processor.metrics.get('peak_strain', 0) or \
                    self.data_processor.metrics.get('peak_strain') is None:
                self.data_processor.metrics['peak_strain'] = strain
                self.peak_strain_label.setText(f"{strain:.2f}")

        except Exception as e:
            logging.exception(f"Error processing I-MON data: {e}")

    def update_spectrum_plots(self, wavelengths, counts, processed_data):
        """Update the live spectrum plots with new data"""
        if wavelengths is None or counts is None:
            return
            
        try:
            # Update transmission plot
            self.live_transmission_curve.setData(wavelengths, counts)
            
            # Update reflection plot if we have processed data
            if processed_data and 'reflection_data' in processed_data:
                reflection = processed_data['reflection_data']
                self.live_reflection_curve.setData(wavelengths, reflection)
                
                # Set appropriate wavelength range based on data
                min_wave = np.min(wavelengths)
                max_wave = np.max(wavelengths)
                self.live_wavelength_plot.setXRange(min_wave, max_wave)
                self.live_reflection_plot.setXRange(min_wave, max_wave)
                
                # Mark the main peak if available
                if 'main_peak_idx' in processed_data and processed_data['main_peak_idx'] is not None:
                    # Remove any existing peak markers
                    for item in self.live_reflection_plot.items():
                        if hasattr(item, 'peak_marker') and item.peak_marker:
                            self.live_reflection_plot.removeItem(item)
                    
                    # Add new peak marker
                    main_idx = processed_data['main_peak_idx']
                    peak_x = wavelengths[main_idx]
                    peak_y = reflection[main_idx]
                    
                    # Create scatter plot item for the peak
                    peak_marker = pg.ScatterPlotItem(
                        pos=[(peak_x, peak_y)],
                        size=15,
                        pen=pg.mkPen((0, 200, 0), width=2),
                        brush=pg.mkBrush(0, 200, 0, 200),
                        symbol='o'
                    )
                    peak_marker.peak_marker = True
                    self.live_reflection_plot.addItem(peak_marker)
                    
                    # Add text to show wavelength value
                    peak_text = pg.TextItem(
                        text=f"{peak_x:.4f} nm\n{processed_data['strain']:.2f} µε",
                        color=(0, 200, 0),
                        anchor=(0, 1)
                    )
                    peak_text.peak_marker = True
                    peak_text.setPos(peak_x, peak_y)
                    self.live_reflection_plot.addItem(peak_text)
                
        except Exception as e:
            logging.exception(f"Error updating spectrum plots: {e}")

    def update_plots(self):
        """Update all plots and visualizations with timestamp synchronization"""
        try:
            # Get data from processor
            accel_real_timestamps = list(self.data_processor.real_timestamps)
            accel_mag = list(self.data_processor.accel_mag)
            gyro_data = list(self.data_processor.gyro_raw)
            orientation_data = list(self.data_processor.orientation)

            # Update acceleration and gyro plots
            if len(accel_real_timestamps) > 1:
                # Optimize by limiting data points
                max_points = 200

                if len(accel_real_timestamps) > max_points:
                    start_idx = len(accel_real_timestamps) - max_points
                    real_timestamps = accel_real_timestamps[start_idx:]
                    accel_mag_display = accel_mag[start_idx:]
                    gyro_z = [g[2] for g in gyro_data[start_idx:]]
                else:
                    real_timestamps = accel_real_timestamps
                    accel_mag_display = accel_mag
                    gyro_z = [g[2] for g in gyro_data]

                # Find common time reference
                common_start_time = None

                # If I-MON data is available, find the latest start time
                if self.imon_timestamps and real_timestamps:
                    accel_start = real_timestamps[0]
                    imon_start = self.imon_timestamps[0]
                    common_start_time = max(accel_start, imon_start)
                else:
                    common_start_time = real_timestamps[0]

                # Convert to relative seconds using common start time
                rel_sec_accel = [(ts - common_start_time).total_seconds() for ts in real_timestamps]

                # Update acceleration and gyro plots
                self.accel_curve.setData(rel_sec_accel, accel_mag_display)
                self.gyro_curve.setData(rel_sec_accel, gyro_z)

                # Update orientation cube if needed
                if orientation_data:
                    latest_orientation = orientation_data[-1]
                    if hasattr(self, 'last_orientation'):
                        diff = sum(abs(a - b) for a, b in zip(latest_orientation, self.last_orientation))
                        if diff > 0.5:  # Update only if significant change
                            self.orientation_cube.update_orientation(*latest_orientation)
                            self.last_orientation = latest_orientation
                    else:
                        self.orientation_cube.update_orientation(*latest_orientation)
                        self.last_orientation = latest_orientation

            # Update strain plot with I-MON data
            if self.imon_active and len(self.imon_timestamps) > 0:
                # Limit data points for better performance
                max_points = 200

                if len(self.imon_timestamps) > max_points:
                    start_idx = len(self.imon_timestamps) - max_points
                    imon_timestamps = self.imon_timestamps[start_idx:]
                    strain_values = self.imon_strains[start_idx:]
                else:
                    imon_timestamps = self.imon_timestamps
                    strain_values = self.imon_strains

                # Use common start time for synchronization
                if common_start_time and imon_timestamps:
                    # Convert to relative seconds using the same reference point
                    rel_sec_strain = [(ts - common_start_time).total_seconds() for ts in imon_timestamps]

                    # Update strain plot with synchronized times
                    self.strain_curve.setData(rel_sec_strain, strain_values)

                    # Set consistent time window for all plots
                    if rel_sec_accel and rel_sec_strain:
                        # Find the overall time span
                        all_times = rel_sec_accel + rel_sec_strain
                        max_time = max(all_times)

                        # Show a consistent window (last 60 seconds)
                        window_size = 60  # seconds
                        x_max = max_time
                        x_min = max(0, x_max - window_size)

                        # Apply to all plots
                        self.accel_plot.setXRange(x_min, x_max)
                        self.gyro_plot.setXRange(x_min, x_max)
                        self.strain_plot.setXRange(x_min, x_max)

                        # Auto-adjust Y range for strain plot
                        if strain_values:
                            min_strain = min(strain_values)
                            max_strain = max(strain_values)
                            range_size = max_strain - min_strain
                            if range_size > 0:
                                margin = range_size * 0.1  # 10% margin
                                self.strain_plot.setYRange(min_strain - margin, max_strain + margin)

        except Exception as e:
            logging.exception("Error in update_plots")

    def browse_imon_file(self):
        """Open file dialog to select I-MON data file"""
        if self.use_xlsx:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select I-MON XLSX File", "", "Excel Files (*.xlsx);;All Files (*)")
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select I-MON Data File", "", "Text Files (*.txt);;All Files (*)")

        if file_path:
            self.imon_path_edit.setText(file_path)
            logging.info(f"Selected I-MON file: {file_path}")

    def toggle_imon_monitor(self):
        """Toggle the I-MON file monitor on/off"""
        if not self.imon_active:
            # Get file path
            file_path = self.imon_path_edit.text().strip()
            if not file_path:
                logging.warning("No I-MON file path specified")
                return

            # Check if file exists
            if not os.path.exists(file_path):
                logging.warning(f"I-MON file not found: {file_path}")
                return

            # Get strain conversion factor
            try:
                strain_factor = float(self.strain_factor_edit.text())
                if strain_factor <= 0:
                    raise ValueError("Strain factor must be positive")
            except ValueError:
                logging.warning("Invalid strain conversion factor, using default")
                strain_factor = 0.78 if not self.use_xlsx else 0.22  # Default based on format

            # Configure and start monitor
            self.imon_monitor.set_file_path(file_path)
            self.imon_monitor.set_strain_conversion_factor(strain_factor)

            if not self.imon_monitor.isRunning():
                self.imon_monitor.start()

            # Update UI
            self.start_monitor_button.setText("Stop Monitoring")
            self.imon_active = True
            logging.info(f"Started I-MON monitoring on file: {file_path}")
        else:
            # Stop monitoring
            if self.imon_monitor.isRunning():
                self.imon_monitor.stop()

            # Update UI
            self.start_monitor_button.setText("Start Monitoring")
            self.imon_active = False
            logging.info("Stopped I-MON monitoring")

    def view_current_spectrum(self):
        """View the current FBG spectrum in a dialog"""
        if self.current_spectrum is not None:
            try:
                if self.use_xlsx:
                    # For XLSX, we need both wavelengths and counts
                    dialog = EnhancedSpectrumViewDialog(
                        self, 
                        self.current_spectrum, 
                        self.xlsx_parser, 
                        xlsx_mode=True,
                        processed_data=self.current_processed_data
                    )
                else:
                    # For TXT, we just need the raw spectrum
                    dialog = EnhancedSpectrumViewDialog(
                        self, 
                        self.current_spectrum, 
                        self.spectral_analyzer, 
                        xlsx_mode=False
                    )
                    
                dialog.exec_()
            except Exception as e:
                logging.exception(f"Error viewing spectrum: {e}")
        else:
            logging.warning("No spectrum data available to view")

    def reprocess_current_spectrum(self):
        """Reprocess the current spectrum with current settings"""
        if not self.current_spectrum:
            logging.warning("No spectrum data available to reprocess")
            return
            
        try:
            # Get updated strain factor
            try:
                strain_factor = float(self.strain_factor_edit.text())
                if strain_factor <= 0:
                    raise ValueError("Strain factor must be positive")
            except ValueError:
                logging.warning("Invalid strain conversion factor, using default")
                strain_factor = 0.78 if not self.use_xlsx else 0.22
                
            if self.use_xlsx:
                # Update the strain conversion factor
                self.xlsx_parser.set_strain_conversion_factor(strain_factor)
                
                # Reprocess the current spectrum
                processed_data = self.xlsx_parser.process_spectrum(self.current_spectrum)
                self.current_processed_data = processed_data
                
                # Update the spectrum plots
                if processed_data:
                    self.update_spectrum_plots(
                        self.current_spectrum['wavelengths'], 
                        self.current_spectrum['counts'], 
                        processed_data
                    )
                    
                    # Show the updated spectrum
                    self.view_current_spectrum()
            else:
                # Update the strain conversion factor for TXT mode
                self.spectral_analyzer.set_strain_conversion_factor(strain_factor)
                
                # For TXT mode, we can't do much without knowing the wavelengths
                logging.info("Reprocessed spectrum with updated strain factor")
                
        except Exception as e:
            logging.exception(f"Error reprocessing spectrum: {e}")

    def save_current_spectrum(self):
        """Save the current spectrum to a CSV file"""
        if not self.current_spectrum:
            logging.warning("No spectrum data available to save")
            return
            
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Spectrum Data", "", "CSV Files (*.csv)")
                
            if not file_path:
                return
                
            if self.use_xlsx:
                # XLSX mode with wavelengths and counts
                with open(file_path, 'w') as f:
                    f.write("Wavelength,Counts,Reflection\n")
                    
                    wavelengths = self.current_spectrum['wavelengths']
                    counts = self.current_spectrum['counts']
                    
                    # Add reflection data if available
                    reflection = None
                    if self.current_processed_data and 'reflection_data' in self.current_processed_data:
                        reflection = self.current_processed_data['reflection_data']
                        
                    for i in range(len(wavelengths)):
                        if reflection is not None:
                            f.write(f"{wavelengths[i]:.6f},{counts[i]:.2f},{reflection[i]:.6f}\n")
                        else:
                            f.write(f"{wavelengths[i]:.6f},{counts[i]:.2f},\n")
                            
                # Also save peak information if available
                if self.current_processed_data and 'main_peak_wavelength' in self.current_processed_data:
                    peak_file = file_path.replace('.csv', '_peaks.csv')
                    with open(peak_file, 'w') as f:
                        f.write("Value,Description\n")
                        f.write(f"{self.current_processed_data['main_peak_wavelength']:.6f},Main Peak Wavelength (nm)\n")
                        if 'strain' in self.current_processed_data and self.current_processed_data['strain'] is not None:
                            f.write(f"{self.current_processed_data['strain']:.6f},Strain (µε)\n")
                
                logging.info(f"Saved spectrum data to {file_path}")
            else:
                # TXT mode - just raw spectrum
                with open(file_path, 'w') as f:
                    f.write("Index,Intensity\n")
                    for i, value in enumerate(self.current_spectrum):
                        f.write(f"{i},{value}\n")
                        
                logging.info(f"Saved raw spectrum data to {file_path}")
                
        except Exception as e:
            logging.exception(f"Error saving spectrum: {e}")

    def reset_reference_spectrum(self):
        """Reset the reference spectrum for strain calculations"""
        self.imon_monitor.reset_reference()
        logging.info("Reset spectral reference")

    def load_imon_data(self):
        """Load and analyze I-MON spectral data from a file"""
        try:
            if self.use_xlsx:
                # XLSX format
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Open I-MON XLSX File", "", "Excel Files (*.xlsx);;All Files (*)")
            else:
                # TXT format
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Open I-MON Data File", "", "Text Files (*.txt);;All Files (*)")

            if file_path:
                # Update the path field
                self.imon_path_edit.setText(file_path)

                # Clear existing data
                self.clear_imon_data()

                if self.use_xlsx:
                    # Initialize the parser
                    parser = IMONXLSXParser()
                    parser.load_file(file_path)
                    
                    # Read and process the file
                    spectrum_data = parser.read_latest_spectrum()
                    
                    if spectrum_data:
                        processed_data = parser.process_spectrum(spectrum_data)
                        
                        # Store the data
                        self.imon_timestamps.append(spectrum_data['timestamp'])
                        if processed_data and processed_data['main_peak_wavelength'] is not None:
                            self.imon_wavelengths.append(processed_data['main_peak_wavelength'])
                            self.imon_strains.append(processed_data['strain'])
                        else:
                            self.imon_wavelengths.append(0)
                            self.imon_strains.append(0)
                        
                        # Store current spectrum for viewing
                        self.current_spectrum = {
                            'wavelengths': spectrum_data['wavelengths'],
                            'counts': spectrum_data['counts']
                        }
                        self.current_wavelengths = spectrum_data['wavelengths']
                        self.current_processed_data = processed_data
                        
                        # Update spectrum plots
                        self.update_spectrum_plots(
                            spectrum_data['wavelengths'], 
                            spectrum_data['counts'], 
                            processed_data
                        )
                        
                        # Enable spectrum viewing
                        self.view_spectrum_button.setEnabled(True)
                        self.reprocess_spectrum_button.setEnabled(True)
                        self.save_spectrum_button.setEnabled(True)
                        
                        # Update peak strain metric
                        if processed_data and processed_data['strain'] is not None:
                            self.data_processor.metrics['peak_strain'] = processed_data['strain']
                            self.peak_strain_label.setText(f"{processed_data['strain']:.2f}")
                        
                        # Store the parser for later use
                        self.xlsx_parser = parser
                        
                        # Show the current spectrum
                        self.view_current_spectrum()
                        
                        logging.info(f"Loaded I-MON XLSX data from {file_path}")
                else:
                    # Original TXT format handling
                    # Initialize the analyzer
                    analyzer = IMONSpectralAnalyzer()
                    analyzer.load_calibration(file_path)
                    self.spectral_analyzer = analyzer

                    # Read and process the file
                    timestamps, strain_values, wavelength_values, raw_spectra = read_imon_spectral_data(
                        file_path, analyzer)

                    if timestamps is not None and strain_values is not None:
                        # Store the data
                        self.imon_timestamps = timestamps.tolist()
                        self.imon_strains = strain_values.tolist()
                        self.imon_wavelengths = wavelength_values.tolist()

                        if len(raw_spectra) > 0:
                            self.current_spectrum = raw_spectra[-1]  # Store most recent spectrum
                            self.view_spectrum_button.setEnabled(True)
                            self.reprocess_spectrum_button.setEnabled(True)
                            self.save_spectrum_button.setEnabled(True)

                        # Update peak strain metric
                        if len(strain_values) > 0:
                            peak_strain = np.max(strain_values)
                            self.data_processor.metrics['peak_strain'] = peak_strain
                            self.peak_strain_label.setText(f"{peak_strain:.2f}")

                        logging.info(f"Loaded I-MON TXT data from {file_path}")

                        # Show the current spectrum
                        self.view_current_spectrum()
        except Exception as e:
            logging.exception("Error in load_imon_data")

    def clear_imon_data(self):
        """Clear the I-MON data buffers"""
        self.imon_timestamps.clear()
        self.imon_wavelengths.clear()
        self.imon_strains.clear()
        self.current_spectrum = None
        self.current_wavelengths = None
        self.current_processed_data = None
        self.view_spectrum_button.setEnabled(False)
        self.reprocess_spectrum_button.setEnabled(False)
        self.save_spectrum_button.setEnabled(False)
        self.strain_curve.setData([], [])
        self.live_transmission_curve.setData([], [])
        self.live_reflection_curve.setData([], [])
        logging.info("Cleared I-MON data")

    def export_session(self):
        """Export session data to CSV"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Session Data", "", "CSV Files (*.csv)")

            if file_path:
                # Get data from processor with real timestamps
                accel_real_timestamps = list(self.data_processor.real_timestamps)
                accel_mag = list(self.data_processor.accel_mag)
                orientation = list(self.data_processor.orientation)

                # Create data arrays
                if len(accel_real_timestamps) > 0:
                    # Find common reference point if iMON data available
                    common_start_time = None

                    if self.imon_timestamps and accel_real_timestamps:
                        accel_start = accel_real_timestamps[0]
                        imon_start = self.imon_timestamps[0]
                        common_start_time = max(accel_start, imon_start)
                    else:
                        common_start_time = accel_real_timestamps[0]

                    # Convert to relative seconds
                    rel_timestamps = [(ts - common_start_time).total_seconds() for ts in accel_real_timestamps]

                    # Create export data for accelerometer
                    with open(file_path, 'w') as f:
                        # Write header
                        f.write("Time,AccelMag,Roll,Pitch,Yaw,Timestamp\n")

                        # Write data
                        for i in range(len(rel_timestamps)):
                            t = rel_timestamps[i]
                            a = accel_mag[i]
                            timestamp_str = accel_real_timestamps[i].strftime("%Y-%m-%d %H:%M:%S.%f")

                            if i < len(orientation):
                                r, p, y = orientation[i]
                                f.write(f"{t:.3f},{a:.3f},{r:.3f},{p:.3f},{y:.3f},{timestamp_str}\n")
                            else:
                                f.write(f"{t:.3f},{a:.3f},,,,{timestamp_str}\n")

                    # Export iMON data if available
                    if self.imon_timestamps and self.imon_strains and self.imon_wavelengths:
                        imon_file = file_path.replace('.csv', '_imon.csv')

                        # Convert to the same time reference
                        imon_rel_times = [(ts - common_start_time).total_seconds() for ts in self.imon_timestamps]

                        with open(imon_file, 'w') as f:
                            f.write("Time,Strain,Wavelength,Timestamp\n")
                            for i in range(len(imon_rel_times)):
                                t = imon_rel_times[i]
                                s = self.imon_strains[i]
                                w = self.imon_wavelengths[i]
                                timestamp_str = self.imon_timestamps[i].strftime("%Y-%m-%d %H:%M:%S.%f")
                                f.write(f"{t:.3f},{s:.3f},{w:.6f},{timestamp_str}\n")

                    # Also export metrics
                    metrics_file = file_path.replace('.csv', '_metrics.csv')
                    with open(metrics_file, 'w') as f:
                        f.write("Metric,Value\n")
                        for key, value in self.data_processor.metrics.items():
                            if value is not None:
                                f.write(f"{key},{value:.3f}\n")

                    logging.info(f"Exported session data to {file_path}")
        except Exception as e:
            logging.exception("Error in export_session")

    def reset_session(self):
        """Reset all data and start fresh"""
        try:
            # Reset data processor
            self.data_processor.reset()

            # Reset iMON data
            self.clear_imon_data()

            # Reset metrics display
            self.draw_duration_label.setText("--")
            self.release_impulse_label.setText("--")
            self.recovery_time_label.setText("--")
            self.peak_strain_label.setText("--")

            # Clear plots
            self.accel_curve.setData([], [])
            self.gyro_curve.setData([], [])
            self.strain_curve.setData([], [])

            logging.info("Reset session data")
        except Exception as e:
            logging.exception("Error in reset_session")

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop the serial thread
        if hasattr(self, 'serialThread'):
            self.serialThread.stop()

        # Stop the I-MON monitor
        if hasattr(self, 'imon_monitor'):
            self.imon_monitor.stop()

        event.accept()


def main():
    try:
        # Performance optimizations for PyQtGraph
        pg.setConfigOptions(antialias=True)  # Smoother lines
        pg.setConfigOptions(useOpenGL=True)  # Use OpenGL acceleration if available
        pg.setConfigOptions(foreground='w', background='k')  # Better contrast

        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create("Fusion"))

        # Create dark palette for better visualization
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))

        app.setPalette(dark_palette)

        window = ArcheryPerformanceGUIXLSX()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.exception("Error in main application startup")


if __name__ == "__main__":
    main()