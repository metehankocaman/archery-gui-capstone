# archery_gui.py (Updated with I-MON Spectral Processing)
import sys, time, logging, os
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QFileDialog, QLabel,
                             QStyleFactory, QPushButton, QGroupBox, QFormLayout,
                             QLineEdit, QComboBox, QCheckBox, QDialog, QDialogButtonBox)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPalette, QColor
import pyqtgraph as pg
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Import our custom modules
from serial_reader_thread import SerialReaderThread
from orientation_cube import OrientationCube
from data_processor import SensorDataProcessor
from imon_file_monitor import IMONFileMonitor
from imon_spectral import IMONSpectralAnalyzer, read_imon_spectral_data

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


# Dialog for displaying FBG spectral data
class SpectrumViewDialog(QDialog):
    """Dialog for displaying FBG spectral data"""

    def __init__(self, parent=None, spectrum=None, analyzer=None):
        super().__init__(parent)
        self.setWindowTitle("FBG Spectrum Viewer")
        self.resize(800, 600)
        self.spectrum = spectrum
        self.analyzer = analyzer or IMONSpectralAnalyzer()

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create matplotlib figure and canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Add close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)

        # Plot the spectrum if provided
        if spectrum is not None:
            self.plot_spectrum()

    def plot_spectrum(self):
        """Plot the spectrum using the analyzer"""
        if self.spectrum is not None and self.analyzer is not None:
            # Process the spectrum
            result = self.analyzer.process_spectrum(self.spectrum)

            # Create the plot
            self.figure.clear()
            self.analyzer.plot_spectrum(self.spectrum, result, show=False)
            self.canvas.draw()


# --- Main GUI Application ---
class ArcheryPerformanceGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Archery Performance Analyzer")

        # Initialize data processor
        self.data_processor = SensorDataProcessor(buffer_size=2000)

        # Initialize I-MON data buffers
        self.imon_timestamps = []
        self.imon_wavelengths = []
        self.imon_strains = []
        self.current_spectrum = None
        self.imon_active = False
        self.spectral_analyzer = IMONSpectralAnalyzer()

        # Set up the UI
        self._setup_ui()

        # Start the serial reader thread
        self.serialThread = SerialReaderThread(port='COM5', baud=115200)
        self.serialThread.newData.connect(self.handle_new_data)
        self.serialThread.start()

        # Create the I-MON file monitor (will be started when path is set)
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

        # File path controls
        file_path_layout = QHBoxLayout()
        self.imon_path_edit = QLineEdit()
        self.imon_path_edit.setPlaceholderText("Path to I-MON data file")

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_imon_file)

        file_path_layout.addWidget(self.imon_path_edit, stretch=3)
        file_path_layout.addWidget(self.browse_button, stretch=1)

        # Strain conversion factor
        self.strain_factor_edit = QLineEdit("0.78")
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
        imon_config_layout.addRow("Data File:", file_path_layout)
        imon_config_layout.addRow("Strain Factor (µε/pm):", self.strain_factor_edit)
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
        """Create the shot analysis tab (placeholder for now)"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        layout.addWidget(QLabel("Shot analysis features will be added here."))

        return tab

    def _optimize_plot_settings(self):
        """Apply performance optimizations to all plots"""
        # Apply to all plot widgets
        for plot in [self.accel_plot, self.gyro_plot, self.strain_plot]:
            # Disable autoRange for better performance
            plot.enableAutoRange(x=False, y=False)

            # Set fixed ranges based on expected data
            plot.setXRange(0, 10)  # 10 seconds of data

            # Optimize rendering
            plot.setDownsampling(auto=True, mode='peak')
            plot.setClipToView(True)

            # Reduce update overhead
            plot.setMouseEnabled(x=True, y=True)  # Allow zoom/pan

        # Apply optimizations after creating plots - with better ranges
        self.accel_plot.setYRange(0, 35)  # Increased range to handle up to 35 m/s²
        self.gyro_plot.setYRange(-150, 150)  # Wider gyro range (-150 to 150 deg/s)
        self.strain_plot.setYRange(0, 1000)  # Initial strain range (0-1000 µε)

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
            spectrum = data['spectrum']

            # Store the data
            self.imon_timestamps.append(timestamp)
            self.imon_wavelengths.append(wavelength)
            self.imon_strains.append(strain)
            self.current_spectrum = spectrum  # Store most recent spectrum

            # Enable spectrum viewer button
            self.view_spectrum_button.setEnabled(True)

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
                strain_factor = 0.78  # Default

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
            dialog = SpectrumViewDialog(self, self.current_spectrum, self.spectral_analyzer)
            dialog.exec_()
        else:
            logging.warning("No spectrum data available to view")

    def reset_reference_spectrum(self):
        """Reset the reference spectrum for strain calculations"""
        self.imon_monitor.reset_reference()
        logging.info("Reset spectral reference")

    def load_imon_data(self):
        """Load and analyze I-MON spectral data from a file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open I-MON Data File", "", "Text Files (*.txt);;All Files (*)")

            if file_path:
                # Update the path field
                self.imon_path_edit.setText(file_path)

                # Clear existing data
                self.clear_imon_data()

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

                    # Update peak strain metric
                    if len(strain_values) > 0:
                        peak_strain = np.max(strain_values)
                        self.data_processor.metrics['peak_strain'] = peak_strain
                        self.peak_strain_label.setText(f"{peak_strain:.2f}")

                    logging.info(f"Loaded I-MON data from {file_path}")

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
        self.view_spectrum_button.setEnabled(False)
        self.strain_curve.setData([], [])
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

        window = ArcheryPerformanceGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.exception("Error in main application startup")


if __name__ == "__main__":
    main()