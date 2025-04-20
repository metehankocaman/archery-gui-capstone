# enhanced_spectrum_dialog.py - Enhanced FBG Spectrum Viewer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import logging


class EnhancedSpectrumViewDialog(QDialog):
    """Enhanced dialog for displaying FBG spectral data with XLSX support"""

    def __init__(self, parent=None, spectrum=None, analyzer=None, xlsx_mode=False, processed_data=None):
        super().__init__(parent)
        self.setWindowTitle("Enhanced FBG Spectrum Viewer")
        self.resize(900, 700)
        self.spectrum = spectrum
        self.xlsx_mode = xlsx_mode
        self.analyzer = analyzer
        self.processed_data = processed_data

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create matplotlib figure and canvas
        self.figure = plt.figure(figsize=(9, 7))
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
            self.figure.clear()
            
            if self.xlsx_mode:
                # XLSX mode uses the enhanced plotting
                if self.processed_data:
                    self.analyzer.plot_spectrum(self.spectrum, self.processed_data, show=False)
                else:
                    # Process the data if not already provided
                    processed_data = self.analyzer.process_spectrum(self.spectrum)
                    self.analyzer.plot_spectrum(self.spectrum, processed_data, show=False)
            else:
                # Original TXT mode
                result = self.analyzer.process_spectrum(self.spectrum)
                self.analyzer.plot_spectrum(self.spectrum, result, show=False)
                
            self.canvas.draw()
            logging.info("Spectrum plot updated")