# Archery Performance Analyzer

## Overview
This repository contains the software components for the "Fiber Bragg Grating Feedback System for Recurve Bows" capstone project developed at Carleton University. The system combines optical fiber strain sensing (FBG) with motion tracking (IMU) to provide comprehensive performance feedback for archers.

The Archery Performance Analyzer captures, processes, and visualizes bow strain and motion data during the archery shot cycle, offering insights into draw consistency, release dynamics, and form stability that are not visible to the naked eye.

## Repository Structure
The repository contains the following main components:

- **GUI and Visualization**: Real-time interface for data display
- **Motion Processing**: Algorithms for IMU data analysis and shot detection
- **Spectral Analysis**: Tools for FBG strain data processing
- **Hardware Communication**: Serial and file-based data acquisition

## Key Components

### Core Application
- `archery_gui.py` / `archery_gui_xlsx.py`: Main application windows and visualization interfaces
- `data_processor.py`: Motion data processing, filtering, and shot detection
- `orientation_cube.py`: 3D bow orientation visualization

### Hardware Communication
- `serial_reader_thread.py`: Multi-threaded communication with ESP32
- `accel_comm.py`: Low-level communication with MPU6050 via ESP32

### FBG Spectral Processing
- `imon_file_monitor.py` / `imon_xlsx_monitor.py`: File monitoring threads for I-MON data
- `imon_spectral.py` / `imon_xlsx_parser.py`: Spectral analysis and peak detection
- `enhanced_spectrum_dialog.py`: Detailed FBG spectrum visualization

### Utilities
- `tab_delimited_reader.py`: Parser for non-standard file formats
- `txt_to_xlsx_converter.py`: Format conversion utility

## Installation

### Prerequisites
- Python 3.8 or higher
- PyQt5
- PyQtGraph
- NumPy
- Matplotlib
- SciPy
- pyserial

### Dependencies Installation
```bash
pip install PyQt5 pyqtgraph numpy matplotlib scipy pyserial
```

### ESP32 Firmware
The `accelerometer.ino` sketch must be uploaded to an ESP32 with MPU6050 connected via I²C:
- ESP32 GPIO21 → MPU6050 SDA
- ESP32 GPIO22 → MPU6050 SCL
- ESP32 3.3V → MPU6050 VCC
- ESP32 GND → MPU6050 GND

## Usage

### Basic Operation
1. Connect the ESP32 to your computer via USB
2. Run the application:
   ```
   python main.py
   ```
3. Select the appropriate COM port for your ESP32
4. For strain data, set the I-MON file path in the GUI
5. Click "Start Monitoring" to begin data acquisition
6. Recorded sessions can be exported using the "Export Session" button

### Shot Analysis
The system automatically detects and analyzes shot phases:
- **Draw**: Initial bow loading phase
- **Release**: Arrow departure and impulse
- **Recovery**: Return to steady state

Key metrics calculated include:
- Draw duration (seconds)
- Release impulse (m/s²)
- Recovery time (seconds)
- Peak strain (μɛ)

## Hardware Requirements

### Required Components
- ESP32-WROOM-32 development board
- MPU6050 6-axis IMU
- Recurve bow with FBG sensor mounting points
- FBG sensors (wavelengths ~1543nm and ~1534nm)
- I-MON interrogator (or compatible spectral analyzer)
- Superluminescent LED source (SLED)
- Single-mode fiber optic components

## Contributors
This project was developed by Team 15 from the Department of Electronics at Carleton University:
- Metehan Kocaman: Software and GUI Development
- Yashil Thakore: Mechanical Design and Integration
- Andrew Pignatelli: Optical System and Interrogation

Supervised by Professor Christopher Smelser

## License
This project is provided for educational and research purposes.
