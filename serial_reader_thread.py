# serial_reader_thread.py (Updated with Timestamps)
from PyQt5.QtCore import QThread, pyqtSignal
from accel_comm import AccelReader
import time
import datetime


class SerialReaderThread(QThread):
    """
    A dedicated QThread that continuously reads data from the accelerometer (ESP32-MPU6050)
    via the serial port and emits a signal (newData) each time a valid data tuple is received.

    The new data tuple format includes a timestamp:
    (timestamp, ax, ay, az, gx, gy, gz)
    where timestamp is a datetime object and all other values are floats.
    """
    newData = pyqtSignal(tuple)  # Signal emitting a tuple of timestamp and sensor values

    def __init__(self, port, baud=115200, parent=None):
        super().__init__(parent)
        self.reader = AccelReader(port, baud, timeout=0.001)  # Shorter timeout
        self._running = True

    def run(self):
        # Continuously read from the serial port and emit valid data
        while self._running:
            data = self.reader.read_data()
            if data is not None:
                # Add timestamp (current date and time) to the data
                current_datetime = datetime.datetime.now()

                # Create a new tuple with timestamp as the first element
                # Original data is (ax, ay, az, gx, gy, gz)
                # New data is (timestamp, ax, ay, az, gx, gy, gz)
                timestamped_data = (current_datetime,) + data

                self.newData.emit(timestamped_data)
            else:
                # Yield to other threads briefly without blocking for too long
                self.msleep(1)  # 1ms sleep instead of 5ms

    def stop(self):
        """Stop the reading thread"""
        self._running = False
        self.wait()