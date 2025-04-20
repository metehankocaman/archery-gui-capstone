# accel_comm.py (Improved Error Handling)
import serial
import logging
import re

class AccelReader:
    def __init__(self, port, baud=115200, timeout=0.001):
        try:
            self.ser = serial.Serial(port, baud, timeout=timeout)
            # Flush any boot messages to start with a clean buffer
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except serial.serialutil.SerialException as e:
            logging.error(f"Warning: Could not open serial port: {e}")
            self.ser = None
        
        # Prepare for line reading
        self.buffer = bytearray()
        
        # Pattern to match valid data lines (6 comma-separated numbers)
        self.valid_pattern = re.compile(r'^-?\d+(\.\d+)?(,-?\d+(\.\d+)?){5}$')
        
        # Known debug messages to silently ignore
        self.known_debug = ['clk_drv:0x00', '===========================']

    def read_data(self):
        """
        Reads a line from the serial port and returns parsed sensor values as a tuple:
        (ax, ay, az, gx, gy, gz).
        Only processes lines that contain exactly five commas and are valid numbers.
        """
        if not self.ser:
            return None
            
        try:
            # Check if data is available
            if self.ser.in_waiting > 0:
                # Read all available data
                new_data = self.ser.read(self.ser.in_waiting)
                
                # Add to our buffer
                self.buffer.extend(new_data)
                
                # Look for complete lines
                if b'\n' in self.buffer:
                    # Extract the first complete line
                    line_end = self.buffer.find(b'\n')
                    line = self.buffer[:line_end].decode('utf-8', errors='ignore').strip()
                    
                    # Remove processed data from buffer
                    self.buffer = self.buffer[line_end+1:]
                    
                    # Skip known debug messages
                    if any(debug in line for debug in self.known_debug):
                        return None
                    
                    # Check if line matches our expected format
                    if self.valid_pattern.match(line):
                        try:
                            # Extra validation to ensure we can parse all values as floats
                            values = [float(x) for x in line.split(",")]
                            if len(values) == 6:
                                return tuple(values)
                        except ValueError:
                            # If we can't convert to float, silently ignore
                            pass
            
        except Exception as e:
            logging.error(f"Error parsing serial data: {e}")
            # Clear buffer on error to resynchronize
            self.buffer = bytearray()
            
        return None