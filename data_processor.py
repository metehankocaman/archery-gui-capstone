# data_processor.py (Updated with Timestamps)
import math
import numpy as np
from collections import deque
import datetime


class SensorDataProcessor:
    """Process raw sensor data and detect archery shot events"""

    def __init__(self, buffer_size=1000):
        # Main data buffers
        self.timestamps = deque(maxlen=buffer_size)  # Relative time (seconds since start)
        self.real_timestamps = deque(maxlen=buffer_size)  # Actual datetime objects
        self.accel_raw = deque(maxlen=buffer_size)  # Raw accelerometer values
        self.gyro_raw = deque(maxlen=buffer_size)  # Raw gyroscope values
        self.accel_mag = deque(maxlen=buffer_size)  # Computed acceleration magnitude
        self.orientation = deque(maxlen=buffer_size)  # roll, pitch, yaw

        # Shot detection state
        self.shot_in_progress = False
        self.shot_start_time = None
        self.shot_release_detected = False
        self.shot_release_time = None

        # Metrics for the last detected shot
        self.metrics = {
            'draw_duration': None,
            'release_impulse': None,
            'recovery_time': None,
            'peak_strain': None
        }

        # For complementary filter
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.last_time = None
        self.first_timestamp = None

        # Shot detection thresholds (you'll need to calibrate these)
        self.DRAW_ACCEL_THRESHOLD = 1.2  # m/s^2, sustained increase during draw
        self.RELEASE_ACCEL_THRESHOLD = 2.5  # m/s^2, spike during release
        self.RECOVERY_THRESHOLD = 0.3  # m/s^2, closeness to baseline

    def process_data(self, real_timestamp, accel_data, gyro_data):
        """
        Process incoming sensor data.

        Args:
            real_timestamp (datetime): actual datetime when data was received
            accel_data (tuple): (ax, ay, az) accelerometer data in m/s^2
            gyro_data (tuple): (gx, gy, gz) gyroscope data in deg/s

        Returns:
            dict: Processed data including orientation and shot metrics
        """
        ax, ay, az = accel_data
        gx, gy, gz = gyro_data

        # Store real timestamp
        self.real_timestamps.append(real_timestamp)

        # Store and calculate relative timestamp
        if not self.timestamps:
            # First data point - initialize reference time
            self.first_timestamp = real_timestamp
            self.timestamps.append(0.0)
        else:
            # Calculate seconds from first timestamp
            time_diff = (real_timestamp - self.first_timestamp).total_seconds()
            self.timestamps.append(time_diff)

        # Store raw data
        self.accel_raw.append(accel_data)
        self.gyro_raw.append(gyro_data)

        # Calculate acceleration magnitude (for shot detection)
        accel_mag = math.sqrt(ax ** 2 + ay ** 2 + az ** 2) * 9.81
        self.accel_mag.append(accel_mag)

        # Get current relative time for filter
        current_time = self.timestamps[-1]

        # Apply complementary filter to get stable orientation
        if self.last_time is not None:
            dt = current_time - self.last_time
            self.roll, self.pitch, self.yaw = self._complementary_filter(
                (ax, ay, az), (gx, gy, gz), dt)
        self.last_time = current_time

        # Store computed orientation
        self.orientation.append((self.roll, self.pitch, self.yaw))

        # Detect shot events (draw, release, recovery)
        self._detect_shot_events(current_time, accel_mag)

        # Return the latest processed data
        return {
            'accel_mag': accel_mag,
            'orientation': (self.roll, self.pitch, self.yaw),
            'metrics': self.metrics,
            'shot_state': 'drawing' if self.shot_in_progress else
            'released' if self.shot_release_detected else 'idle'
        }

    def _complementary_filter(self, accel_data, gyro_data, dt, alpha=0.98):
        """
        Fuse accelerometer and gyroscope data for stable orientation.

        Args:
            accel_data (tuple): (ax, ay, az) accelerometer data
            gyro_data (tuple): (gx, gy, gz) gyroscope data
            dt (float): time step in seconds
            alpha (float): filter coefficient (0.9-0.98 works well)

        Returns:
            tuple: (roll, pitch, yaw) in degrees
        """
        ax, ay, az = accel_data
        gx, gy, gz = gyro_data

        # Calculate angles from accelerometer
        accel_roll = math.atan2(ay, az) * 180.0 / math.pi
        accel_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi

        # Integrate gyroscope data
        gyro_roll = self.roll + gx * dt
        gyro_pitch = self.pitch + gy * dt
        gyro_yaw = self.yaw + gz * dt  # yaw from gyro integration only

        # Complementary filter
        roll = alpha * gyro_roll + (1 - alpha) * accel_roll
        pitch = alpha * gyro_pitch + (1 - alpha) * accel_pitch
        yaw = gyro_yaw  # yaw comes from gyro only

        return roll, pitch, yaw

    def _detect_shot_events(self, timestamp, accel_mag):
        """
        Detect archery shot events using acceleration magnitude.

        Args:
            timestamp (float): current time in seconds
            accel_mag (float): acceleration magnitude
        """
        # Simple threshold-based detection (will need calibration)

        # 1. Detect draw start
        if not self.shot_in_progress and not self.shot_release_detected:
            if accel_mag > self.DRAW_ACCEL_THRESHOLD:
                self.shot_in_progress = True
                self.shot_start_time = timestamp
                print(f"Draw detected at {timestamp:.2f}s")

        # 2. Detect release
        elif self.shot_in_progress and not self.shot_release_detected:
            if accel_mag > self.RELEASE_ACCEL_THRESHOLD:
                self.shot_release_detected = True
                self.shot_release_time = timestamp

                # Calculate draw duration
                draw_duration = self.shot_release_time - self.shot_start_time
                self.metrics['draw_duration'] = draw_duration

                # Calculate release impulse (using last 5 readings for max)
                recent_accel = list(self.accel_mag)[-5:]
                release_impulse = max(recent_accel)
                self.metrics['release_impulse'] = release_impulse

                print(f"Release detected at {timestamp:.2f}s. Draw duration: {draw_duration:.2f}s")

        # 3. Detect recovery (return to baseline)
        elif self.shot_release_detected:
            # Assume 9.8 m/s^2 (gravity) is the baseline
            baseline = 9.8
            if abs(accel_mag - baseline) < self.RECOVERY_THRESHOLD:
                # Calculate recovery time
                recovery_time = timestamp - self.shot_release_time
                self.metrics['recovery_time'] = recovery_time

                # Reset shot state for next shot
                self.shot_in_progress = False
                self.shot_release_detected = False

                print(f"Recovery detected. Total shot time: {recovery_time + self.metrics['draw_duration']:.2f}s")

    def get_current_metrics(self):
        """Get the current shot metrics"""
        return self.metrics

    def reset(self):
        """Reset all buffers and state"""
        self.timestamps.clear()
        self.real_timestamps.clear()
        self.accel_raw.clear()
        self.gyro_raw.clear()
        self.accel_mag.clear()
        self.orientation.clear()

        self.shot_in_progress = False
        self.shot_start_time = None
        self.shot_release_detected = False
        self.shot_release_time = None
        self.first_timestamp = None
        self.last_time = None

        self.metrics = {
            'draw_duration': None,
            'release_impulse': None,
            'recovery_time': None,
            'peak_strain': None
        }