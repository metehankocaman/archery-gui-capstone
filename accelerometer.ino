/*
 * Complete MPU6050 Code for Archery Performance System
 * 
 * This code properly configures the MPU6050, performs calibration,
 * and streams acceleration and gyroscope data at maximum rate.
 * 
 * Connections:
 * MPU6050 ---- ESP32
 * VCC     ---- 3.3V
 * GND     ---- GND
 * SCL     ---- GPIO 22
 * SDA     ---- GPIO 21
 * INT     ---- Not Connected
 */

#include <Wire.h>
#include <MPU6050_tockn.h>

// Create MPU6050 object
MPU6050 mpu6050(Wire);

// Configuration
#define SDA_PIN 21
#define SCL_PIN 22
#define SERIAL_BAUD 115200
#define ENABLE_DEBUG false  // Set to true for debug output

// Variables for accelerometer calibration
float accel_cal_x = 0;
float accel_cal_y = 0;
float accel_cal_z = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(SERIAL_BAUD);
  
  // Wait for Serial to be ready
  delay(100);
  
  if (ENABLE_DEBUG) {
    Serial.println("Archery Performance Sensor System");
    Serial.println("Initializing MPU6050...");
  }
  
  // Initialize I2C communication
  Wire.begin(SDA_PIN, SCL_PIN);
  
  // Initialize MPU6050
  mpu6050.begin();
  
  // Set gyroscope range if needed (default is ±250 degrees/sec)
  // MPU6050_GYRO_FS_250 is default, options are 250, 500, 1000, 2000
  
  // Set accelerometer range if needed (default is ±2g)
  // MPU6050_ACCEL_FS_2 is default, options are 2, 4, 8, 16
  
  // Calibrate gyroscope - keep device still!
  if (ENABLE_DEBUG) {
    Serial.println("Calibrating gyroscope - keep device still...");
  }
  mpu6050.calcGyroOffsets(ENABLE_DEBUG);
  
  // Simple accelerometer calibration (assuming flat position)
  calibrateAccelerometer();
  
  if (ENABLE_DEBUG) {
    Serial.println("Setup complete");
    Serial.println("Format: ax,ay,az,gx,gy,gz");
  }
  
  // Brief delay to ensure everything is ready
  delay(200);
}

void loop() {
  // Update sensor readings
  mpu6050.update();
  
  // Get corrected accelerometer readings
  float ax = mpu6050.getAccX() - accel_cal_x;
  float ay = mpu6050.getAccY() - accel_cal_y;
  float az = mpu6050.getAccZ() - accel_cal_z; // Remove the +1.0 gravity adjustment
  
  // Get gyroscope readings (already calibrated by the library)
  float gx = mpu6050.getGyroX();
  float gy = mpu6050.getGyroY();
  float gz = mpu6050.getGyroZ();
  
  // Send data in comma-separated format: ax,ay,az,gx,gy,gz
  Serial.print(ax, 4); Serial.print(",");
  Serial.print(ay, 4); Serial.print(",");
  Serial.print(az, 4); Serial.print(",");
  Serial.print(gx, 2); Serial.print(",");
  Serial.print(gy, 2); Serial.print(",");
  Serial.println(gz, 2);

  // Debug output if enabled
  if (ENABLE_DEBUG) {
    // Calculate acceleration magnitude (in g)
    float accel_mag = sqrt(ax*ax + ay*ay + az*az);
    
    // Convert to m/s²
    float accel_mag_ms2 = accel_mag * 9.81;
    
    Serial.print("Accel Mag (g): ");
    Serial.print(accel_mag, 4);
    Serial.print("  Accel Mag (m/s²): ");
    Serial.println(accel_mag_ms2, 4);
  }

  // Minimal delay for maximum transmission rate
  // The MPU6050 has a max sample rate of 1kHz
  delay(1);
}

void calibrateAccelerometer() {
  if (ENABLE_DEBUG) {
    Serial.println("Calibrating accelerometer - place device flat...");
  }
  
  // Take multiple readings for accuracy
  const int numSamples = 100;
  float sumX = 0, sumY = 0, sumZ = 0;
  
  for (int i = 0; i < numSamples; i++) {
    mpu6050.update();
    sumX += mpu6050.getAccX();
    sumY += mpu6050.getAccY();
    sumZ += mpu6050.getAccZ();
    delay(10);
  }
  
  // Calculate offsets (assuming Z should be 1g when flat)
  accel_cal_x = sumX / numSamples;
  accel_cal_y = sumY / numSamples;
  accel_cal_z = (sumZ / numSamples) - 1.0; // Subtract 1g to account for gravity
  
  if (ENABLE_DEBUG) {
    Serial.println("Accelerometer calibration complete");
    Serial.print("Offsets: X=");
    Serial.print(accel_cal_x, 4);
    Serial.print(" Y=");
    Serial.print(accel_cal_y, 4);
    Serial.print(" Z=");
    Serial.println(accel_cal_z, 4);
  }
}