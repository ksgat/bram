#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_BNO08x.h>

// Seeed XIAO ESP32-C3 Arduino variants normally define D4/D5. These fallbacks
// match the XIAO ESP32-C3 GPIO mapping if a board package omits the aliases.
#ifndef D4
#define D4 6
#endif

#ifndef D5
#define D5 7
#endif

static constexpr int kSdaPin = D4;
static constexpr int kSclPin = D5;
static constexpr int kBno08xResetPin = -1;
static constexpr uint32_t kSerialBaud = 115200;
static constexpr uint32_t kI2cClockHz = 400000;
static constexpr uint32_t kReportIntervalUs = 10000;
static constexpr uint32_t kPrintPeriodMs = 100;

Adafruit_BNO08x bno08x(kBno08xResetPin);
sh2_SensorValue_t sensorValue;

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct ImuState {
  Vec3 accelMps2;
  Vec3 gyroRadps;
  Vec3 magUt;
  float quatI = 0.0f;
  float quatJ = 0.0f;
  float quatK = 0.0f;
  float quatReal = 1.0f;
  float rollDeg = 0.0f;
  float pitchDeg = 0.0f;
  float yawDeg = 0.0f;
  uint8_t rotationAccuracy = 0;
  bool hasAccel = false;
  bool hasGyro = false;
  bool hasMag = false;
  bool hasRotation = false;
};

ImuState imu;
uint32_t lastPrintMs = 0;

float radToDeg(float radians) {
  return radians * 180.0f / PI;
}

void updateEulerFromQuaternion() {
  const float qw = imu.quatReal;
  const float qx = imu.quatI;
  const float qy = imu.quatJ;
  const float qz = imu.quatK;

  const float sinrCosp = 2.0f * (qw * qx + qy * qz);
  const float cosrCosp = 1.0f - 2.0f * (qx * qx + qy * qy);
  imu.rollDeg = radToDeg(atan2f(sinrCosp, cosrCosp));

  const float sinp = 2.0f * (qw * qy - qz * qx);
  if (fabsf(sinp) >= 1.0f) {
    imu.pitchDeg = radToDeg(copysignf(PI / 2.0f, sinp));
  } else {
    imu.pitchDeg = radToDeg(asinf(sinp));
  }

  const float sinyCosp = 2.0f * (qw * qz + qx * qy);
  const float cosyCosp = 1.0f - 2.0f * (qy * qy + qz * qz);
  imu.yawDeg = radToDeg(atan2f(sinyCosp, cosyCosp));
}

void scanI2cBus() {
  Serial.println("Scanning I2C bus...");
  int found = 0;
  for (uint8_t address = 1; address < 127; ++address) {
    Wire.beginTransmission(address);
    const uint8_t error = Wire.endTransmission();
    if (error == 0) {
      Serial.printf("  found device at 0x%02X\n", address);
      ++found;
    }
  }
  if (found == 0) {
    Serial.println("  no I2C devices found; check SDA/SCL, power, and common ground");
  }
}

bool enableReport(sh2_SensorId_t reportId, const char* name) {
  if (!bno08x.enableReport(reportId, kReportIntervalUs)) {
    Serial.printf("Could not enable %s\n", name);
    return false;
  }
  Serial.printf("Enabled %s\n", name);
  return true;
}

void enableReports() {
  enableReport(SH2_ROTATION_VECTOR, "rotation vector");
  enableReport(SH2_ACCELEROMETER, "accelerometer");
  enableReport(SH2_GYROSCOPE_CALIBRATED, "gyroscope");
  enableReport(SH2_MAGNETIC_FIELD_CALIBRATED, "magnetometer");
}

void updateStateFromSensorEvent(const sh2_SensorValue_t& value) {
  switch (value.sensorId) {
    case SH2_ROTATION_VECTOR:
      imu.quatI = value.un.rotationVector.i;
      imu.quatJ = value.un.rotationVector.j;
      imu.quatK = value.un.rotationVector.k;
      imu.quatReal = value.un.rotationVector.real;
      imu.rotationAccuracy = value.status;
      imu.hasRotation = true;
      updateEulerFromQuaternion();
      break;
    case SH2_ACCELEROMETER:
      imu.accelMps2.x = value.un.accelerometer.x;
      imu.accelMps2.y = value.un.accelerometer.y;
      imu.accelMps2.z = value.un.accelerometer.z;
      imu.hasAccel = true;
      break;
    case SH2_GYROSCOPE_CALIBRATED:
      imu.gyroRadps.x = value.un.gyroscope.x;
      imu.gyroRadps.y = value.un.gyroscope.y;
      imu.gyroRadps.z = value.un.gyroscope.z;
      imu.hasGyro = true;
      break;
    case SH2_MAGNETIC_FIELD_CALIBRATED:
      imu.magUt.x = value.un.magneticField.x;
      imu.magUt.y = value.un.magneticField.y;
      imu.magUt.z = value.un.magneticField.z;
      imu.hasMag = true;
      break;
  }
}

void printImuState() {
  Serial.printf(
      "rpy_deg=[%7.2f %7.2f %7.2f] quat=[%.4f %.4f %.4f %.4f] acc=%u\n",
      imu.rollDeg,
      imu.pitchDeg,
      imu.yawDeg,
      imu.quatReal,
      imu.quatI,
      imu.quatJ,
      imu.quatK,
      imu.rotationAccuracy);

  Serial.printf("  accel_mps2=[%7.3f %7.3f %7.3f] gyro_radps=[%7.3f %7.3f %7.3f]\n",
                imu.accelMps2.x,
                imu.accelMps2.y,
                imu.accelMps2.z,
                imu.gyroRadps.x,
                imu.gyroRadps.y,
                imu.gyroRadps.z);

  Serial.printf("  mag_uT=[%7.2f %7.2f %7.2f] reports=[rot:%d accel:%d gyro:%d mag:%d]\n",
                imu.magUt.x,
                imu.magUt.y,
                imu.magUt.z,
                imu.hasRotation ? 1 : 0,
                imu.hasAccel ? 1 : 0,
                imu.hasGyro ? 1 : 0,
                imu.hasMag ? 1 : 0);
}

void setup() {
  Serial.begin(kSerialBaud);
  delay(1500);

  Serial.println();
  Serial.println("BNO080/BNO085 IMU test for XIAO ESP32-C3");
  Serial.printf("I2C pins: SDA=D4/GPIO%d, SCL=D5/GPIO%d\n", kSdaPin, kSclPin);

  Wire.begin(kSdaPin, kSclPin);
  Wire.setClock(kI2cClockHz);
  scanI2cBus();

  if (!bno08x.begin_I2C(BNO08x_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("BNO08x not found at 0x4A. Some boards use 0x4B; trying that next.");
    if (!bno08x.begin_I2C(0x4B, &Wire)) {
      Serial.println("BNO08x not found. Check VIN/3V3, GND, SDA on D4, and SCL on D5.");
      while (true) {
        delay(1000);
      }
    }
  }

  Serial.println("BNO08x detected");
  enableReports();
}

void loop() {
  if (bno08x.wasReset()) {
    Serial.println("BNO08x reset detected; re-enabling reports");
    enableReports();
  }

  while (bno08x.getSensorEvent(&sensorValue)) {
    updateStateFromSensorEvent(sensorValue);
  }

  const uint32_t now = millis();
  if (now - lastPrintMs >= kPrintPeriodMs) {
    lastPrintMs = now;
    printImuState();
  }
}
