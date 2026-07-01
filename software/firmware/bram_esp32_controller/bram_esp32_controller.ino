#include <Arduino.h>

#define BRAM_ENABLE_BLUEPAD32 1

#if BRAM_ENABLE_BLUEPAD32
#include <Bluepad32.h>
#endif

#include "bram_controller.hpp"

// Demo input switch:
// false = Xbox/gamepad over BLE via Bluepad32
// true  = serial commands like: f 0.5 y -0.3
static constexpr bool kUseSerialInput = true;

static constexpr bool kPrintDebug = true;
static constexpr bool kUseImuCorrection = false;

// when i wire
static constexpr int kFrontServoPin = 4;
static constexpr int kBackLeftServoPin = 5;
static constexpr int kBackRightServoPin = 18;

static constexpr int kServoPins[3] = {
    kFrontServoPin,
    kBackLeftServoPin,
    kBackRightServoPin,
};

// Tune these after checking the linkage. Start conservative if anything binds.
static constexpr int kServoMinUs[3] = {1000, 1000, 1000};
static constexpr int kServoNeutralUs[3] = {1500, 1500, 1500};
static constexpr int kServoMaxUs[3] = {2000, 2000, 2000};
static constexpr bool kServoInvert[3] = {false, false, false};

static constexpr int kPwmHz = 50;
static constexpr int kPwmResolutionBits = 16;
static constexpr int kPwmChannels[3] = {0, 1, 2};
static constexpr uint32_t kControlPeriodMs = 20;
static constexpr uint32_t kInputTimeoutMs = 600;
static constexpr float kDeadband = 0.08f;
static constexpr float kCommandSlewPerTick = 0.08f;

bram::BramController controller;

float targetForward = 0.0f;
float targetYaw = 0.0f;
float commandForward = 0.0f;
float commandYaw = 0.0f;
uint32_t lastInputMs = 0;
uint32_t lastControlMs = 0;
uint32_t stepIndex = 0;

#if BRAM_ENABLE_BLUEPAD32
ControllerPtr gamepads[BP32_MAX_GAMEPADS];

void onConnectedController(ControllerPtr ctl) {
  for (int i = 0; i < BP32_MAX_GAMEPADS; ++i) {
    if (gamepads[i] == nullptr) {
      gamepads[i] = ctl;
      if (kPrintDebug) {
        Serial.printf("Gamepad connected slot=%d\n", i);
      }
      return;
    }
  }
  if (kPrintDebug) {
    Serial.println("Gamepad connected but no slot available");
  }
}

void onDisconnectedController(ControllerPtr ctl) {
  for (int i = 0; i < BP32_MAX_GAMEPADS; ++i) {
    if (gamepads[i] == ctl) {
      gamepads[i] = nullptr;
      if (kPrintDebug) {
        Serial.printf("Gamepad disconnected slot=%d\n", i);
      }
      targetForward = 0.0f;
      targetYaw = 0.0f;
      return;
    }
  }
}
#endif

float clipFloat(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

float applyDeadband(float value) {
  value = clipFloat(value, -1.0f, 1.0f);
  if (fabsf(value) < kDeadband) return 0.0f;
  const float sign = value < 0.0f ? -1.0f : 1.0f;
  return sign * ((fabsf(value) - kDeadband) / (1.0f - kDeadband));
}

float slew(float current, float target) {
  const float delta = clipFloat(target - current, -kCommandSlewPerTick, kCommandSlewPerTick);
  return current + delta;
}

int actionToPulseUs(float action, int servoIndex) {
  action = clipFloat(action, -1.0f, 1.0f);
  if (kServoInvert[servoIndex]) {
    action = -action;
  }
  const int neutral = kServoNeutralUs[servoIndex];
  if (action >= 0.0f) {
    return neutral + static_cast<int>((kServoMaxUs[servoIndex] - neutral) * action);
  }
  return neutral + static_cast<int>((neutral - kServoMinUs[servoIndex]) * action);
}

uint32_t pulseUsToDuty(int pulseUs) {
  static constexpr uint32_t maxDuty = (1UL << kPwmResolutionBits) - 1UL;
  static constexpr float periodUs = 1000000.0f / static_cast<float>(kPwmHz);
  return static_cast<uint32_t>(
      clipFloat((static_cast<float>(pulseUs) / periodUs) * maxDuty, 0.0f, maxDuty));
}

void attachServoPwm() {
  for (int i = 0; i < 3; ++i) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(kServoPins[i], kPwmHz, kPwmResolutionBits);
#else
    ledcSetup(kPwmChannels[i], kPwmHz, kPwmResolutionBits);
    ledcAttachPin(kServoPins[i], kPwmChannels[i]);
#endif
  }
}

void writeServoPulse(int servoIndex, int pulseUs) {
  const uint32_t duty = pulseUsToDuty(pulseUs);
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(kServoPins[servoIndex], duty);
#else
  ledcWrite(kPwmChannels[servoIndex], duty);
#endif
}

void writeServos(const bram::ServoAction& action) {
  for (int i = 0; i < 3; ++i) {
    writeServoPulse(i, actionToPulseUs(action.values[i], i));
  }
}

bool parseSerialCommand(const char* line, float& forward, float& yaw) {
  if (strcmp(line, "stop") == 0 || strcmp(line, "s") == 0) {
    forward = 0.0f;
    yaw = 0.0f;
    return true;
  }
  if (sscanf(line, "f %f y %f", &forward, &yaw) == 2) {
    return true;
  }
  if (sscanf(line, "%f %f", &forward, &yaw) == 2) {
    return true;
  }
  return false;
}

void updateSerialInput() {
  static char line[80];
  static size_t length = 0;
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      if (length > 0) {
        line[length] = '\0';
        float forward = 0.0f;
        float yaw = 0.0f;
        if (parseSerialCommand(line, forward, yaw)) {
          targetForward = clipFloat(forward, -1.0f, 1.0f);
          targetYaw = clipFloat(yaw, -1.0f, 1.0f);
          lastInputMs = millis();
          if (kPrintDebug) {
            Serial.printf("serial forward=%.3f yaw=%.3f\n", targetForward, targetYaw);
          }
        } else if (kPrintDebug) {
          Serial.printf("bad command: %s\n", line);
        }
      }
      length = 0;
    } else if (length + 1 < sizeof(line)) {
      line[length++] = c;
    }
  }
}

#if BRAM_ENABLE_BLUEPAD32
ControllerPtr activeGamepad() {
  for (int i = 0; i < BP32_MAX_GAMEPADS; ++i) {
    if (gamepads[i] && gamepads[i]->isConnected() && gamepads[i]->hasData()) {
      return gamepads[i];
    }
  }
  return nullptr;
}

void updateBleInput() {
  BP32.update();
  ControllerPtr ctl = activeGamepad();
  if (!ctl) return;

  // Xbox-style mapping: left stick vertical = forward/back, right stick horizontal = yaw.
  targetForward = applyDeadband(-static_cast<float>(ctl->axisY()) / 512.0f);
  targetYaw = applyDeadband(static_cast<float>(ctl->axisRX()) / 512.0f);
  lastInputMs = millis();

  if (ctl->b()) {
    targetForward = 0.0f;
    targetYaw = 0.0f;
  }
}
#endif

void setupInput() {
  Serial.begin(115200);
  delay(250);
  Serial.println("Bram ESP32 controller boot");
  Serial.println("Serial command format: f 0.5 y -0.3   or: 0.5 -0.3   or: stop");

#if BRAM_ENABLE_BLUEPAD32
  if (!kUseSerialInput) {
    BP32.setup(&onConnectedController, &onDisconnectedController);
    BP32.enableVirtualDevice(false);
    Serial.println("Bluepad32 BLE gamepad input enabled");
  }
#else
  Serial.println("Bluepad32 disabled at compile time; serial input only");
#endif
}

void updateInput() {
  if (kUseSerialInput) {
    updateSerialInput();
    return;
  }
#if BRAM_ENABLE_BLUEPAD32
  updateBleInput();
#endif
}

void setup() {
  setupInput();
  attachServoPwm();
  bram::ServoAction neutral{};
  writeServos(neutral);
  lastInputMs = millis();
  lastControlMs = millis();
}

void loop() {
  updateInput();

  const uint32_t now = millis();
  if (now - lastControlMs < kControlPeriodMs) {
    delay(1);
    return;
  }
  lastControlMs += kControlPeriodMs;

  if (now - lastInputMs > kInputTimeoutMs) {
    targetForward = 0.0f;
    targetYaw = 0.0f;
  }

  commandForward = slew(commandForward, targetForward);
  commandYaw = slew(commandYaw, targetYaw);

  float headingError = 0.0f;
  float yawRate = 0.0f;
  if (kUseImuCorrection) {
    // Fill this in once the 9DoF IMU driver is wired:
    // headingError = desired_heading - measured_heading;
    // yawRate = measured_gyro_z_rad_s;
  }

  const bram::ServoAction servoAction =
      controller.action(commandForward, commandYaw, stepIndex, headingError, yawRate);
  writeServos(servoAction);
  ++stepIndex;

  if (kPrintDebug && stepIndex % 25 == 0) {
    Serial.printf("cmd f=%.2f y=%.2f action=[%.3f %.3f %.3f]\n",
                  commandForward,
                  commandYaw,
                  servoAction.values[0],
                  servoAction.values[1],
                  servoAction.values[2]);
  }
}
