#include <Arduino.h>

#ifndef BRAM_ENABLE_BLUEPAD32
#define BRAM_ENABLE_BLUEPAD32 0
#endif

#ifndef BRAM_ENABLE_BLE_COMMANDS
#define BRAM_ENABLE_BLE_COMMANDS 1
#endif

#if BRAM_ENABLE_BLUEPAD32
#include <Bluepad32.h>
#endif

#if BRAM_ENABLE_BLE_COMMANDS
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#endif

#include "bram_controller.hpp"

float clipFloat(float value, float low, float high);

// Demo input switch. Leave native BLE commands as the default Arduino path;
// flip this to true for USB serial bring-up commands like: f 0.5 y -0.3.
static constexpr bool kUseSerialInput = false;
static constexpr bool kAllowSerialOverride = true;

static constexpr bool kPrintDebug = true;
static constexpr bool kUseImuCorrection = false;

static const char* kBleDeviceName = "BRAM";
static const char* kBleServiceUuid = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
static const char* kBleRxUuid = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
static const char* kBleTxUuid = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

static constexpr float kGamepadAxisScale = 512.0f;
static constexpr bool kInvertGamepadForward = true;
static constexpr bool kInvertGamepadYaw = false;

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
bool bleControllerConnected = false;
bool bleCommandConnected = false;

void setTargetCommand(float forward, float yaw, const char* source) {
  targetForward = clipFloat(forward, -1.0f, 1.0f);
  targetYaw = clipFloat(yaw, -1.0f, 1.0f);
  lastInputMs = millis();
  if (kPrintDebug) {
    Serial.printf("%s forward=%.3f yaw=%.3f\n", source, targetForward, targetYaw);
  }
}

#if BRAM_ENABLE_BLUEPAD32
ControllerPtr gamepads[BP32_MAX_GAMEPADS];

void onConnectedController(ControllerPtr ctl) {
  for (int i = 0; i < BP32_MAX_GAMEPADS; ++i) {
    if (gamepads[i] == nullptr) {
      gamepads[i] = ctl;
      if (kPrintDebug) {
        Serial.printf("Gamepad connected slot=%d\n", i);
      }
      bleControllerConnected = true;
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
      bleControllerConnected = false;
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

void handleCommandLine(const char* line, const char* source) {
  float forward = 0.0f;
  float yaw = 0.0f;
  if (parseSerialCommand(line, forward, yaw)) {
    setTargetCommand(forward, yaw, source);
  } else if (kPrintDebug) {
    Serial.printf("bad %s command: %s\n", source, line);
  }
}

void updateSerialInput() {
  static char line[80];
  static size_t length = 0;
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      if (length > 0) {
        line[length] = '\0';
        handleCommandLine(line, "serial");
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
    if (gamepads[i] && gamepads[i]->isConnected()) {
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
  const float forwardSign = kInvertGamepadForward ? -1.0f : 1.0f;
  const float yawSign = kInvertGamepadYaw ? -1.0f : 1.0f;
  const float forward =
      applyDeadband(forwardSign * static_cast<float>(ctl->axisY()) / kGamepadAxisScale);
  const float yaw =
      applyDeadband(yawSign * static_cast<float>(ctl->axisRX()) / kGamepadAxisScale);
  setTargetCommand(forward, yaw, "gamepad");

  if (ctl->b()) {
    setTargetCommand(0.0f, 0.0f, "gamepad");
  }
}
#endif

#if BRAM_ENABLE_BLE_COMMANDS
BLECharacteristic* bleTxCharacteristic = nullptr;

class BramBleServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer*) override {
    bleCommandConnected = true;
    if (kPrintDebug) {
      Serial.println("BLE command client connected");
    }
  }

  void onDisconnect(BLEServer* server) override {
    bleCommandConnected = false;
    setTargetCommand(0.0f, 0.0f, "ble");
    server->startAdvertising();
    if (kPrintDebug) {
      Serial.println("BLE command client disconnected; advertising restarted");
    }
  }
};

class BramBleRxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) override {
    String value = characteristic->getValue();
    if (value.length() == 0) return;

    char line[80];
    const size_t count =
        value.length() < sizeof(line) - 1 ? value.length() : sizeof(line) - 1;
    memcpy(line, value.c_str(), count);
    line[count] = '\0';
    for (size_t i = 0; i < count; ++i) {
      if (line[i] == '\n' || line[i] == '\r') {
        line[i] = '\0';
        break;
      }
    }

    handleCommandLine(line, "ble");
    if (bleTxCharacteristic) {
      char reply[80];
      snprintf(reply, sizeof(reply), "f %.3f y %.3f\n", targetForward, targetYaw);
      bleTxCharacteristic->setValue(reply);
      bleTxCharacteristic->notify();
    }
  }
};

void setupBleCommands() {
  BLEDevice::init(kBleDeviceName);
  BLEServer* server = BLEDevice::createServer();
  server->setCallbacks(new BramBleServerCallbacks());

  BLEService* service = server->createService(kBleServiceUuid);
  BLECharacteristic* rxCharacteristic = service->createCharacteristic(
      kBleRxUuid, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  bleTxCharacteristic = service->createCharacteristic(
      kBleTxUuid, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);

  rxCharacteristic->setCallbacks(new BramBleRxCallbacks());
  bleTxCharacteristic->addDescriptor(new BLE2902());
  bleTxCharacteristic->setValue("BRAM ready\n");

  service->start();
  BLEAdvertising* advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(kBleServiceUuid);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  Serial.println("Input mode: native BLE command service");
  Serial.println("BLE device name: BRAM");
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
    Serial.println("Input mode: Bluepad32 BLE gamepad");
    if (kAllowSerialOverride) {
      Serial.println("Serial override enabled; send stop or f/y commands over USB.");
    }
  } else {
    Serial.println("Input mode: USB serial");
  }
#elif BRAM_ENABLE_BLE_COMMANDS
  if (!kUseSerialInput) {
    setupBleCommands();
    if (kAllowSerialOverride) {
      Serial.println("Serial override enabled; send stop or f/y commands over USB.");
    }
  } else {
    Serial.println("Input mode: USB serial");
  }
#else
  Serial.println("Bluepad32 disabled at compile time; serial input only");
#endif
}

void updateInput() {
  if (kUseSerialInput || kAllowSerialOverride) {
    updateSerialInput();
  }
  if (kUseSerialInput) return;
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
    Serial.printf("cmd f=%.2f y=%.2f ble=%d action=[%.3f %.3f %.3f]\n",
                  commandForward,
                  commandYaw,
                  (bleControllerConnected || bleCommandConnected) ? 1 : 0,
                  servoAction.values[0],
                  servoAction.values[1],
                  servoAction.values[2]);
  }
}
