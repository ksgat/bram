# Bram Firmware

The firmware runs movement_v2 actor policies online. It does not run MuJoCo,
PPO, or Python on-device; it runs exported tiny MLP actor weights from
`bram_esp32_controller/bram_policy_data.hpp` using the IMU quaternion and recent
servo-action history.

Movement V2 defaults to primitive arbitration for demos. If forward/back and yaw are both commanded at the same time, the sketch prioritizes translation and suppresses yaw. Set `kBlendMixedCommands = true` in `bram_esp32_controller/bram_esp32_controller.ino` only after the isolated primitives are reliable enough to re-enable mixed-arc blending.

## Layout

- `bram_esp32_controller/bram_esp32_controller.ino` is the Arduino sketch.
- `bram_esp32_controller/bram_controller.hpp` is the runtime controller logic.
- `bram_esp32_controller/bram_policy_controller.hpp` is the online actor runtime.
- `bram_esp32_controller/bram_policy_data.hpp` is generated actor weight data.
- `bram_esp32_controller/bram_controller_data.hpp` is generated fallback controller data.
- `bno08x_imu_test/bno08x_imu_test.ino` is a standalone BNO080/BNO085 IMU bring-up sketch for a XIAO ESP32-C3 with SDA on `D4` and SCL on `D5`.
- `tools/export_bram_firmware.py` regenerates the data header from a movement_v2 primitive bundle.

## Regenerate Policy Data

From the repo root:

```bash
.venv-rl/bin/python software/movement_v2/export_policy_header.py
```

`kUseOnlinePolicies = true` in the sketch selects this path. The older
`BramController` CPG/table path remains available as a fallback when
`kUseOnlinePolicies` is set to `false`.

## Arduino Dependencies

- ESP32 Arduino core.
- For the IMU test sketch, and for online policy IMU input when
  `BRAM_ENABLE_BNO08X_POLICY_IMU=1`: `Adafruit BNO08x` from Arduino Library Manager.

Native BLE command input is the default Arduino demo path:

```cpp
static constexpr bool kUseSerialInput = false;
```

The XIAO advertises as `BRAM` using a Nordic-UART-style BLE service:

- service UUID: `6e400001-b5a3-f393-e0a9-e50e24dcca9e`
- RX/write UUID: `6e400002-b5a3-f393-e0a9-e50e24dcca9e`
- TX/notify UUID: `6e400003-b5a3-f393-e0a9-e50e24dcca9e`

Send the same ASCII commands over BLE RX:

USB serial override is also enabled by default, so you can still send commands while BLE is compiled in:

```text
f 0.5 y -0.3
0.5 -0.3
stop
```

Set `kUseSerialInput = true` for serial-only bring-up.

The Bluepad32 gamepad hook is left behind `BRAM_ENABLE_BLUEPAD32`, but it is disabled by default. The Arduino Library Manager package named `Bluepad32 for NINA-W10 boards` is for NINA coprocessor boards and does not compile as native ESP32-C3 gamepad input.

## Compile Check

From the repo root:

```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 software/firmware/bram_esp32_controller
```

IMU test sketch:

```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 software/firmware/bno08x_imu_test
```

## Safety Notes

- Power servos directly from the 2S battery only if the servos are rated for `8.4 V`.
- Power the ESP32/control electronics from the BEC/regulator, not from the servo rail unless the board regulator supports it.
- Tie battery ground, servo ground, BEC ground, and ESP32 ground together.
- Start with conservative servo pulse limits in the sketch before testing on the robot.
- Bench test with servos disconnected from the mechanism first.
