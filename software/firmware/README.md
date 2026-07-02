# Bram Firmware


The firmware does not run MuJoCo, PPO, Python, or a neural network. It uses the exported deterministic controller data from `software/gait_discovery/exports/bram_grid_controller_export.json`: forward/back CPG params, learned yaw action tables, and mixed-arc blend grids.

## Layout

- `bram_esp32_controller/bram_esp32_controller.ino` is the Arduino sketch.
- `bram_esp32_controller/bram_controller.hpp` is the runtime controller logic.
- `bram_esp32_controller/bram_controller_data.hpp` is generated controller data.
- `tools/export_bram_firmware.py` regenerates the data header from the gait-discovery export.

## Regenerate Controller Data

From the repo root:

```bash
python3 software/firmware/tools/export_bram_firmware.py
```

## Arduino Dependencies

- ESP32 Arduino core.

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

## Safety Notes

- Power servos directly from the 2S battery only if the servos are rated for `8.4 V`.
- Power the ESP32/control electronics from the BEC/regulator, not from the servo rail unless the board regulator supports it.
- Tie battery ground, servo ground, BEC ground, and ESP32 ground together.
- Start with conservative servo pulse limits in the sketch before testing on the robot.
- Bench test with servos disconnected from the mechanism first.
