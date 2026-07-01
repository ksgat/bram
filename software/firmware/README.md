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
- Bluepad32 Arduino library for BLE gamepad input.

Set `kUseSerialInput = true` in the sketch for serial debug mode. Set it to `false` for BLE controller demo mode.

If Bluepad32 is not installed and you only want serial mode, set `BRAM_ENABLE_BLUEPAD32` to `0` at the top of the sketch.

## Safety Notes

- Power servos from the BEC, not from the ESP32.
- Tie ESP32 ground and BEC/servo ground together.
- Start with conservative servo pulse limits in the sketch before testing on the robot.
- Bench test with servos disconnected from the mechanism first.
