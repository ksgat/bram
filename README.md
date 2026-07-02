# BRAM!
<img width="602" height="537" alt="Screenshot 2026-07-01 at 9 38 01 PM" src="https://github.com/user-attachments/assets/5d3faa53-d28b-4857-beb6-91a0eb61cc4c" />

BRAM is the underactuated, three legged, equilateral triangle from hell.

The project is mostly about control. BRAM’s motion is developed in MuJoCo first, where gait search and reinforcement
learning are used to discover usable crawling and turning behaviors. The current controller combines deterministic
forward/back gait patterns with learned yaw primitives, then blends them into joystick-style movement for the physical
robot.

Right now BRAM can crawl. The next major behavior is self-righting, so the robot can flip itself back over and continue
moving. The goal demo is a side-by-side comparison of the MuJoCo simulation and the real robot crawling, turning, and
eventually recovering from a flip.

## Project Structure

```text
BRAM
├── CAD/                  # STEP files for the full assembly and major mechanical parts
├── software/             # All robot software
│   ├── firmware/         # ESP32/XIAO firmware for the physical robot
│   └── gait_discovery/   # MuJoCo model, gait search, RL experiments, and controller exports
├── BOM.csv               # Bill of materials with parts, costs, and purchase links
├── JOURNAL.md            # High-level project journal and development summary
└── README.md             # Project overview, wiring notes, BOM table, and documentation
```


# Wiring diagram
Everything is handwired, no pcb.
<img width="990" height="537" alt="Screenshot 2026-07-01 at 11 14 12 PM" src="https://github.com/user-attachments/assets/baa2494c-bb71-4c4b-b6cb-f2fee5b389f7" />


## Bill of Materials

Source CSV: [BOM.csv](BOM.csv)

| Part | Quantity | Unit Cost | Total Cost | HC Cost | Link |
| --- | ---: | ---: | ---: | ---: | --- |
| 45 kg-cm servo | 3 | $24.99 | $74.97 | $74.97 | [Amazon](https://www.amazon.com/gp/product/B0CNFZ8BW8?smid=A28WSOZPYZ1XET&psc=1) |
| Xiao ESP32C3 | 1 | $10.99 | $10.99 | $10.99 | [Amazon](https://www.amazon.com/gp/product/B0DRNSV5CS?th=1) |
| 3/8 OD carbon rod | 1 | $7.73 | $7.73 | $0.00 | [Windcatcher RC](https://windcatcherrc.com/product/carbon-fiber-tube-10mm-x-8mm-x-1000mm/) |
| BEC | 1 | $8.99 | $8.99 | $8.99 | [Amazon](https://www.amazon.com/gp/product/B0DKTMGBHL?smid=A38CU2XC1RY0BO&psc=1) |
| 9DoF IMU | 1 | $20.49 | $20.49 | $20.49 | [Amazon](https://www.amazon.com/EC-Buying-Accelerometer-Gyroscope-Magnetometer/dp/B0CDGZMLPP) |
| M5 hardware | 1 | $9.99 | $9.99 | $9.99 | [Amazon](https://www.amazon.com/gp/product/B0FG2CT35D?th=1) |
| Ovionic 2S LiPo 1000 mAh | 1 | $23.99 | $23.99 | $23.99 | [Amazon](https://www.amazon.com/gp/product/B07CVBJ3SL?smid=A1KODDOPEPALCP&psc=1) |
| **Total** |  |  | **$157.15** | **$149.42** |  |
