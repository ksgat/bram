from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_KEYFRAMES = Path(__file__).with_name("bram_v2_keyframes.json")


@dataclass(frozen=True)
class Segment:
    start: np.ndarray
    end: np.ndarray
    steps: int


class KeyframeController:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.servo_order = list(payload["servo_order"])
        self.rate_hz = int(payload.get("rate_hz", 50))
        self.postures = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in payload["postures"].items()
        }
        self.sequences = payload["sequences"]

    @classmethod
    def from_file(cls, path: Path) -> "KeyframeController":
        with path.open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    def segment(self, item: dict) -> Segment:
        return Segment(
            start=self.postures[item["from"]],
            end=self.postures[item["to"]],
            steps=max(1, int(item["steps"])),
        )

    def sequence_once(self, name: str) -> np.ndarray:
        if name not in self.sequences:
            known = ", ".join(sorted(self.sequences))
            raise KeyError(f"unknown sequence {name!r}; known sequences: {known}")
        chunks = [interpolate(self.segment(item)) for item in self.sequences[name]]
        if not chunks:
            return np.zeros((0, len(self.servo_order)), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def sequence(self, name: str, cycles: int = 1) -> np.ndarray:
        one = self.sequence_once(name)
        if cycles <= 1:
            return one
        return np.concatenate([one] * cycles, axis=0)


def interpolate(segment: Segment) -> np.ndarray:
    if segment.steps <= 1:
        return segment.end.reshape(1, -1).astype(np.float32)
    alpha = np.linspace(0.0, 1.0, segment.steps, endpoint=False, dtype=np.float32)
    return (segment.start[None, :] * (1.0 - alpha[:, None]) +
            segment.end[None, :] * alpha[:, None]).astype(np.float32)


def action_to_pulse_us(
    action: np.ndarray,
    *,
    neutral_us: int = 1500,
    min_us: int = 1000,
    max_us: int = 2000,
) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0)
    positive = neutral_us + (max_us - neutral_us) * np.maximum(action, 0.0)
    negative = neutral_us + (neutral_us - min_us) * np.minimum(action, 0.0)
    return np.where(action >= 0.0, positive, negative).round().astype(np.int32)


def write_csv(path: Path, rows: np.ndarray, servo_order: Iterable[str]) -> None:
    pulses = action_to_pulse_us(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *servo_order, *[f"{name}_us" for name in servo_order]])
        for step, (action, pulse) in enumerate(zip(rows, pulses, strict=True)):
            writer.writerow([step, *[f"{x:.6f}" for x in action], *pulse.tolist()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument("--sequence", default="walk_forward")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    controller = KeyframeController.from_file(args.keyframes)
    rows = controller.sequence(args.sequence, cycles=args.cycles)

    for step, values in enumerate(rows[: args.steps]):
        parts = " ".join(
            f"{name}={value: .6f}"
            for name, value in zip(controller.servo_order, values, strict=True)
        )
        print(f"step={step:04d} {parts}")

    if args.csv:
        write_csv(args.csv, rows, controller.servo_order)
        print(f"wrote {args.csv} rows={len(rows)} rate_hz={controller.rate_hz}")


if __name__ == "__main__":
    main()

