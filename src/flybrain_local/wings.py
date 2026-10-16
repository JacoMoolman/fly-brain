"""Explicit, testable bridge from MaleCNS DNg02 firing to wing actuators.

DNg02 is associated with wing stroke amplitude regulation. The numerical gains
and oscillator below are an engineered motor bridge, not a reconstructed muscle
model or a claim of validated biological dynamics.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WingCommand:
    left_rate: float = 0.0
    right_rate: float = 0.0
    left_amplitude: float = 0.0
    right_amplitude: float = 0.0
    frequency: float = 0.0


def decode_wings(rates, indices, sides) -> WingCommand:
    left = indices[sides[indices] == "L"]
    right = indices[sides[indices] == "R"]
    left_rate = float(np.mean(rates[left])) if len(left) else 0.0
    right_rate = float(np.mean(rates[right])) if len(right) else 0.0
    power = np.clip((np.array([left_rate, right_rate]) - 1.0) / 400.0, 0, 1)
    frequency = 180.0 + 40.0 * float(np.mean(power)) if np.max(power) > 0 else 0.0
    return WingCommand(left_rate, right_rate, float(power[0] * 0.9),
                       float(power[1] * 0.9), frequency)
