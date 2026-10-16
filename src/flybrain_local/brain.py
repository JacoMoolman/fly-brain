from __future__ import annotations

import math
import os
import base64
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from .wings import decode_wings

# Inhibitory (chloride) reversal potential. Acts as a floor on the membrane potential:
# the model is current-based, so without it summed inhibition is unbounded.
INHIBITORY_FLOOR = -75.0

# Amplitude of the current injected into a button's target descending neurons. This is
# an engineered optogenetic-style activation, not a measured physiological input: the
# targets sit under heavy network inhibition, so the drive has to exceed it to make
# them fire at all.
# Measured on this graph: 34 produced zero spikes in the targets (they sit at the
# inhibitory floor); 300 gives a cleanly lateralised response (114 spikes on the
# stimulated side vs 3 on the other) and saturates by ~1000.
STIMULUS_DRIVE = 300.0


@njit(cache=True)
def _advance(
    ptr,
    post,
    weight,
    voltage,
    conductance,
    refractory,
    drive,
    queue,
    queue_count,
    cursor,
    steps,
    dt,
    counts,
    active,
    active_flag,
    active_count,
):
    membrane_decay = math.exp(-dt / 20.0)
    conductance_decay = math.exp(-dt / 5.0)
    coupling = (membrane_decay - conductance_decay) / 3.0
    delay_slots = queue.shape[0]
    delay_ticks = int(round(1.8 / dt))
    refractory_ticks = int(round(2.2 / dt))
    for _ in range(steps):
        slot = cursor % delay_slots
        for k in range(active_count[0]):
            neuron = active[k]
            if refractory[neuron] > 0:
                refractory[neuron] -= 1
            if refractory[neuron] == 0:
                voltage[neuron] = (
                    -52.0
                    + (voltage[neuron] + 52.0) * membrane_decay
                    + drive[neuron] * (1.0 - membrane_decay)
                    + conductance[neuron] * coupling
                )
                # Inhibition cannot drag a real neuron below its inhibitory reversal
                # potential. Without this floor, summed inhibition across the full
                # graph parked cells at -100 to -160 mV, where no stimulus could ever
                # bring them back to the -45 mV threshold.
                if voltage[neuron] < INHIBITORY_FLOOR:
                    voltage[neuron] = INHIBITORY_FLOOR
                conductance[neuron] *= conductance_decay
                if voltage[neuron] > -45.0:
                    counts[neuron] += 1
                    future = (cursor + delay_ticks) % delay_slots
                    queue[future, queue_count[future]] = neuron
                    queue_count[future] += 1
        for q in range(queue_count[slot]):
            neuron = queue[slot, q]
            for edge in range(ptr[neuron], ptr[neuron + 1]):
                target = post[edge]
                if refractory[target] > 0:
                    continue
                conductance[target] += weight[edge]
                if active_flag[target] == 0:
                    active_flag[target] = 1
                    active[active_count[0]] = target
                    active_count[0] += 1
        queue_count[slot] = 0
        future = (cursor + delay_ticks) % delay_slots
        for q in range(queue_count[future]):
            neuron = queue[future, q]
            voltage[neuron] = -52.0
            conductance[neuron] = 0.0
            refractory[neuron] = refractory_ticks
        cursor += 1
    return cursor


@dataclass(frozen=True)
class MotorCommand:
    left: float
    right: float
    forward_rate: float
    backward_rate: float
    turn_rate: float
    assist: float


class MaleCNSBrain:
    """Full retained MaleCNS graph with fixed-weight LIF proxy dynamics."""

    def __init__(self, graph_path: Path, dt_ms: float = 0.1, backend: str | None = None):
        if dt_ms != 0.1:
            raise ValueError("The fixed-step implementation requires dt_ms=0.1")
        graph = np.load(graph_path, allow_pickle=False)
        for name in (
            "ptr",
            "post",
            "weight",
            "ids",
            "retina",
            "retina_uv",
            "retina_side",
            "lamina",
            "positions",
            "position_measured",
            "cell_type",
            "superclass",
            "neuron_class",
            "soma_side",
            "readout_indices",
            "display_indices",
        ):
            setattr(self, name, graph[name])
        self.dt_ms = dt_ms
        self.n = len(self.ids)
        # The released R1-R6 somaSide is usually unknown. Older prepared graphs
        # assigned every unknown receptor to the right eye. Recover its hemisphere
        # from the strongest outgoing mapped lamina connection at load time.
        lamina_mask = np.zeros(self.n, dtype=bool)
        lamina_mask[self.lamina] = True
        for k, neuron in enumerate(self.retina):
            if self.soma_side[neuron] in ("L", "R"):
                continue
            start, end = self.ptr[neuron], self.ptr[neuron + 1]
            posts = self.post[start:end]
            mapped = lamina_mask[posts]
            if np.any(mapped):
                partner = posts[mapped][np.argmax(np.abs(self.weight[start:end][mapped]))]
                self.retina_side[k] = 0 if self.soma_side[partner] == "L" else 1
        self.cursor = 0
        self.voltage = np.full(self.n, -52, dtype=np.float32)
        self.conductance = np.zeros(self.n, dtype=np.float32)
        self.drive = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        delay_slots = int(round(1.8 / dt_ms)) + 1
        self.queue = np.zeros((delay_slots, self.n), dtype=np.int32)
        self.queue_count = np.zeros(delay_slots, dtype=np.int32)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        self.wing_indices = np.flatnonzero(np.char.startswith(self.cell_type, "DNg02_")).astype(np.int32)
        if not len(self.wing_indices):
            raise ValueError("The graph has no DNg02 wing-control population")
        initial = np.unique(np.r_[self.retina, self.lamina, self.readout_indices, self.wing_indices]).astype(np.int32)
        self._initial_active = initial
        self.active[: len(initial)] = initial
        self.active_flag[initial] = 1
        self.active_count = np.asarray([len(initial)], dtype=np.int32)
        self.filtered_luminance = np.zeros(len(self.retina), dtype=np.float32)
        self.filtered_rates = np.zeros(self.n, dtype=np.float32)
        self.total_spikes = 0
        self.simulated_ms = 0.0
        self.last_wall_seconds = 0.0
        self._indices_by_type = {
            cell_type: np.flatnonzero(self.cell_type == cell_type).astype(np.int32)
            for cell_type in ("DNa02", "DNp09", "oDN1", "MDN", "DNp01", "DNp20", "DNpe017")
        }
        self._gpu = None
        self.backend = "CPU / Numba"
        self.backend_error = None
        self.stimulus_drive = float(os.environ.get("FLYBRAIN_STIMULUS_DRIVE", STIMULUS_DRIVE))
        selected = backend or os.environ.get("FLYBRAIN_BRAIN_BACKEND", "auto")
        if selected not in {"auto", "cpu", "cuda"}:
            raise ValueError("Brain backend must be auto, cpu or cuda")
        if selected != "cpu":
            try:
                from .gpu_brain import CUDABrain
                self._gpu = CUDABrain(self)
                self.backend = self._gpu.name
            except Exception as exc:
                if selected == "cuda":
                    raise
                self.backend_error = f"CUDA initialization failed: {exc}"
                print(self.backend_error, flush=True)

    def reset(self) -> None:
        self.cursor = 0
        self.voltage.fill(-52)
        self.conductance.fill(0)
        self.drive.fill(0)
        self.refractory.fill(0)
        self.queue.fill(0)
        self.queue_count.fill(0)
        self.counts.fill(0)
        self.filtered_luminance.fill(0)
        self.filtered_rates.fill(0)
        self.active.fill(0)
        self.active_flag.fill(0)
        self.active[: len(self._initial_active)] = self._initial_active
        self.active_flag[self._initial_active] = 1
        self.active_count[0] = len(self._initial_active)
        self.total_spikes = 0
        self.simulated_ms = 0.0
        if self._gpu is not None:
            self._gpu.reset(self)

    def stimulus_indices(self, stimulus: str | None) -> np.ndarray:
        target_by_stimulus = {
            "loom": "DNp01",
            "forward": "DNp09",
            "backward": "MDN",
            "left": "DNa02",
            "right": "DNa02",
        }
        target_type = target_by_stimulus.get(stimulus or "")
        if not target_type:
            return np.empty(0, dtype=np.int32)
        indices = self._indices_by_type[target_type]
        if stimulus in ("left", "right"):
            desired_side = "R" if stimulus == "left" else "L"
            indices = indices[self.soma_side[indices] == desired_side]
        return indices

    def _stimulate(self, stimulus: str | None) -> None:
        self.drive[self.stimulus_indices(stimulus)] = self.stimulus_drive

    def step(
        self,
        luminance: np.ndarray,
        duration_ms: float = 15.0,
        stimulus: str | None = None,
        wing_stimulus: bool = False,
    ) -> np.ndarray:
        luminance = np.asarray(luminance, dtype=np.float32)
        if luminance.shape != (len(self.retina),) or not np.isfinite(luminance).all():
            raise ValueError("One finite luminance value is required per mapped R1-R6 neuron")
        steps = int(round(duration_ms / self.dt_ms))
        alpha = 1 - math.exp(-(steps * self.dt_ms) / 10.0)
        self.filtered_luminance += alpha * (np.clip(luminance, 0, 1) - self.filtered_luminance)
        self.drive.fill(0)
        self.drive[self.lamina] = 12.0
        # Modeled phototransduction: preserve scene contrast across the full
        # luminance range. The previous saturating curve made ordinary objects
        # and the background nearly identical at the neural input.
        self.drive[self.retina] = 60.0 * self.filtered_luminance
        self._stimulate(stimulus)
        if wing_stimulus:
            self.drive[self.wing_indices] = self.stimulus_drive
        self.counts.fill(0)
        started = time.perf_counter()
        if self._gpu is not None:
            self.cursor = self._gpu.advance(self, steps)
        else:
            self.cursor = _advance(
                self.ptr,
                self.post,
                self.weight,
                self.voltage,
                self.conductance,
                self.refractory,
                self.drive,
                self.queue,
                self.queue_count,
                self.cursor,
                steps,
                self.dt_ms,
                self.counts,
                self.active,
                self.active_flag,
                self.active_count,
            )
        self.last_wall_seconds = time.perf_counter() - started
        self.total_spikes += int(self.counts.sum())
        self.simulated_ms += steps * self.dt_ms
        seconds = duration_ms / 1000.0
        rate_alpha = 1 - math.exp(-seconds / 0.1)
        self.filtered_rates += rate_alpha * (self.counts / seconds - self.filtered_rates)
        return self.counts

    def _population_rate(self, cell_type: str, side: str | None = None) -> float:
        indices = self._indices_by_type.get(cell_type, np.empty(0, dtype=np.int32))
        if side is not None:
            indices = indices[self.soma_side[indices] == side]
        return float(np.mean(self.filtered_rates[indices])) if len(indices) else 0.0

    def decode_motor(self, assist: float = 0.0) -> MotorCommand:
        forward = self._population_rate("DNp09") + self._population_rate("oDN1")
        backward = self._population_rate("MDN")
        left_rate = self._population_rate("DNa02", "L")
        right_rate = self._population_rate("DNa02", "R")
        turn = float(np.clip((right_rate - left_rate) / 80.0, -0.65, 0.65))
        neural_forward = float(np.clip((forward - backward) / 120.0, -0.45, 0.55))
        base = float(np.clip(assist + neural_forward, -1.2, 1.2))
        return MotorCommand(
            left=float(np.clip(base - turn, -1.4, 1.4)),
            right=float(np.clip(base + turn, -1.4, 1.4)),
            forward_rate=forward,
            backward_rate=backward,
            turn_rate=right_rate - left_rate,
            assist=assist,
        )

    def decode_wings(self):
        return decode_wings(self.filtered_rates, self.wing_indices, self.soma_side)

    def layout_payload(self) -> dict:
        indices = self.display_indices
        return {
            "indices": indices.tolist(),
            "all_ids": [str(i) for i in self.ids],
            "all_superclass": self.superclass.tolist(),
            "all_neuron_class": self.neuron_class.tolist(),
            "positions": np.round(self.positions[indices], 4).tolist(),
            "measured": self.position_measured[indices].astype(np.uint8).tolist(),
            "superclass": self.superclass[indices].tolist(),
            "neuron_class": self.neuron_class[indices].tolist(),
        }

    def activity_payload(self, limit: int | None = None) -> list[dict]:
        active = np.flatnonzero(self.counts)
        if limit is not None and len(active) > limit:
            active = active[np.argpartition(self.counts[active], -limit)[-limit:]]
        active = active[np.argsort(self.counts[active])[::-1]]
        return [
            {
                "index": int(index),
                "spikes": int(self.counts[index]),
            }
            for index in active
        ]

    def activity_binary(self) -> str:
        """All spiking neurons as compact little-endian (index, count) pairs."""
        active = np.flatnonzero(self.counts)
        pairs = np.column_stack((active, self.counts[active])).astype("<u4")
        return base64.b64encode(pairs.tobytes()).decode("ascii")
