"""CUDA implementation of the existing fixed-step LIF model using NVIDIA Warp.

The full sparse graph and delayed spike queue stay on the GPU. CUDA graphs batch
the substeps; only input drive and per-neuron spike counts cross the bus per slice.
"""
import math

import numpy as np
import warp as wp

from .brain import INHIBITORY_FLOOR


@wp.kernel
def integrate(voltage: wp.array(dtype=float), conductance: wp.array(dtype=float),
              refractory: wp.array(dtype=int), drive: wp.array(dtype=float),
              active: wp.array(dtype=int), counts: wp.array(dtype=int),
              queue: wp.array2d(dtype=int), queue_count: wp.array(dtype=int),
              cursor: wp.array(dtype=int), membrane_decay: float,
              conductance_decay: float, coupling: float, inhibitory_floor: float):
    i = wp.tid()
    if active[i] != 0:
        if refractory[i] > 0:
            refractory[i] = refractory[i] - 1
        if refractory[i] == 0:
            voltage[i] = (-52.0 + (voltage[i] + 52.0) * membrane_decay
                          + drive[i] * (1.0 - membrane_decay) + conductance[i] * coupling)
            # Matches the CPU kernel: inhibition cannot pull below the reversal potential.
            if voltage[i] < inhibitory_floor:
                voltage[i] = inhibitory_floor
            conductance[i] = conductance[i] * conductance_decay
            if voltage[i] > -45.0:
                counts[i] = counts[i] + 1
                future = (cursor[0] + 18) % 19
                offset = wp.atomic_add(queue_count, future, 1)
                queue[future, offset] = i
                # Inputs arriving this tick would be discarded by the CPU's final
                # spike reset. Resetting here lets propagation skip those inputs.
                voltage[i] = -52.0
                conductance[i] = 0.0
                refractory[i] = 22


@wp.kernel
def propagate(ptr: wp.array(dtype=int), post: wp.array(dtype=int),
              weight: wp.array(dtype=float), conductance: wp.array(dtype=float),
              refractory: wp.array(dtype=int), active: wp.array(dtype=int),
              active_count: wp.array(dtype=int), queue: wp.array2d(dtype=int),
              queue_count: wp.array(dtype=int), cursor: wp.array(dtype=int)):
    task = wp.tid()
    spike = task // 64
    lane = task % 64
    slot = cursor[0] % 19
    while spike < queue_count[slot]:
        neuron = queue[slot, spike]
        edge = ptr[neuron] + lane
        while edge < ptr[neuron + 1]:
            target = post[edge]
            if refractory[target] == 0:
                wp.atomic_add(conductance, target, weight[edge])
                if active[target] == 0:
                    previous = wp.atomic_cas(active, target, 0, 1)
                    if previous == 0:
                        wp.atomic_add(active_count, 0, 1)
            edge = edge + 64
        spike = spike + 1024


@wp.kernel
def reset_spikes(voltage: wp.array(dtype=float), conductance: wp.array(dtype=float),
                 refractory: wp.array(dtype=int), queue: wp.array2d(dtype=int),
                 queue_count: wp.array(dtype=int), cursor: wp.array(dtype=int)):
    i = wp.tid()
    future = (cursor[0] + 18) % 19
    while i < queue_count[future]:
        neuron = queue[future, i]
        voltage[neuron] = -52.0
        conductance[neuron] = 0.0
        refractory[neuron] = 22
        i = i + 4096


@wp.kernel
def tick(queue_count: wp.array(dtype=int), cursor: wp.array(dtype=int)):
    queue_count[cursor[0] % 19] = 0
    cursor[0] = (cursor[0] + 1) % 19


class CUDABrain:
    def __init__(self, brain):
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.name = f"CUDA / {self.device.name}"
        self.n = brain.n
        self.ptr = wp.array(brain.ptr.astype(np.int32), dtype=int, device=self.device)
        self.post = wp.array(brain.post, dtype=int, device=self.device)
        self.weight = wp.array(brain.weight, dtype=float, device=self.device)
        self.voltage = wp.zeros(self.n, dtype=float, device=self.device)
        self.conductance = wp.zeros(self.n, dtype=float, device=self.device)
        self.refractory = wp.zeros(self.n, dtype=int, device=self.device)
        self.drive = wp.zeros(self.n, dtype=float, device=self.device)
        self.active = wp.zeros(self.n, dtype=int, device=self.device)
        self.active_count = wp.zeros(1, dtype=int, device=self.device)
        self.counts = wp.zeros(self.n, dtype=int, device=self.device)
        self.queue = wp.zeros((19, self.n), dtype=int, device=self.device)
        self.queue_count = wp.zeros(19, dtype=int, device=self.device)
        self.cursor = wp.zeros(1, dtype=int, device=self.device)
        self.graphs = {}
        self.reset(brain)
        wp.load_module(device=self.device)

    def reset(self, brain):
        self.voltage.fill_(-52.0)
        for array in (self.conductance, self.refractory, self.drive, self.counts,
                      self.queue, self.queue_count, self.cursor):
            array.zero_()
        self.active.assign(brain.active_flag.astype(np.int32))
        self.active_count.assign(brain.active_count)

    def _substep(self):
        membrane = math.exp(-0.1 / 20.0)
        conductance = math.exp(-0.1 / 5.0)
        wp.launch(integrate, self.n, inputs=[self.voltage, self.conductance,
                  self.refractory, self.drive, self.active, self.counts, self.queue,
                  self.queue_count, self.cursor, membrane, conductance,
                  (membrane - conductance) / 3.0, INHIBITORY_FLOOR], device=self.device)
        wp.launch(propagate, min(self.n * 64, 65536), inputs=[self.ptr, self.post, self.weight,
                  self.conductance, self.refractory, self.active, self.active_count,
                  self.queue, self.queue_count, self.cursor], device=self.device)
        wp.launch(tick, 1, inputs=[self.queue_count, self.cursor], device=self.device)

    def advance(self, brain, steps):
        self.drive.assign(brain.drive)
        if steps not in self.graphs:
            with wp.ScopedCapture(device=self.device) as capture:
                self.counts.zero_()
                for _ in range(steps):
                    self._substep()
            self.graphs[steps] = capture.graph
        wp.capture_launch(self.graphs[steps])
        brain.counts[:] = self.counts.numpy()
        brain.active_count[:] = self.active_count.numpy()
        return brain.cursor + steps

    def sync_state(self, brain):
        """Diagnostic synchronization, excluded from the simulation hot path."""
        brain.voltage[:] = self.voltage.numpy()
        brain.conductance[:] = self.conductance.numpy()
        brain.refractory[:] = self.refractory.numpy()
