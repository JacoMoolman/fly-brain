"""FlyGym hybrid walking rules with invariant lookups moved out of each tick.

The rules and spline evaluation match FlyGym's HybridTurningController. Cached
DOF mappings and gain knots avoid rebuilding anatomy objects thousands of times.
"""
import numpy as np
from flygym_demo.complex_terrain import HybridTurningController, LocomotionAction
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
from flygym_demo.complex_terrain.hybrid_controller import (
    _CORRECTION_VECTORS, _RIGHT_LEG_CORRECTION_SIGN,
)


class CachedTurningController(HybridTurningController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        flat_dofs = [dof_spec_to_jointdof(leg, spec) for leg in self.legs
                     for spec in self.preprogrammed_steps.dofs_per_leg]
        self._output_indices = np.array([flat_dofs.index(dof) for dof in self.output_dof_order])
        self._gain_knots = []
        self._correction_vectors = []
        self._neutral = []
        for leg in self.legs:
            start, end = self.preprogrammed_steps.swing_period[leg]
            self._gain_knots.append(np.array([start, (start+end)/2, end+self.swing_extension,
                                              (end+2*np.pi)/2, 2*np.pi]))
            correction = _CORRECTION_VECTORS[leg[1]].copy()
            if leg.startswith("r"):
                correction *= _RIGHT_LEG_CORRECTION_SIGN
            self._correction_vectors.append(correction)
            self._neutral.append(self.preprogrammed_steps.neutral_pos[leg][:, 0])
        self._gain_values = np.array([0.0, 0.8, 0.0, -0.1, 0.0])

    def step(self, descending_signal, obs):
        signal = np.asarray(descending_signal, dtype=float)
        if signal.shape != (2,):
            raise ValueError("descending_signal must have shape (2,)")
        self.cpg_network.intrinsic_amps = np.repeat(np.abs(signal), 3)
        freqs = self._base_intrinsic_freqs.copy()
        freqs[:3] *= 1 if signal[0] >= 0 else -1
        freqs[3:] *= 1 if signal[1] >= 0 else -1
        self.cpg_network.intrinsic_freqs = freqs
        retract = self._select_retraction_leg(obs)
        if retract is not None and self.retraction_correction[retract] > self.retraction_persistence_initiation_threshold:
            self.retraction_persistence_counter[retract] = 1
        self._update_persistence_counter()
        stumbling = self._get_stumbling_mask(obs)
        self.cpg_network.step()
        angles = np.empty((6, 7))
        adhesion = np.empty(6, dtype=bool)
        net = np.empty(6)
        for i, leg in enumerate(self.legs):
            self._update_retraction_correction(i, retract)
            self._update_stumbling_correction(i, stumbling[i])
            correction = self.stumbling_correction[i]
            if self.retraction_correction[i] > 0:
                correction = self.retraction_correction[i]
                self.stumbling_correction[i] = 0
            phase = self.cpg_network.curr_phases[i]
            magnitude = self.cpg_network.curr_magnitudes[i]
            neutral = self._neutral[i]
            leg_angles = neutral + magnitude * (self.preprogrammed_steps._psi_funcs[leg](phase) - neutral)
            gain = np.interp(phase % (2*np.pi), self._gain_knots[i], self._gain_values)
            net[i] = min(max(correction, 0), self.max_correction) * gain
            angles[i] = leg_angles + net[i] * self._correction_vectors[i]
            adhesion[i] = self.enable_adhesion and self._get_adhesion_onoff(leg, phase)
        self.last_info = {"net_corrections": net,
                          "retraction_correction": self.retraction_correction.copy(),
                          "stumbling_correction": self.stumbling_correction.copy(),
                          "stumbling_mask": stumbling.copy(), "leg_to_correct_retraction": retract}
        return LocomotionAction(angles.ravel()[self._output_indices], adhesion)
