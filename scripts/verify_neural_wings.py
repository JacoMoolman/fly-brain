"""Prove neural causality and physical wing motion, without a browser."""
import numpy as np
from flybrain_local.brain import MaleCNSBrain, MotorCommand
from flybrain_local.body import FlyGymBody
from flybrain_local.paths import ProjectPaths
from flybrain_local.wings import decode_wings

brain = MaleCNSBrain(ProjectPaths.discover().prepared / "brain_graph.npz")
body = FlyGymBody(brain.retina_uv, brain.retina_side)
leg_command = MotorCommand(0, 0, 0, 0, 0, 0)
luminance = np.full(len(brain.retina), .11)
angles, rates = [], []
for i in range(100):
    brain.step(luminance, duration_ms=6, wing_stimulus=True)
    wing_command = brain.decode_wings()
    body.step(leg_command, .006, wing_command)
    angles.append(body.wing_state()["joint_angles"])
    rates.append([wing_command.left_rate, wing_command.right_rate])
    assert np.isfinite(body.sim.mj_data.qpos).all()
    assert not body.sim.mj_data.xfrc_applied.any()
angles = np.array(angles)
assert np.min(rates[-1]) > 10, rates[-1]
assert np.all(np.ptp(angles[30:, [1, 3]], axis=0) > .5), angles[-1]
assert body.position[2] < 2, "No flight or artificial lift is permitted"
print("Live DNg02 rates:", rates[-1], "stroke range:", np.ptp(angles[30:, [1, 3]], axis=0), flush=True)

# Changing only the neural output must change the physical wings, even though
# stimulation remains on in the preceding experiment. No UI state is involved.
brain.filtered_rates[brain.wing_indices] = 0
silent = brain.decode_wings()
assert silent.frequency == 0 and silent.left_amplitude == 0 and silent.right_amplitude == 0
quiet_angles = []
for i in range(100):
    body.step(leg_command, .006, silent)
    quiet_angles.append(body.wing_state()["joint_angles"])
assert np.max(np.ptp(np.array(quiet_angles)[-20:, [1, 3]], axis=0)) < .03

# Left and right outputs are independent and preserve the dataset's side labels.
brain.filtered_rates[brain.wing_indices[brain.soma_side[brain.wing_indices] == "L"]] = 100
left_only = brain.decode_wings()
assert left_only.left_amplitude > 0 and left_only.right_amplitude == 0
body.reset()
brain.reset()
assert brain.decode_wings().frequency == 0
assert not body.sim.mj_data.xfrc_applied.any()
print("Neural silencing, side specificity, physical motion, no lift and reset passed.")
