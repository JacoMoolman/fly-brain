"""Check actual translation and signed turns from manual walking commands."""
import numpy as np
from flybrain_local.body import FlyGymBody
from flybrain_local.brain import MotorCommand
MANUAL_DRIVE = {
    'forward': (0.7, 0.7),
    'backward': (-0.7, -0.7),
    'left': (-0.5, 0.5),
    'right': (0.5, -0.5),
    'loom': (1.0, 1.0),
}
body = FlyGymBody(np.zeros((1, 2)), np.zeros(1, dtype=np.uint8))
results = {}
for name, (left, right) in MANUAL_DRIVE.items():
    body.reset()
    state = body.step(MotorCommand(left, right, 0, 0, 0, 0), 0.3)
    w, x, y, z = body.sim.mj_data.xquat[body.body_ids[body.thorax_index]]
    yaw = float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    results[name] = {'displacement': state.displacement, 'yaw_radians': yaw}
    print(name, results[name], flush=True)
assert results['forward']['displacement'][0] > 0.1
assert results['backward']['displacement'][0] < -0.1
assert results['left']['yaw_radians'] > 0.05
assert results['right']['yaw_radians'] < -0.05
assert results['loom']['displacement'][0] > 0.1
print('All walking directions passed.')
