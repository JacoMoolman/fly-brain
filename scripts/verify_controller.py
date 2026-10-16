"""Cached body observations and walking rules must match upstream FlyGym."""
import time
import numpy as np
from flybrain_local.body import FlyGymBody
from flygym_demo.complex_terrain import HybridTurningController, HybridControllerObservation, apply_locomotion_action
b=FlyGymBody(np.zeros((1,2)),np.zeros(1))
reference=HybridTurningController(timestep=b.controller.timestep,preprogrammed_steps=b.preprogrammed_steps,output_dof_order=b.dof_order)
reference.reset(seed=0); b.controller.reset(seed=0)
started=time.perf_counter()
for i in range(500):
    obs=b._observation()
    original=HybridControllerObservation.from_sim(b.sim,b.fly.name)
    for name in ['thorax_z','tarsus5_z','stumbling_contact_forces','fly_heading']:
        assert np.allclose(getattr(obs,name),getattr(original,name),atol=1e-12), name
    signal=np.array([.8,.8]) if i<250 else np.array([.2,1.2])
    expected=reference.step(signal,original)
    actual=b.controller.step(signal,obs)
    assert np.allclose(actual.joint_angles,expected.joint_angles,atol=1e-12)
    assert np.array_equal(actual.adhesion_onoff,expected.adhesion_onoff)
    apply_locomotion_action(b.sim,b.fly.name,actual); b.sim.step()
print('500 controller and observation parity steps passed.')
