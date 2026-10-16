from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flygym import Simulation
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)


asset_root = Path(r"J:\models\fruit fly")
output = asset_root / "results" / "verification"
output.mkdir(parents=True, exist_ok=True)
fly = make_locomotion_fly(name="verification_fly", add_adhesion=True, colorize=True)
camera = fly.add_tracking_camera(
    name="verification_camera",
    pos_offset=(-0.5, -7.5, 0.0),
    rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
    fovy=35.0,
)
world = FlatGroundWorld()
world.add_fly(
    fly,
    [0, 0, 0.8],
    Rotation3D("quat", [1, 0, 0, 0]),
    bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
    add_ground_contact_sensors=False,
)
sim = Simulation(world)
renderer = sim.set_renderer([camera], camera_res=(240, 320), playback_speed=1.0, output_fps=20)
steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
controller = HybridTurningController(
    timestep=sim.timestep,
    preprogrammed_steps=steps,
    output_dof_order=dof_order,
)
sim.reset()
controller.reset(seed=0)
apply_locomotion_action(
    sim,
    fly.name,
    LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool),
    ),
)
sim.warmup()
start = sim.get_body_positions(fly.name).copy()
for _ in range(int(0.3 / sim.timestep)):
    observation = HybridControllerObservation.from_sim(sim, fly.name)
    action = controller.step(np.array([0.7, 0.7]), observation)
    apply_locomotion_action(sim, fly.name, action)
    sim.step_with_profile()
    sim.render_as_needed_with_profile()
finish = sim.get_body_positions(fly.name).copy()
video = output / "flygym-walk.mp4"
renderer.save_video(video)
report = {
    "simulated_seconds": float(sim.mj_data.time),
    "rendered_frames": len(renderer.frames[camera.name]),
    "max_body_displacement_mm": float(np.max(np.linalg.norm(finish - start, axis=1))),
    "video": str(video),
}
(output / "flygym-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
