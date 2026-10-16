from __future__ import annotations

from dataclasses import dataclass
import os

import mujoco as mj
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.compose.fly.base_fly import BaseFly
from flygym.utils.math import Rotation3D
from flygym.utils.mjcf import add_actuator
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

from .brain import MotorCommand
from .controller import CachedTurningController
from .wings import WingCommand

# Every segment's visible geometry sits at that segment's body origin with no extra
# local offset (see NeuroMechFly._add_one_body_and_geoms), so the world position and
# orientation MuJoCo reports for a body IS the placement of its mesh. Right-side
# segments reuse the mirrored left-side mesh (see NeuroMechFly._add_mesh_assets).
MESH_SCALE = BaseFly.SCALE
GROUND_HALF_SIZE = 22.0
WALL_INNER_LIMIT = 19.5
WALL_HALF_HEIGHT = 6.0
# While the wings beat the body is raised HOVER_RISE mm above standing height,
# far below the 12 mm walls. This is placement, not aerodynamics.
HOVER_RISE = 3.0
HOVER_CLIMB_RATE = 6.0
# Wing stroke amplitude (0 to 0.9) needed to stay up; weaker residual DNg02 firing lets the fly drop.
HOVER_MIN_AMPLITUDE = 0.3


def _scene_shapes() -> list[dict]:
    shapes = [
        {"name": "apple", "shape": "sphere", "position": [7.0, 2.0, 1.1],
         "size": [1.1], "color": [0.88, 0.08, 0.04]},
        {"name": "banana", "shape": "ellipsoid", "position": [10.0, -3.0, 0.75],
         "size": [2.2, 0.65, 0.7], "color": [0.94, 0.78, 0.08]},
        {"name": "orange", "shape": "sphere", "position": [-5.0, -6.0, 1.0],
         "size": [1.0], "color": [0.98, 0.55, 0.05]},
        {"name": "pear", "shape": "ellipsoid", "position": [-10.0, -2.0, 1.25],
         "size": [0.9, 0.9, 1.25], "color": [0.72, 0.8, 0.2]},
        {"name": "mushroom_stem", "shape": "cylinder", "position": [4.0, 9.0, 0.8],
         "size": [0.4, 0.8], "color": [0.93, 0.9, 0.82]},
        {"name": "mushroom_cap", "shape": "ellipsoid", "position": [4.0, 9.0, 1.8],
         "size": [1.3, 1.3, 0.6], "color": [0.85, 0.12, 0.1]},
        {"name": "rock_0", "shape": "ellipsoid", "position": [3.0, -12.0, 0.6],
         "size": [2.0, 1.4, 0.9], "color": [0.5, 0.5, 0.48]},
        {"name": "rock_1", "shape": "ellipsoid", "position": [-13.0, 13.0, 0.5],
         "size": [1.4, 1.8, 0.8], "color": [0.5, 0.5, 0.48]},
    ]
    for i, (x, y) in enumerate(((12.0, 8.0), (-7.0, 9.0), (11.0, -10.0), (-14.0, -12.0))):
        shapes.extend((
            {"name": f"tree_{i}_trunk", "shape": "cylinder",
             "position": [x, y, 3.5], "size": [0.7, 3.5], "color": [0.36, 0.2, 0.08]},
            {"name": f"tree_{i}_crown", "shape": "sphere",
             "position": [x, y, 9.5], "size": [3.2], "color": [0.1, 0.48, 0.12]},
        ))
    h = WALL_HALF_HEIGHT
    for name, position, size in (
        ("north", [0, WALL_INNER_LIMIT + 0.5, h], [20, 0.5, h]),
        ("south", [0, -WALL_INNER_LIMIT - 0.5, h], [20, 0.5, h]),
        ("east", [WALL_INNER_LIMIT + 0.5, 0, h], [0.5, 20, h]),
        ("west", [-WALL_INNER_LIMIT - 0.5, 0, h], [0.5, 20, h]),
    ):
        shapes.append({"name": f"wall_{name}", "shape": "box", "position": position,
                       "size": size, "color": [0.26, 0.32, 0.36]})
    return shapes


def _mesh_for_segment(name: str) -> tuple[str, bool]:
    if name[0] == "r":
        return f"l{name[1:]}", True
    return name, False


@dataclass(frozen=True)
class BodyState:
    simulated_seconds: float
    position: tuple[float, float, float]
    displacement: tuple[float, float, float]


class FlyGymBody:
    """Official FlyGym body driven by decoded MaleCNS population activity."""

    def __init__(self, retina_uv: np.ndarray, retina_side: np.ndarray):
        self.fly = make_locomotion_fly(name="male_cns_fly", add_adhesion=True, colorize=True)
        self.fly.bodyseg_to_mjcfbody[BodySegment("c_thorax")].add_geom(
            name="scene_collision_body", type=mj.mjtGeom.mjGEOM_SPHERE,
            size=(0.75, 0.0, 0.0), rgba=(0.0, 0.0, 0.0, 0.0),
            contype=2, conaffinity=2, group=3, mass=1e-8,
        )
        self._wing_joints = []
        self._wing_actuators = []
        for side, sign in (("l", -1), ("r", 1)):
            wing = self.fly.bodyseg_to_mjcfbody[BodySegment(f"{side}_wing")]
            rotation = np.empty(9)
            mj.mju_quat2Mat(rotation, np.asarray(wing.quat) / np.linalg.norm(wing.quat))
            rotation = rotation.reshape(3, 3)
            for motion, axis in (("spread", [0, 0, sign]), ("stroke", [0, 1, 0])):
                joint = wing.add_joint(name=f"{side}_wing_{motion}",
                    type=mj.mjtJoint.mjJNT_HINGE, axis=rotation.T @ axis,
                    damping=0.0002, armature=1e-7, limited=True,
                    range=[-1.8, 1.8])
                actuator = add_actuator(self.fly.mjcf_root, "position",
                    name=f"{side}_wing_{motion}_motor", joint=joint.name,
                    kp=20, kv=0.002, forcerange=(-15, 15))
                self._wing_joints.append(joint)
                self._wing_actuators.append(actuator)
        # Additional wing DOFs require a rebuilt neutral keyframe. They remain
        # separate from FlyGym's leg-only controller and its actuator ordering.
        self.fly._neutral_keyframe.qpos = []
        self.fly._neutral_keyframe.ctrl = []
        self.fly._rebuild_neutral_keyframe()
        self.world = FlatGroundWorld(half_size=GROUND_HALF_SIZE)
        self.target_shapes = _scene_shapes()
        geom_types = {"sphere": mj.mjtGeom.mjGEOM_SPHERE,
                      "ellipsoid": mj.mjtGeom.mjGEOM_ELLIPSOID,
                      "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
                      "box": mj.mjtGeom.mjGEOM_BOX}
        for shape in self.target_shapes:
            self.world.mjcf_root.worldbody.add_geom(
                name=shape["name"], type=geom_types[shape["shape"]], pos=shape["position"],
                size=shape["size"], rgba=(*shape["color"], 1.0),
                friction=(1.0, 0.005, 0.0001), contype=2, conaffinity=2,
            )
        self.world.add_fly(
            self.fly,
            [0, 0, 0.8],
            Rotation3D("quat", [1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=True,
        )
        # Keep every segment as its own compiled body. With fusestatic on, jointless
        # segments (the head, in a locomotion-only model) are fused into their parent
        # and vanish from the model; mj_name2id then returns -1, which numpy silently
        # turns into xpos[-1] - so the head was being drawn at the last segment's
        # transform, down by the right hind leg.
        self.fly.mjcf_root.compiler.fusestatic = False
        self.world.mjcf_root.compiler.fusestatic = False
        self.sim = Simulation(self.world, timestep=float(os.environ.get("FLYBRAIN_PHYSICS_DT", "0.0005")))
        self.preprogrammed_steps = PreprogrammedSteps()
        self.dof_order = self.fly.get_actuated_jointdofs_order("position")
        self.control_stride = 2
        self.controller = CachedTurningController(
            timestep=self.sim.timestep * self.control_stride,
            preprogrammed_steps=self.preprogrammed_steps,
            output_dof_order=self.dof_order,
        )
        self.bodyseg_order = self.fly.get_bodysegs_order()
        self.segment_names = [segment.name for segment in self.bodyseg_order]
        self.thorax_index = self.segment_names.index("c_thorax")
        self.body_ids = np.asarray([
            mj.mj_name2id(self.sim.mj_model, mj.mjtObj.mjOBJ_BODY,
                          self.fly.bodyseg_to_mjcfbody[segment].name)
            for segment in self.bodyseg_order
        ], dtype=np.int32)
        missing = [name for name, body_id in zip(self.segment_names, self.body_ids)
                   if body_id < 0]
        if missing:
            raise RuntimeError(
                f"These segments have no compiled MuJoCo body, so their reported pose "
                f"would silently be another segment's: {missing}"
            )
        self._thorax_id = int(self.body_ids[self.thorax_index])
        model = self.sim.mj_model
        root_joint = int(model.body_jntadr[int(model.body_rootid[self._thorax_id])])
        self._root_qpos = int(model.jnt_qposadr[root_joint])
        self._root_dof = int(model.jnt_dofadr[root_joint])
        self._gravity = float(-model.opt.gravity[2])
        self._foot_ids = self.body_ids[[self.segment_names.index(f"{leg}_tarsus5")
                                        for leg in self.controller.legs]]
        self._contact_rows = np.full(self.sim.mj_model.ngeom, -1, dtype=int)
        geom_map = self.sim._internal_geomid_by_bodyseg_by_fly[self.fly.name]
        for i, leg in enumerate(self.controller.legs):
            for j, link in enumerate(("tibia", "tarsus1", "tarsus2")):
                self._contact_rows[geom_map[BodySegment(f"{leg}_{link}")]] = i * 3 + j
        self._ground_geoms = np.zeros(self.sim.mj_model.ngeom, dtype=bool)
        self._ground_geoms[self.sim._internal_ground_geom_ids] = True
        self.retina_uv = np.asarray(retina_uv, dtype=np.float32)
        self.retina_side = np.asarray(retina_side, dtype=np.uint8)
        u = self.retina_uv[:, 0]
        v = self.retina_uv[:, 1]
        # Each eye samples a broad, overlapping hemifield. The MaleCNS hex map
        # establishes relative receptor positions, but absolute optics are modeled.
        azimuth = np.where(self.retina_side == 0,
                           np.deg2rad(150.0 - 180.0 * u),
                           np.deg2rad(30.0 - 180.0 * u))
        elevation = np.deg2rad(55.0 * (1.0 - 2.0 * v))
        self._eye_rays = np.column_stack((
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        )).astype(np.float64)
        self._eye_ids = [int(self.body_ids[self.segment_names.index(f"{side}_eye")])
                         for side in ("l", "r")]
        self._head_id = int(self.body_ids[self.segment_names.index("c_head")])
        self._ray_geomid = np.empty(1, dtype=np.int32)
        self._scene_luminance = np.dot(self.sim.mj_model.geom_rgba[:, :3],
                                       np.array([0.2126, 0.7152, 0.0722]))
        self._scene_geom_names = {
            mj.mj_name2id(self.sim.mj_model, mj.mjtObj.mjOBJ_GEOM, shape["name"]): shape["name"]
            for shape in self.target_shapes
        }
        scene_ids = list(self._scene_geom_names) + list(self.sim._internal_ground_geom_ids)
        self.sim.mj_model.geom_group[scene_ids] = 1
        self._ray_groups = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        self.last_vision = {"left_mean": 0.0, "right_mean": 0.0, "visible_objects": {}}
        self.initial_position = np.zeros(3, dtype=np.float64)
        self._walking_dt = self.sim.timestep
        self._wing_actuator_ids = np.array([mj.mj_name2id(self.sim.mj_model,
            mj.mjtObj.mjOBJ_ACTUATOR, actuator.name) for actuator in self._wing_actuators])
        self._wing_joint_ids = np.array([mj.mj_name2id(self.sim.mj_model,
            mj.mjtObj.mjOBJ_JOINT, joint.name) for joint in self._wing_joints])
        if np.any(self._wing_actuator_ids < 0) or np.any(self._wing_joint_ids < 0):
            raise RuntimeError("Wing actuator or joint name was lost during model composition")
        self._wing_qpos = self.sim.mj_model.jnt_qposadr[self._wing_joint_ids]
        self.reset()

    def skeleton_manifest(self) -> dict:
        segments = []
        for name in self.segment_names:
            mesh, mirror = _mesh_for_segment(name)
            segments.append({"name": name, "mesh": mesh, "mirror": mirror})
        return {
            "segments": segments,
            "mesh_scale": float(MESH_SCALE),
            "thorax_index": self.thorax_index,
            "ground_half_size": GROUND_HALF_SIZE,
            "targets": self.target_shapes,
            "eye_facets": int(len(self._eye_rays)),
        }

    @property
    def position(self) -> np.ndarray:
        return self.sim.mj_data.xpos[self.body_ids[self.thorax_index]].copy()

    def reset(self) -> None:
        self.sim.reset()
        self.controller.reset(seed=0)
        self._control_tick = 0
        initial_action = LocomotionAction(
            joint_angles=self.preprogrammed_steps.default_pose_by_dof_order(self.dof_order),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        apply_locomotion_action(self.sim, self.fly.name, initial_action)
        self.sim.warmup()
        self.initial_position = self.position.copy()
        self.wing_command = WingCommand()
        self._wing_phase = 0.0
        self._wing_envelope = np.zeros(2)
        self._hover_target = None
        self._hover_yaw = 0.0
        self._hover_vz = 0.0
        self.sim.mj_data.ctrl[self._wing_actuator_ids] = 0
        self.sim.mj_data.xfrc_applied[:] = 0

    def step(self, command: MotorCommand, duration_seconds: float, wings: WingCommand | None = None) -> BodyState:
        self.wing_command = wings or WingCommand()
        moving = self.wing_command.frequency > 0 or np.max(self._wing_envelope) > 0.001
        self.sim.mj_model.opt.timestep = min(self._walking_dt, 0.0001) if moving else self._walking_dt
        self.control_stride = max(1, round(self.controller.timestep / self.sim.timestep))
        descending_signal = np.asarray([command.left, command.right], dtype=np.float64)
        steps = max(1, int(round(duration_seconds / self.sim.timestep)))
        for _ in range(steps):
            if self._control_tick % self.control_stride == 0:
                observation = self._observation()
                action = self.controller.step(descending_signal, observation)
                if self._hover_target is not None:
                    action = LocomotionAction(joint_angles=action.joint_angles,
                                              adhesion_onoff=np.zeros(6, dtype=bool))
                apply_locomotion_action(self.sim, self.fly.name, action)
            self._step_wings()
            self._apply_lift()
            self.sim.step()
            self._control_tick += 1
        return self.state()

    def _step_wings(self) -> None:
        # Only actuator targets are set; MuJoCo integrates the physical wing joints.
        # Body lift is a separate engineered force, see _apply_lift.
        dt = self.sim.timestep
        command = self.wing_command
        amplitudes = np.array([command.left_amplitude, command.right_amplitude])
        self._wing_envelope += (1 - np.exp(-dt / 0.02)) * (amplitudes - self._wing_envelope)
        self._wing_phase = (self._wing_phase + 2 * np.pi * command.frequency * dt) % (2 * np.pi)
        spread = np.clip(self._wing_envelope / 0.12, 0, 1) * 1.25
        stroke = self._wing_envelope * np.sin(self._wing_phase)
        self.sim.mj_data.ctrl[self._wing_actuator_ids] = [spread[0], stroke[0], spread[1], stroke[1]]

    def _lift_level(self) -> float:
        return float(np.clip(np.max(self._wing_envelope) / 0.12, 0.0, 1.0))

    def _apply_lift(self) -> None:
        # Not flight physics: while the wings beat, the body is simply placed higher, held
        # level and facing the same way; when they stop it is lowered back onto its legs.
        data = self.sim.mj_data
        base = self.initial_position[2]
        height = data.xpos[self._thorax_id, 2]
        dt = self.sim.timestep
        if np.max(self._wing_envelope) > HOVER_MIN_AMPLITUDE:
            if self._hover_target is None:
                w, x, y, z = data.qpos[self._root_qpos + 3:self._root_qpos + 7]
                self._hover_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                start = height
            else:
                start = self._hover_target
            self._hover_vz = 0.0
            target = min(base + HOVER_RISE, start + HOVER_CLIMB_RATE * dt)
        elif self._hover_target is not None:
            # Wings stopped: drop under gravity, still level, until the legs reach standing height.
            self._hover_vz -= self._gravity * dt
            target = self._hover_target + self._hover_vz * dt
            if target <= base:
                self._hover_target = None
                data.qvel[self._root_dof:self._root_dof + 6] = 0
                return
        else:
            return
        self._hover_target = target
        q = self._root_qpos
        data.qpos[q + 2] += target - height
        data.qpos[q + 3:q + 7] = (np.cos(self._hover_yaw / 2), 0.0, 0.0, np.sin(self._hover_yaw / 2))
        data.qvel[self._root_dof:self._root_dof + 6] = 0

    def wing_state(self) -> dict:
        from dataclasses import asdict
        return {**asdict(self.wing_command),
                "joint_angles": self.sim.mj_data.qpos[self._wing_qpos].tolist(),
                "actuator_torques": self.sim.mj_data.actuator_force[self._wing_actuator_ids].tolist(),
                "lift_level": self._lift_level(),
                "source": "MaleCNS DNg02 spike rates; engineered oscillator and actuator gains; body is placed level 3 mm up while wings beat"}

    def _observation(self):
        data = self.sim.mj_data
        forces = np.zeros((18, 3))
        wrench = np.empty(6)
        contacts = data.contact
        for i in range(data.ncon):
            g1, g2 = contacts.geom[i]
            r1, r2 = self._contact_rows[g1], self._contact_rows[g2]
            if (r1 >= 0 and self._ground_geoms[g2]) or (r2 >= 0 and self._ground_geoms[g1]):
                mj.mj_contactForce(self.sim.mj_model, data, i, wrench)
                force = contacts.frame[i].reshape(3, 3).T @ wrench[:3]
                if r1 >= 0:
                    forces[r1] -= force
                if r2 >= 0:
                    forces[r2] += force
        return HybridControllerObservation(float(data.xpos[self._thorax_id, 2]),
            data.xpos[self._foot_ids, 2], forces.reshape(6, 3, 3),
            data.xmat[self._thorax_id].reshape(3, 3)[:, 0].copy())

    def sensory_luminance(self) -> np.ndarray:
        data = self.sim.mj_data
        model = self.sim.mj_model
        orientation = data.xmat[self._head_id].reshape(3, 3)
        directions = self._eye_rays @ orientation.T
        luminance = np.full(len(directions), 0.75, dtype=np.float32)
        visible_objects = {}
        for side in (0, 1):
            eye_offset = np.array([0.0, 0.15 if side == 0 else -0.15, 0.0])
            origin = data.xpos[self._eye_ids[side]] + orientation @ eye_offset
            for index in np.flatnonzero(self.retina_side == side):
                distance = mj.mj_ray(model, data, origin, directions[index], self._ray_groups,
                                     True, -1, self._ray_geomid)
                if distance >= 0:
                    geom_id = int(self._ray_geomid[0])
                    luminance[index] = self._scene_luminance[geom_id]
                    name = self._scene_geom_names.get(geom_id)
                    if name:
                        visible_objects[name] = visible_objects.get(name, 0) + 1
        self.last_vision = {
            "left_mean": float(luminance[self.retina_side == 0].mean()) if np.any(self.retina_side == 0) else 0.0,
            "right_mean": float(luminance[self.retina_side == 1].mean()) if np.any(self.retina_side == 1) else 0.0,
            "visible_objects": visible_objects,
        }
        return luminance

    def pose(self) -> dict:
        positions = self.sim.mj_data.xpos[self.body_ids]
        quaternions = self.sim.mj_data.xquat[self.body_ids]
        return {
            "positions": np.round(positions, 4).tolist(),
            "quaternions": np.round(quaternions, 5).tolist(),
        }

    def state(self) -> BodyState:
        position = self.position
        displacement = position - self.initial_position
        return BodyState(
            simulated_seconds=float(self.sim.mj_data.time),
            position=tuple(float(value) for value in position),
            displacement=tuple(float(value) for value in displacement),
        )
