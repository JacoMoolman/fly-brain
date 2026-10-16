"""Check scene sampling and observe unassisted, unstimulated behavior."""

import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from flybrain_local.body import FlyGymBody, WALL_INNER_LIMIT
from flybrain_local.brain import MaleCNSBrain, MotorCommand
from flybrain_local.paths import ProjectPaths
from flybrain_local.runtime import IntegratedRuntime


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=40)
parser.add_argument("--check-buttons", action="store_true")
args = parser.parse_args()
paths = ProjectPaths.discover()
brain = MaleCNSBrain(paths.prepared / "brain_graph.npz", backend="cpu")
body = FlyGymBody(brain.retina_uv, brain.retina_side)
initial = body.sensory_luminance()
first_objects = body.last_vision["visible_objects"].copy()
assert len(initial) == len(brain.retina)
assert np.std(initial) > 0.01
assert first_objects.get("apple", 0) > 0
assert first_objects.get("banana", 0) > 0
assert np.bincount(brain.retina_side, minlength=2).min() > 100
for name in ("wall_north", "wall_south", "wall_east", "wall_west"):
    geom = mj.mj_name2id(body.sim.mj_model, mj.mjtObj.mjOBJ_GEOM, name)
    assert body.sim.mj_model.geom_contype[geom] != 0
    assert body.sim.mj_model.geom_conaffinity[geom] != 0

body.sim.mj_data.qpos[3:7] = [0, 0, 0, 1]
mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
rotated = body.sensory_luminance()
assert body.last_vision["visible_objects"].get("apple", 0) == 0
image_change = float(np.mean(np.abs(initial - rotated)))
print(json.dumps({"image_change": image_change}), flush=True)
assert image_change > 0.001


def neural_response(image):
    brain.reset()
    totals = np.zeros(3, dtype=np.int64)
    for _ in range(40):
        brain.step(image, duration_ms=6, stimulus=None)
        totals += [brain.counts[brain.retina].sum(), brain.counts[brain.lamina].sum(),
                   brain.counts[brain.readout_indices].sum()]
    return totals.tolist()


first_response = neural_response(initial)
rotated_response = neural_response(rotated)
assert first_response[0] != rotated_response[0]
print(json.dumps({"neural_counts_initial": first_response,
                  "neural_counts_rotated": rotated_response}), flush=True)

body.reset()
body.sim.mj_data.qpos[0] += 17.0
mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
wall_test = body.step(MotorCommand(0.8, 0.8, 0, 0, 0, 0), 0.6)
print(json.dumps({"wall_test_x": wall_test.position[0], "contacts": body.sim.mj_data.ncon}), flush=True)
assert wall_test.position[0] < WALL_INNER_LIMIT + 0.5

runtime = IntegratedRuntime()
runtime.start()
samples = []
button_result = None
try:
    deadline = time.monotonic() + max(240, args.iterations * 2)
    last_iteration = -1
    while time.monotonic() < deadline:
        state = runtime.state()
        if state["status"] == "error":
            raise RuntimeError(state["error"])
        iteration = state.get("loop", {}).get("iteration", 0)
        if iteration >= 1 and iteration != last_iteration and iteration % 25 == 0:
            sample = {
                "iteration": iteration,
                "simulated_ms": state["brain"]["simulated_ms"],
                "retina_spikes": int(runtime.brain.counts[runtime.brain.retina].sum()),
                "motor": state["motor"],
                "position": state["body"]["position"],
                "displacement": state["body"]["displacement"],
                "vision": state["vision"],
            }
            samples.append(sample)
            print(json.dumps(sample), flush=True)
            last_iteration = iteration
        if iteration >= args.iterations:
            break
        time.sleep(0.05)
    else:
        raise TimeoutError(f"Simulation did not reach {args.iterations} closed-loop slices")
    assert runtime.assist == 0
    assert runtime.held_stimulus is None
    assert all(abs(value) <= WALL_INNER_LIMIT for value in state["body"]["position"][:2])
    if args.check_buttons:
        start_iteration = state["loop"]["iteration"]
        start_x = state["body"]["position"][0]
        runtime.control(stimulus="forward")
        peak_drive = 0.0
        while time.monotonic() < deadline and runtime.state()["loop"]["iteration"] < start_iteration + 50:
            button_state = runtime.state()
            peak_drive = max(peak_drive, button_state["motor"]["left"], button_state["motor"]["right"])
            time.sleep(0.02)
        button_result = {"peak_forward_drive": peak_drive,
                         "x_displacement": button_state["body"]["position"][0] - start_x}
        print(json.dumps({"button_check": button_result}), flush=True)
        assert peak_drive > 0.05
finally:
    runtime.stop()

report = {"initial_visible_objects": first_objects,
          "neural_counts_initial": first_response,
          "neural_counts_rotated": rotated_response,
          "wall_test_x": wall_test.position[0],
          "button_check": button_result,
          "samples": samples,
          "final_iteration": state["loop"]["iteration"]}
output = paths.results / "verification" / "vision-observation.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(f"Saved {output}", flush=True)
