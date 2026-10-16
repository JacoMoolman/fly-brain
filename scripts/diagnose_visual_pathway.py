"""Observe sensory propagation and neural suppression without body commands.

All ablations operate on this script's separate brain instance. The application,
prepared graph, and motor decoder are never edited by this diagnostic.
"""
import argparse
import json
import time
from dataclasses import asdict

import mujoco as mj
import numpy as np

from flybrain_local.body import FlyGymBody
from flybrain_local.brain import MaleCNSBrain
from flybrain_local.paths import ProjectPaths


paths = ProjectPaths.discover()
parser = argparse.ArgumentParser()
parser.add_argument("--cases", nargs="*", default=[])
parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
args = parser.parse_args()
suffix = "-cpu" if args.backend == "cpu" else ""
output = paths.results / "verification" / f"visual-pathway-diagnosis{suffix}.json"
brain = MaleCNSBrain(paths.prepared / "brain_graph.npz", backend=args.backend)
body = FlyGymBody(brain.retina_uv, brain.retina_side)
scene = body.sensory_luminance().copy()
initial_qpos = body.sim.mj_data.qpos.copy()
views = []
for yaw in np.linspace(-np.pi, np.pi, 12, endpoint=False):
    body.sim.mj_data.qpos[:] = initial_qpos
    body.sim.mj_data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
    views.append(body.sensory_luminance().copy())

types, type_inverse = np.unique(brain.cell_type, return_inverse=True)
classes, class_inverse = np.unique(brain.superclass, return_inverse=True)
selected = {name: np.flatnonzero(brain.cell_type == name)
            for name in ("R1-R6", "L1", "L2", "L3", "L4", "L5", "DNa02", "DNp09", "oDN1", "MDN",
                         "CL366", "LAL206", "PS059", "MBON31", "MBON32")}
incoming_edges = np.flatnonzero(np.isin(brain.post, brain.readout_indices))
incoming_sources = np.searchsorted(brain.ptr, incoming_edges, side="right") - 1
incoming_targets = brain.post[incoming_edges]
retinal_mask = np.zeros(brain.n, dtype=bool)
retinal_mask[brain.retina] = True
retinal_edges = np.concatenate([np.arange(brain.ptr[i], brain.ptr[i + 1]) for i in brain.retina])
negative_readout_edges = incoming_edges[brain.weight[incoming_edges] < 0]
original_retinal_weights = brain.weight[retinal_edges].copy()
original_readout_weights = brain.weight[negative_readout_edges].copy()

report = {
    "backend": brain.backend,
    "readout_population_sizes": {k: len(v) for k, v in selected.items()},
    "retinal_edge_signs": {
        "positive": int(np.sum(original_retinal_weights > 0)),
        "negative": int(np.sum(original_retinal_weights < 0)),
        "zero": int(np.sum(original_retinal_weights == 0)),
    },
    "cases": {},
}
if args.cases and output.exists():
    report["cases"] = json.loads(output.read_text())["cases"]

lamina_gain = 1.0
original_stimulate = brain._stimulate


def diagnostic_stimulate(stimulus):
    brain.drive[brain.lamina] *= lamina_gain
    original_stimulate(stimulus)


brain._stimulate = diagnostic_stimulate


def sync_weights():
    if brain._gpu is not None:
        brain._gpu.weight.assign(brain.weight)


def populations():
    if brain._gpu is not None:
        brain._gpu.sync_state(brain)
    return {
        name: {"neurons": len(idx), "rate_hz": float(brain.filtered_rates[idx].mean()),
               "voltage_mean": float(brain.voltage[idx].mean()),
               "voltage_min": float(brain.voltage[idx].min()),
               "voltage_max": float(brain.voltage[idx].max()),
               "synaptic_state_mean": float(brain.conductance[idx].mean()),
               "at_inhibitory_floor": int(np.sum(brain.voltage[idx] <= -74.99))}
        for name, idx in selected.items() if len(idx)
    }


def run_case(name, input_mode, disable_retinal_edges=False, disable_readout_inhibition=False,
             lamina_background=1.0):
    global lamina_gain
    lamina_gain = lamina_background
    brain.weight[retinal_edges] = original_retinal_weights
    brain.weight[negative_readout_edges] = original_readout_weights
    if disable_retinal_edges:
        brain.weight[retinal_edges] = 0
    if disable_readout_inhibition:
        brain.weight[negative_readout_edges] = 0
    sync_weights()
    brain.reset()
    early = np.zeros(brain.n, np.int64)
    late = np.zeros(brain.n, np.int64)
    trace = []
    started = time.perf_counter()
    for step in range(200):
        frame = scene if input_mode == "scene" else (
            views[(step // 4) % len(views)] if input_mode == "moving" else np.zeros_like(scene))
        brain.step(frame, duration_ms=6)
        if step < 50:
            early += brain.counts
        if step >= 100:
            late += brain.counts
        if step + 1 in (10, 20, 50, 100, 200):
            sample = {"ms": brain.simulated_ms, "motor": asdict(brain.decode_motor(0)),
                      "populations": populations()}
            trace.append(sample)
    source_activity = late[incoming_sources] / 0.6
    offered = source_activity * brain.weight[incoming_edges]
    readout_inputs = {}
    for neuron in brain.readout_indices:
        rows = np.flatnonzero(incoming_targets == neuron)
        ranked = rows[np.argsort(np.abs(offered[rows]))[::-1][:5]]
        readout_inputs[str(brain.ids[neuron])] = {
            "type": str(brain.cell_type[neuron]), "side": str(brain.soma_side[neuron]),
            "voltage": float(brain.voltage[neuron]),
            "synaptic_state": float(brain.conductance[neuron]),
            "late_spikes": int(late[neuron]),
            "offered_excitation_per_second": float(offered[rows][offered[rows] > 0].sum()),
            "offered_inhibition_per_second": float(offered[rows][offered[rows] < 0].sum()),
            "top_sources": [{"id": str(brain.ids[incoming_sources[r]]),
                             "type": str(brain.cell_type[incoming_sources[r]]),
                             "rate_hz": float(source_activity[r]),
                             "weight": float(brain.weight[incoming_edges[r]]),
                             "signed_weight_times_rate": float(offered[r])} for r in ranked],
        }
    totals_by_type = np.bincount(type_inverse, weights=late, minlength=len(types))
    totals_by_class = np.bincount(class_inverse, weights=late, minlength=len(classes))
    ranking = np.argsort(totals_by_type)[::-1][:20]
    output_activity = {}
    for category in ("descending_neuron", "vnc_motor"):
        neurons = np.flatnonzero(brain.superclass == category)
        active = neurons[late[neurons] > 0]
        ranking_out = active[np.argsort(late[active])[::-1][:12]]
        output_activity[category] = {
            "total_neurons": len(neurons), "spiking_neurons": len(active),
            "top_neurons": [{"id": str(brain.ids[i]), "type": str(brain.cell_type[i]),
                             "rate_hz": float(late[i] / 0.6)} for i in ranking_out],
        }
    result = {
        "trace": trace,
        "early_ms": 300, "late_ms": 600,
        "early_spikes": int(early.sum()), "late_spikes": int(late.sum()),
        "late_spiking_neurons": int(np.count_nonzero(late)),
        "late_selected_spikes": {k: int(late[v].sum()) for k, v in selected.items()},
        "late_spikes_by_superclass": {str(k): int(v) for k, v in zip(classes, totals_by_class)},
        "most_active_types": {str(types[i]): int(totals_by_type[i]) for i in ranking},
        "readout_inputs": readout_inputs,
        "other_output_activity": output_activity,
        "wall_seconds": time.perf_counter() - started,
    }
    report["cases"][name] = result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": name, "late_spikes": result["late_spikes"],
                      "late_selected_spikes": result["late_selected_spikes"],
                      "end_populations": trace[-1]["populations"]}), flush=True)


print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
cases = {
    "scene": {"input_mode": "scene"},
    "dark": {"input_mode": "dark"},
    "changing_view": {"input_mode": "moving"},
    "retinal_outputs_disconnected": {"input_mode": "scene", "disable_retinal_edges": True},
    "readout_inhibition_removed": {"input_mode": "scene", "disable_readout_inhibition": True},
    "lamina_background_removed": {"input_mode": "scene", "lamina_background": 0.0},
}
for name in args.cases or cases:
    run_case(name, **cases[name])
print(f"Saved {output}", flush=True)
