from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict

import numpy as np

from .body import FlyGymBody
from .brain import MaleCNSBrain, MotorCommand
from .paths import ProjectPaths

# Biological time advanced per closed-loop iteration. Wall time per iteration scales
# with this, so a smaller slice means more frequent (smoother) visual updates at the
# same simulated-per-wall-second rate.
SLICE_MS = 6.0

# A button press drives its target descending neurons for this much fly-time. A single
# 3 ms pulse on two neurons is invisible by the time it reaches the motor decoder, so
# the stimulus is held long enough to actually build a firing rate.
STIMULUS_MS = 300.0


class IntegratedRuntime:
    def __init__(self):
        self.paths = ProjectPaths.discover()
        self.manifest = json.loads((self.paths.prepared / "manifest.json").read_text())
        self.brain: MaleCNSBrain | None = None
        self.body: FlyGymBody | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.paused = False
        self.wing_stimulus = False
        self.assist = 0.0
        self.pending_stimulus: str | None = None
        self.held_stimulus: str | None = None
        self.stimulus_ms_left = 0.0
        self.reset_requested = False
        self.error: str | None = None
        self.iteration = 0
        self.started_at = time.time()
        self.ready = False
        self.body_wall_seconds = 0.0
        self.latest = {
            "status": "loading",
            "error": None,
            "dataset": self.manifest["dataset"],
            "sex": self.manifest["sex"],
            "neurons": self.manifest["neurons"],
            "edges": self.manifest["retained_edge_rows"],
            "synaptic_contacts": self.manifest["retained_synaptic_contacts"],
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="male-cns-flygym", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def control(self, *, paused=None, assist=None, stimulus=None, reset=False, wing_stimulus=None) -> dict:
        with self._lock:
            if wing_stimulus is not None:
                self.wing_stimulus = bool(wing_stimulus)
            if paused is not None:
                self.paused = bool(paused)
            if assist is not None:
                self.assist = float(np.clip(assist, 0.0, 1.2))
            if stimulus is not None:
                allowed = {"loom", "forward", "backward", "left", "right", "none"}
                if stimulus not in allowed:
                    raise ValueError(f"Unknown stimulus: {stimulus}")
                self.pending_stimulus = None if stimulus == "none" else stimulus
                if stimulus == "none":
                    self.held_stimulus = None
                    self.stimulus_ms_left = 0.0
            if reset:
                self.reset_requested = True
            return {"paused": self.paused, "assist": self.assist, "wing_stimulus": self.wing_stimulus}

    def _snapshot(self, command: MotorCommand, body_state, activity, loop_seconds, sensory_mean):
        return {
            "status": "error" if self.error else ("paused" if self.paused else "running"),
            "error": self.error,
            "dataset": self.manifest["dataset"],
            "sex": self.manifest["sex"],
            "neurons": self.manifest["neurons"],
            "edges": self.manifest["retained_edge_rows"],
            "synaptic_contacts": self.manifest["retained_synaptic_contacts"],
            "brain": {
                "simulated_ms": round(self.brain.simulated_ms, 1),
                "total_spikes": self.brain.total_spikes,
                "frame_spikes": int(self.brain.counts.sum()),
                "active_neurons": int(self.brain.active_count[0]),
                "step_wall_ms": round(self.brain.last_wall_seconds * 1000, 2),
                "backend": self.brain.backend,
                "backend_error": self.brain.backend_error,
                "sensory_mean": round(float(sensory_mean), 4),
                "activity": [],
                "activity_binary": activity if isinstance(activity, str) else "",
            },
            "motor": asdict(command),
            "vision": self.body.last_vision,
            "body": {**asdict(body_state), "pose": self.body.pose(), "wings": self.body.wing_state()},
            "control": {
                "wing_stimulus": self.wing_stimulus,
                "paused": self.paused,
                "assist": self.assist,
                "stimulus": self.held_stimulus,
                "stimulus_ms_left": round(max(self.stimulus_ms_left, 0.0), 1),
            },
            "loop": {
                "iteration": self.iteration,
                "wall_ms": round(loop_seconds * 1000, 2),
                "realtime_factor": round(SLICE_MS / (loop_seconds * 1000), 3) if loop_seconds else 0,
                "body_wall_ms": round(self.body_wall_seconds * 1000, 2),
                "uptime_seconds": round(time.time() - self.started_at, 1),
            },
            "disclaimer": "Full MaleCNS wiring with modeled LIF dynamics and an engineered brain-body bridge.",
        }

    def _run(self) -> None:
        try:
            self.brain = MaleCNSBrain(self.paths.prepared / "brain_graph.npz")
            self.body = FlyGymBody(self.brain.retina_uv, self.brain.retina_side)
            with self._lock:
                self.latest = self._snapshot(
                    self.brain.decode_motor(self.assist), self.body.state(), [], 0.0, 0.0
                )
                self.ready = True
        except Exception:
            self.error = traceback.format_exc()
            with self._lock:
                self.latest["status"] = "error"
                self.latest["error"] = self.error
            return
        luminance = np.full(len(self.brain.retina), 0.35, dtype=np.float32)
        try:
            while not self._stop.is_set():
                if self.paused and not self.reset_requested:
                    time.sleep(0.05)
                    continue
                started = time.perf_counter()
                with self._lock:
                    if self.reset_requested:
                        self.brain.reset()
                        self.body.reset()
                        self.wing_stimulus = False
                        self.iteration = 0
                        self.reset_requested = False
                        self.pending_stimulus = None
                        self.held_stimulus = None
                        self.stimulus_ms_left = 0.0
                        luminance = self.body.sensory_luminance()
                        self.latest = self._snapshot(
                            self.brain.decode_motor(self.assist), self.body.state(), [], 0, luminance.mean())
                        if self.paused:
                            continue
                    if self.pending_stimulus:
                        self.held_stimulus = self.pending_stimulus
                        self.stimulus_ms_left = STIMULUS_MS
                        self.pending_stimulus = None
                    stimulus = None
                    if self.stimulus_ms_left > 0:
                        stimulus = self.held_stimulus
                        self.stimulus_ms_left -= SLICE_MS
                        if self.stimulus_ms_left <= 0:
                            self.held_stimulus = None
                    assist = self.assist
                if self.iteration % 4 == 0:
                    luminance = self.body.sensory_luminance()
                self.brain.step(luminance, duration_ms=SLICE_MS, stimulus=stimulus, wing_stimulus=self.wing_stimulus)
                # The legs follow the decoded descending-neuron activity. No manual
                # override: a button injects current into its target DNs and whatever
                # the network does with it is what moves the fly.
                command = self.brain.decode_motor(assist)
                body_started = time.perf_counter()
                body_state = self.body.step(command, SLICE_MS / 1000.0, self.brain.decode_wings())
                self.body_wall_seconds = time.perf_counter() - body_started
                self.iteration += 1
                activity = self.brain.activity_binary()
                with self._lock:
                    self.latest = self._snapshot(
                        command,
                        body_state,
                        activity,
                        time.perf_counter() - started,
                        float(luminance.mean()),
                    )
        except Exception:
            self.error = traceback.format_exc()
            with self._lock:
                self.latest["status"] = "error"
                self.latest["error"] = self.error

    def state(self) -> dict:
        with self._lock:
            snapshot = dict(self.latest)
            snapshot["status"] = "error" if self.error else (
                "paused" if self.paused and self.ready else self.latest["status"])
            if self.ready and not self.error and not self.paused:
                snapshot["status"] = "running"
            snapshot["control"] = {"wing_stimulus": self.wing_stimulus, "paused": self.paused, "assist": self.assist,
                                   "stimulus": self.held_stimulus,
                                   "queued_stimulus": self.pending_stimulus,
                                   "stimulus_ms_left": max(0, self.stimulus_ms_left)}
            return snapshot

    def layout(self) -> dict | None:
        if self.brain is None:
            return None
        payload = self.brain.layout_payload()
        payload["manifest"] = self.manifest
        return payload

    def skeleton(self) -> dict | None:
        if self.body is None:
            return None
        return self.body.skeleton_manifest()
