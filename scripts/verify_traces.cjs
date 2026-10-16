// Numerical and lifecycle checks for the live graph instruments.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const elements = new Map();
const context = vm.createContext({
  performance: { now: () => 0 },
  ResizeObserver: class { observe() {} },
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, { getContext: () => ({}), addEventListener() {} });
      return elements.get(id);
    },
  },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../gui/live-traces.js"), "utf8"), context);
const traces = vm.runInContext("new LiveTraces()", context);
const pairs = new Uint32Array(20);
for (let i = 0; i < 10; i++) { pairs[i * 2] = i; pairs[i * 2 + 1] = 1; }
const packet = Buffer.from(pairs.buffer).toString("base64");
function state(ms, total, status = "running") {
  return { status, neurons: 1000, brain: { simulated_ms: ms, total_spikes: total, activity_binary: packet },
    motor: { left: 0.2, right: 1.2 }, control: { stimulus: "left" } };
}
traces.append(state(0, 0), 0);
traces.append(state(6, 10), 0.1);
assert.equal(traces.history.length, 1);
assert.ok(Math.abs(traces.history[0].rate - 10 / 0.006 / 1000) < 1e-10);
assert.equal(traces.history[0].firing, 1);
traces.append(state(6, 10, "paused"), 0.5);
assert.equal(traces.history.length, 1, "Pause must not invent samples");
traces.append(state(18, 34), 2.5);
assert.ok(Math.abs(traces.history[1].rate - 2) < 1e-10, "Skipped frames use cumulative totals");
assert.equal(traces.history[1].gap, true, "Missing time is not connected as observed data");
assert.equal(traces.history[1].left, 0.2);
assert.equal(traces.history[1].right, 1.2);
traces.append(state(0, 0), 3);
assert.equal(traces.history.length, 0, "Reset must clear previous-run history");
for (let i = 1; i <= 8200; i++) traces.append(state(i * 6, i * 10), 3 + i / 1000);
assert.equal(traces.history.length, 8000);
assert.ok(traces.niceCeiling(9.8) >= 9.8);
assert.equal(traces.niceCeiling(0), 1);
console.log("Live traces: units, binary counts, skipped frames, pause, reset, gaps and bounded history passed.");
