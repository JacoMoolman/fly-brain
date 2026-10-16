/* Live instruments for the existing simulation. See live-traces.md for units. */
const GREEN_SHADES = [
  [0, "20, 110, 50"],
  [0.5, "40, 225, 105"],
  [1, "200, 255, 210"],
];

class LiveTraces {
  constructor() {
    this.history = [];
    this.previous = null;
    this.windowSeconds = 300;
    this.lastDraw = 0;
    this.dirty = true;
    this.hidden = false;
    this.status = "loading";
    this.charts = [
      {
        id: "brainTrace", kind: "brain",
        series: [
          { key: "rate", alpha: 1, width: 1.8, dash: [] },
          { key: "firing", alpha: 0.45, width: 1.2, dash: [] },
        ],
      },
      {
        id: "motorTrace", kind: "motor",
        series: [
          { key: "left", alpha: 1, width: 1.8, dash: [] },
          { key: "right", alpha: 0.55, width: 1.4, dash: [5, 3] },
        ],
      },
    ];
    for (const chart of this.charts) {
      chart.canvas = document.getElementById(chart.id);
      chart.context = chart.canvas.getContext("2d");
      chart.canvas.addEventListener("mousemove", (event) => {
        chart.hoverX = event.clientX - chart.canvas.getBoundingClientRect().left;
        this.dirty = true;
      });
      chart.canvas.addEventListener("mouseleave", () => {
        chart.hoverX = null;
        this.dirty = true;
      });
    }
    document.getElementById("traceWindow").addEventListener("change", (event) => {
      this.windowSeconds = Number(event.target.value);
      this.dirty = true;
      this.updateClock();
    });
    document.getElementById("toggleTraces").addEventListener("click", (event) => {
      this.hidden = !this.hidden;
      document.body.classList.toggle("traces-hidden", this.hidden);
      event.target.textContent = this.hidden ? "Show graphs" : "Hide graphs";
      event.target.setAttribute("aria-expanded", String(!this.hidden));
      this.dirty = true;
    });
    new ResizeObserver(() => { this.dirty = true; }).observe(document.getElementById("tracePanels"));
  }

  updateClock() {
    const state = this.status === "paused" ? "Paused" : this.status === "running" ? "Live" : "Waiting";
    document.getElementById("traceClock").textContent =
      `${state} · last ${this.windowSeconds} s of wall time · rates use fly time`;
  }

  append(state, wallSeconds = performance.now() / 1000) {
    this.status = state.status;
    this.updateClock();
    const brain = state.brain;
    const simulated = Number(brain.simulated_ms) / 1000;
    const total = Number(brain.total_spikes);
    const neurons = Number(state.neurons);
    if (![simulated, total, neurons].every(Number.isFinite) || neurons <= 0) return;
    const current = { simulated, total, wallSeconds };
    if (!this.previous || simulated < this.previous.simulated || total < this.previous.total) {
      this.history.length = 0;
      this.previous = current;
      this.dirty = true;
      for (const id of ["brainTraceValue", "motorTraceValue"]) {
        document.getElementById(id).textContent = "Collecting";
      }
      return;
    }
    if (simulated === this.previous.simulated) return;
    const dt = simulated - this.previous.simulated;
    const encoded = brain.activity_binary || "";
    const padding = encoded.endsWith("==") ? 2 : encoded.endsWith("=") ? 1 : 0;
    const firingCount = encoded ? (encoded.length * 3 / 4 - padding) / 8 : (brain.activity || []).length;
    const point = {
      wall: wallSeconds, simulated,
      rate: (total - this.previous.total) / dt / neurons,
      firing: firingCount / neurons * 100,
      left: Number(state.motor.left), right: Number(state.motor.right),
      gap: wallSeconds - this.previous.wallSeconds > 1,
      command: state.control.stimulus || "brain control",
    };
    this.previous = current;
    if (![point.rate, point.firing, point.left, point.right].every(Number.isFinite)) return;
    this.history.push(point);
    while (this.history.length > 30000 || this.history[0].wall < wallSeconds - 301) this.history.shift();
    document.getElementById("brainTraceValue").textContent = `${point.rate.toFixed(2)} Hz · ${point.firing.toFixed(2)}%`;
    document.getElementById("motorTraceValue").textContent = `L ${point.left.toFixed(2)}  R ${point.right.toFixed(2)}`;
    this.dirty = true;
  }

  range(points, keys, nonNegative, cap) {
    const values = points.flatMap((point) => keys.map((key) => point[key]));
    let low = values.length ? Math.min(...values) : 0;
    let high = values.length ? Math.max(...values) : 1;
    const span = Math.max(high - low, Math.abs(high) * 0.02, 1e-3);
    low -= span * 0.08;
    high += span * 0.08;
    if (nonNegative) low = Math.max(0, low);
    return { low, high: Math.min(cap, high) };
  }

  draw(timestamp = performance.now()) {
    if (this.hidden || !this.dirty || timestamp - this.lastDraw < 100) return;
    this.lastDraw = timestamp;
    this.dirty = false;
    const end = this.history.length ? this.history[this.history.length - 1].wall : performance.now() / 1000;
    const start = end - this.windowSeconds;
    const visible = this.history.filter((point) => point.wall >= start);
    for (const chart of this.charts) this.drawChart(chart, visible, start, end);
  }

  drawChart(chart, points, start, end) {
    const canvas = chart.canvas;
    const ctx = chart.context;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const pixelsW = Math.round(width * ratio), pixelsH = Math.round(height * ratio);
    if (canvas.width !== pixelsW || canvas.height !== pixelsH) {
      canvas.width = pixelsW; canvas.height = pixelsH;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const dual = chart.kind === "brain";
    const left = 38, right = width - (dual ? 40 : 12), top = 8, bottom = height - 20;
    if (right <= left || bottom <= top) return;

    const shared = this.range(points, chart.series.map((s) => s.key), false, Infinity);
    const scales = chart.series.map((s) =>
      dual ? this.range(points, [s.key], true, s.key === "firing" ? 100 : Infinity) : shared);
    const x = (time) => left + (time - start) / (end - start) * (right - left);
    const yOf = (scale) => (value) => bottom - (value - scale.low) / (scale.high - scale.low) * (bottom - top);

    ctx.save();
    ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();

    chart.series.forEach((s, index) => {
      if (!points.length) return;
      const y = yOf(scales[index]);
      const shade = ctx.createLinearGradient(0, bottom, 0, top);
      for (const [stop, rgb] of GREEN_SHADES) shade.addColorStop(stop, `rgba(${rgb}, ${s.alpha})`);
      ctx.strokeStyle = shade;
      ctx.lineWidth = s.width;
      ctx.lineJoin = "round";
      ctx.setLineDash(s.dash);
      ctx.shadowColor = `rgba(60, 255, 120, ${0.8 * s.alpha})`;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      points.forEach((point, i) => {
        if (i === 0 || point.gap) ctx.moveTo(x(point.wall), y(point[s.key]));
        else ctx.lineTo(x(point.wall), y(point[s.key]));
      });
      ctx.stroke();
    });
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);

    ctx.globalCompositeOperation = "destination-out";
    const fade = ctx.createLinearGradient(left, 0, right, 0);
    fade.addColorStop(0, "rgba(0, 0, 0, 0.8)");
    fade.addColorStop(0.6, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = fade;
    ctx.fillRect(left, top, right - left, bottom - top);

    ctx.globalCompositeOperation = "destination-over";
    const cell = 12;
    for (let gx = left, n = 0; gx <= right; gx += cell, n++) {
      ctx.strokeStyle = n % 5 === 0 ? "rgba(60, 255, 120, 0.13)" : "rgba(60, 255, 120, 0.05)";
      ctx.beginPath(); ctx.moveTo(gx + 0.5, top); ctx.lineTo(gx + 0.5, bottom); ctx.stroke();
    }
    for (let gy = bottom, n = 0; gy >= top; gy -= cell, n++) {
      ctx.strokeStyle = n % 5 === 0 ? "rgba(60, 255, 120, 0.13)" : "rgba(60, 255, 120, 0.05)";
      ctx.beginPath(); ctx.moveTo(left, gy - 0.5); ctx.lineTo(right, gy - 0.5); ctx.stroke();
    }
    ctx.fillStyle = "#020a05";
    ctx.fillRect(left, top, right - left, bottom - top);
    ctx.restore();

    ctx.font = "9px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    const label = (scale, x0, align, suffix) => {
      const digits = scale.high - scale.low < 1 ? 3 : scale.high - scale.low < 10 ? 2 : 1;
      const y = yOf(scale);
      ctx.textAlign = align;
      for (const value of [scale.low, (scale.low + scale.high) / 2, scale.high]) {
        ctx.fillText(`${Number(value.toFixed(digits))}${suffix}`, x0, y(value));
      }
    };
    ctx.fillStyle = "#4fae6c";
    label(scales[0], left - 6, "right", "");
    if (dual) {
      ctx.fillStyle = "#2f7a48";
      label(scales[1], right + 6, "left", "%");
    }
    ctx.fillStyle = "#4fae6c";
    for (let i = 0; i <= 2; i++) {
      ctx.textAlign = i === 0 ? "left" : i === 2 ? "right" : "center";
      ctx.fillText(i === 2 ? "now" : `-${this.windowSeconds * (1 - i / 2)} s`,
        left + i / 2 * (right - left), height - 7);
    }
    if (!points.length) {
      ctx.textAlign = "center"; ctx.fillText("Waiting for live samples", (left + right) / 2, (top + bottom) / 2);
      return;
    }

    const latest = points[points.length - 1];
    chart.series.forEach((s, index) => {
      const y = yOf(scales[index]);
      ctx.fillStyle = `rgba(210, 255, 220, ${s.alpha})`;
      ctx.shadowColor = "rgba(60, 255, 120, 0.9)";
      ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.arc(x(latest.wall), y(latest[s.key]), 2.5, 0, Math.PI * 2); ctx.fill();
    });
    ctx.shadowBlur = 0;

    if (chart.hoverX != null && chart.hoverX >= left && chart.hoverX <= right) {
      const time = start + (chart.hoverX - left) / (right - left) * (end - start);
      const nearest = points.reduce((best, point) => Math.abs(point.wall - time) < Math.abs(best.wall - time) ? point : best);
      ctx.strokeStyle = "rgba(120, 255, 160, 0.5)"; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x(nearest.wall), top); ctx.lineTo(x(nearest.wall), bottom); ctx.stroke();
      ctx.setLineDash([]);
      const text = chart.kind === "motor" ? `L ${nearest.left.toFixed(2)} / R ${nearest.right.toFixed(2)}` :
        `${nearest.rate.toFixed(2)} Hz · ${nearest.firing.toFixed(2)}% firing`;
      const boxW = Math.min(200, right - left);
      const boxX = Math.min(Math.max(left, x(nearest.wall) + 8), right - boxW);
      ctx.fillStyle = "#04120a"; ctx.fillRect(boxX, top, boxW, 40);
      ctx.strokeStyle = "#1f6b3a"; ctx.strokeRect(boxX, top, boxW, 40);
      ctx.fillStyle = "#c8ffd6"; ctx.textAlign = "left";
      ctx.fillText(text, boxX + 7, top + 12);
      ctx.fillStyle = "#4fae6c";
      ctx.fillText(`Fly ${nearest.simulated.toFixed(3)} s · ${nearest.command}`, boxX + 7, top + 28, boxW - 14);
    }
  }
}
