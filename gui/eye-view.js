/* Fly vision panel: eye luminance and facets landing on each scene object, both measured by the eye rays. */
class EyeView {
  constructor() {
    this.facets = 0;
    this.bars = new Map();
  }

  setLayout(manifest) {
    this.facets = Number(manifest.eye_facets) || 0;
    document.getElementById("eyeRayCount").textContent =
      this.facets ? `bar: vs most seen · %: of ${this.facets.toLocaleString()} facets` : "restart server for %";
    this.buildBars(manifest.targets || []);
  }

  buildBars(targets) {
    const container = document.getElementById("objectBars");
    container.textContent = "";
    this.bars.clear();
    const groups = new Map();
    const groupOf = (name) => {
      if (name.startsWith("wall_")) return "walls";
      if (name.startsWith("tree_")) return "trees";
      if (name.startsWith("rock_")) return "rocks";
      if (name.startsWith("mushroom_")) return "mushroom";
      return name;
    };
    for (const target of targets) {
      const key = groupOf(target.name);
      if (!groups.has(key)) groups.set(key, { names: [], color: target.color });
      const group = groups.get(key);
      group.names.push(target.name);
      if (/_(crown|cap)$/.test(target.name)) group.color = target.color;
    }
    for (const [key, group] of groups) {
      const row = document.createElement("div");
      row.className = "object-bar";
      const label = document.createElement("span");
      label.textContent = key.replace(/_/g, " ");
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = "fill";
      const [r, g, b] = group.color.map((value) => Math.round(value * 255));
      fill.style.background = `rgb(${r}, ${g}, ${b})`;
      track.append(fill);
      const value = document.createElement("output");
      value.textContent = "0%";
      row.append(label, track, value);
      container.append(row);
      this.bars.set(key, { names: group.names, row, fill, value });
    }
  }

  update(vision) {
    if (!vision) return;
    for (const [side, mean] of [["Left", vision.left_mean], ["Right", vision.right_mean]]) {
      if (Number.isFinite(mean)) document.getElementById(`lum${side}Value`).textContent = mean.toFixed(2);
    }
    if (!this.facets) return;
    const counts = vision.visible_objects || {};
    const hitsByBar = new Map();
    for (const [key, bar] of this.bars) {
      hitsByBar.set(key, bar.names.reduce((sum, name) => sum + (counts[name] || 0), 0));
    }
    // Bars are scaled to the most-seen object (at least 1% of facets) so small objects stay visible.
    const scale = Math.max(...hitsByBar.values(), this.facets * 0.01);
    for (const [key, bar] of this.bars) {
      const hits = hitsByBar.get(key);
      const percent = hits / this.facets * 100;
      bar.fill.style.width = `${hits / scale * 100}%`;
      bar.value.textContent = `${percent < 10 ? percent.toFixed(1) : Math.round(percent)}%`;
      bar.row.classList.toggle("seen", hits > 0);
    }
  }
}
