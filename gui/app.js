const statusDot = document.querySelector("#statusDot");

const statusText = document.querySelector("#statusText");

const pauseButton = document.querySelector("#pauseButton");

const assist = document.querySelector("#assist");

const assistValue = document.querySelector("#assistValue");

let confirmedAssist = null;
let assistEditing = false;
let assistEditSequence = 0;

const bodyCanvasEl = document.querySelector("#bodyCanvas");

const bodyWaiting = document.querySelector("#frameWaiting");



let layout = null;

let skeleton = null;

let paused = false;

let wingStimulus = false;

let pointByIndex = new Map();

let bodyReady = false;

let activity = new Float32Array(0);

const liveTraces = new LiveTraces();

const eyeView = new EyeView();

let neuralGlow = null;

let glowPixels = null;

let glowTexture = null;

const glowingNeurons = new Set();

const GLOW_TEXTURE_WIDTH = 512;

const activeOnlyUniform = { value: 0 };

const activeFrameNeurons = new Set();

let lastActivityFrame = null;

function clearActiveFrame() {

  for (const index of activeFrameNeurons) glowPixels[index * 4 + 3] = 0;

  activeFrameNeurons.clear();

  if (glowTexture) glowTexture.needsUpdate = true;

}

function wireActivityMaterial(material) {

  material.onBeforeCompile = (shader) => {

    shader.uniforms.neuralActivity = { value: glowTexture };

    shader.uniforms.activeOnly = activeOnlyUniform;

    shader.vertexShader = "attribute vec2 neuronUV; uniform sampler2D neuralActivity; varying vec2 neuronSignal;\n" + shader.vertexShader;

    shader.vertexShader = shader.vertexShader.replace("#include <begin_vertex>",

      "#include <begin_vertex>\nneuronSignal = texture2D(neuralActivity, neuronUV).ra;");

    shader.fragmentShader = "uniform float activeOnly; varying vec2 neuronSignal;\n" + shader.fragmentShader;

    shader.fragmentShader = shader.fragmentShader.replace("#include <color_fragment>",

      "#include <color_fragment>\nif (activeOnly > 0.5 && neuronSignal.y < 0.5) discard;\nfloat pulse = max(neuronSignal.x, activeOnly * neuronSignal.y);\ndiffuseColor.rgb = mix(diffuseColor.rgb, vec3(1.0, 0.96, 0.72), pulse);\ndiffuseColor.a = mix(diffuseColor.a, 1.0, pulse);");

  };

  material.customProgramCacheKey = () => "live-activity-v2";

}



function number(value) {

  return new Intl.NumberFormat("en-US").format(value || 0);

}



function fitRenderer(renderer, camera, canvas) {

  const box = canvas.getBoundingClientRect();

  const ratio = Math.min(window.devicePixelRatio || 1, 2);

  const width = Math.max(1, Math.round(box.width));

  const height = Math.max(1, Math.round(box.height));

  renderer.setPixelRatio(ratio);

  renderer.setSize(width, height, false);

  camera.aspect = width / Math.max(1, height);

  camera.updateProjectionMatrix();

}



// ===================== brain: live 3D point cloud =====================

const brainCanvasEl = document.querySelector("#brainCanvas");

const brainRenderer = new THREE.WebGLRenderer({ canvas: brainCanvasEl, antialias: true });

brainRenderer.setClearColor(0x090c10, 1);

const brainScene = new THREE.Scene();

const brainCamera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);

const brainControls = new THREE.OrbitControls(brainCamera, brainRenderer.domElement);

brainControls.enableDamping = true;

brainControls.dampingFactor = 0.08;

brainControls.autoRotate = true;

brainControls.autoRotateSpeed = 0.5;

const BRAIN_IDLE_RESUME_MS = 3000;

let brainIdleTimer = null;

brainControls.addEventListener("start", () => {
  brainControls.autoRotate = false;
  clearTimeout(brainIdleTimer);
});

// Resume the slow spin once nobody has touched the view for a few seconds, except while a single neuron is being inspected.
brainControls.addEventListener("end", () => {
  clearTimeout(brainIdleTimer);
  brainIdleTimer = setTimeout(() => {
    if (!overviewCamera && brainSpinToggle.checked) brainControls.autoRotate = true;
  }, BRAIN_IDLE_RESUME_MS);
});

const brainSpinToggle = document.querySelector("#brainSpin");

brainSpinToggle.addEventListener("change", () => {
  clearTimeout(brainIdleTimer);
  brainControls.autoRotate = brainSpinToggle.checked && !overviewCamera;
});



// MaleCNS assigns every retained neuron a "superclass" label. These prefixes are the

// dataset's own top-level anatomical/functional split (central brain vs optic lobe vs

// ventral nerve cord, then sensory/motor/ascending/descending role within each) - this

// is real annotation data, not an invented subdivision.

const REGION_RULES = [

  ["cb_", "Central brain", 0xe8a23c],

  ["ol_", "Optic lobe", 0x4c8de8],

  ["vnc_", "Ventral nerve cord", 0x4cbf7a],

  ["visual_", "Visual projection", 0x3ce0c8],

  ["sensory_", "Peripheral sensory", 0xe8d23c],

  ["ascending_neuron", "Ascending (VNC to brain)", 0xe85c9e],

  ["descending_neuron", "Descending (brain to VNC)", 0xe85c3c],

  ["efferent_", "Efferent", 0x9c5ce8],

  ["ENS", "Enteric nervous system", 0x8a8f6c],

];

const FALLBACK_REGION = ["Other", 0x888888];



// Named structures MaleCNS labels individually. These are the famous parts of the fly

// brain, and they get priority over the coarse region color so they stand out.

const CLASS_COLORS = {

  Kenyon_Cell: [0xff5ce0, "Mushroom body (Kenyon cells)"],

  CX: [0xffa53c, "Central complex"],

  ALPN: [0x3cffd0, "Antennal lobe projection"],

  ALLN: [0x24c8a4, "Antennal lobe local"],

  DAN: [0xff3c6e, "Dopaminergic"],

  MBON: [0xff8cf0, "Mushroom body output"],

  olfactory: [0x7cff9e, "Olfactory"],

  gustatory: [0xffe45c, "Gustatory"],

  visual: [0x7cf0ff, "Visual"],

  mechanosensory: [0xc08cff, "Mechanosensory"],

  mechanosensory_tactile: [0xa86cff, "Mechanosensory (tactile)"],

  mechanosensory_proprioceptive: [0x8a4cff, "Mechanosensory (proprioceptive)"],

  thermosensory: [0xff9a6c, "Thermosensory"],

  hygrosensory: [0x6cd0ff, "Hygrosensory"],

  unknown_sensory: [0x9aa0a6, "Unclassified sensory"],

};

const SPARE_PALETTE = [0xe07cff, 0x7ce0a0, 0xe0c67c, 0x7c9ae0, 0xe07c9a, 0x9ae07c];

const spareByClass = new Map();



function regionFor(superclass) {

  for (const [prefix, label, color] of REGION_RULES) {

    if (superclass === prefix || superclass.startsWith(prefix)) return [label, color];

  }

  return FALLBACK_REGION;

}



// A named class wins; otherwise fall back to the neuron's coarse anatomical region.

function paletteFor(neuronClass, superclass) {

  if (neuronClass) {

    const known = CLASS_COLORS[neuronClass];

    if (known) return [known[1], known[0]];

    if (!spareByClass.has(neuronClass)) {

      spareByClass.set(neuronClass, SPARE_PALETTE[spareByClass.size % SPARE_PALETTE.length]);

    }

    return [neuronClass.replace(/_/g, " "), spareByClass.get(neuronClass)];

  }

  return regionFor(superclass);

}



function buildLegend(payload) {

  const counts = new Map();

  const classes = payload.neuron_class || [];

  for (let i = 0; i < payload.superclass.length; i += 1) {

    const [label, color] = paletteFor(classes[i] || "", payload.superclass[i]);

    const entry = counts.get(label) || { color, n: 0 };

    entry.n += 1;

    counts.set(label, entry);

  }

  const ordered = Array.from(counts.entries()).sort((a, b) => b[1].n - a[1].n).slice(0, 18);

  const legend = document.querySelector("#brainLegend");

  legend.textContent = "";

  for (const [label, entry] of ordered) {

    const row = document.createElement("div");

    const swatch = document.createElement("i");

    swatch.style.background = `#${entry.color.toString(16).padStart(6, "0")}`;

    row.appendChild(swatch);

    row.appendChild(document.createTextNode(label));

    legend.appendChild(row);

  }

}



let brainColorAttr = null;

let brainBaseColors = null;

let brainBoostColors = null;

let somaPoints = null;



function buildBrainPoints(payload) {

  const total = payload.manifest.neurons;

  neuralGlow = new Float32Array(total);

  glowPixels = new Uint8Array(GLOW_TEXTURE_WIDTH * Math.ceil(total / GLOW_TEXTURE_WIDTH) * 4);

  glowTexture = new THREE.DataTexture(glowPixels, GLOW_TEXTURE_WIDTH,

    Math.ceil(total / GLOW_TEXTURE_WIDTH), THREE.RGBAFormat, THREE.UnsignedByteType);

  glowTexture.minFilter = THREE.NearestFilter;

  glowTexture.magFilter = THREE.NearestFilter;

  glowTexture.needsUpdate = true;

  const n = payload.positions.length;

  const positions = new Float32Array(n * 3);

  brainBaseColors = new Float32Array(n * 3);

  brainBoostColors = new Float32Array(n * 3);

  const colors = new Float32Array(n * 3);

  let maxR = 0.5;

  const tmp = new THREE.Color();

  activity = new Float32Array(n);

  const classes = payload.neuron_class || [];

  for (let i = 0; i < n; i += 1) {

    const p = payload.positions[i];

    positions[i * 3] = p[0];

    positions[i * 3 + 1] = p[1];

    positions[i * 3 + 2] = p[2];

    maxR = Math.max(maxR, Math.hypot(p[0], p[1], p[2]));



    const [, regionColor] = paletteFor(classes[i] || "", payload.superclass[i]);

    const dim = classes[i] ? 0.78 : 0.5;

    tmp.setHex(regionColor);

    brainBaseColors[i * 3] = tmp.r * dim;

    brainBaseColors[i * 3 + 1] = tmp.g * dim;

    brainBaseColors[i * 3 + 2] = tmp.b * dim;

    // Activity glow brightens toward white while keeping the region's hue, so an

    // active neuron still visibly belongs to its color-coded section.

    brainBoostColors[i * 3] = tmp.r + (1 - tmp.r) * 0.75;

    brainBoostColors[i * 3 + 1] = tmp.g + (1 - tmp.g) * 0.75;

    brainBoostColors[i * 3 + 2] = tmp.b + (1 - tmp.b) * 0.75;

  }

  colors.set(brainBaseColors);

  const geometry = new THREE.BufferGeometry();

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  brainColorAttr = geometry.attributes.color;

  const material = new THREE.PointsMaterial({

    size: Math.max(0.006, maxR * 0.0045),

    vertexColors: true,

    sizeAttenuation: true,

    transparent: true,

    opacity: 0.24,

    depthWrite: false,

    depthTest: false,

  });

  const neuronUV = new Float32Array(n * 2);

  payload.indices.forEach((index, i) => {

    neuronUV[i * 2] = (index % GLOW_TEXTURE_WIDTH + 0.5) / GLOW_TEXTURE_WIDTH;

    neuronUV[i * 2 + 1] = (Math.floor(index / GLOW_TEXTURE_WIDTH) + 0.5) / glowTexture.image.height;

  });

  geometry.setAttribute("neuronUV", new THREE.BufferAttribute(neuronUV, 2));

  wireActivityMaterial(material);

  somaPoints = new THREE.Points(geometry, material);

  brainScene.add(somaPoints);

  brainCamera.position.set(maxR * 0.3, maxR * 0.2, maxR * 2.4);

  brainControls.target.set(0, 0, 0);

  brainControls.update();

  buildLegend(payload);

  setBrainView();

}



function decayAndPaintBrainActivity() {

  if (!brainColorAttr) return;

  const arr = brainColorAttr.array;

  let dirty = false;

  for (let i = 0; i < activity.length && i < brainColorAttr.count; i += 1) {

    const glow = activity[i];

    if (glow <= 0.002) continue;

    dirty = true;

    arr[i * 3] = brainBaseColors[i * 3] + glow * (brainBoostColors[i * 3] - brainBaseColors[i * 3]);

    arr[i * 3 + 1] = brainBaseColors[i * 3 + 1] + glow * (brainBoostColors[i * 3 + 1] - brainBaseColors[i * 3 + 1]);

    arr[i * 3 + 2] = brainBaseColors[i * 3 + 2] + glow * (brainBoostColors[i * 3 + 2] - brainBaseColors[i * 3 + 2]);

    activity[i] *= 0.91;

    if (activity[i] <= 0.002) {

      activity[i] = 0;

      arr[i * 3] = brainBaseColors[i * 3];

      arr[i * 3 + 1] = brainBaseColors[i * 3 + 1];

      arr[i * 3 + 2] = brainBaseColors[i * 3 + 2];

    }

  }

  if (dirty) brainColorAttr.needsUpdate = true;

}



function boostBrainActivity(spikes, encoded) {

  clearActiveFrame();

  function boost(index, count) {

    if (neuralGlow && index < neuralGlow.length) {

      neuralGlow[index] = Math.min(1, 0.65 + count * 0.12);

      glowingNeurons.add(index);

      if (count > 0) { glowPixels[index * 4 + 3] = 255; activeFrameNeurons.add(index); }

    }

    const point = pointByIndex.get(index);

    if (point !== undefined) activity[point] = Math.min(1, 0.35 + count * 0.2);

  }

  if (encoded) {

    const raw = atob(encoded);

    const bytes = new Uint8Array(raw.length);

    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    const pairs = new DataView(bytes.buffer);

    for (let i = 0; i + 7 < bytes.length; i += 8) {

      boost(pairs.getUint32(i, true), pairs.getUint32(i + 4, true));

    }

  } else {

    for (const spike of spikes || []) boost(spike.index, spike.spikes);

  }

}



function paintNeuriteActivity() {

  if (!glowTexture || !glowingNeurons.size) return;

  for (const index of glowingNeurons) {

    const value = neuralGlow[index];

    glowPixels[index * 4] = Math.round(255 * value);

    neuralGlow[index] *= 0.965;

    if (neuralGlow[index] < 0.008) {

      neuralGlow[index] = 0;

      glowPixels[index * 4] = 0;

      glowingNeurons.delete(index);

    }

  }

  glowTexture.needsUpdate = true;

}



async function loadLayout() {

  try {

    const response = await fetch("/api/layout", { cache: "no-store" });

    if (!response.ok) throw new Error(`layout ${response.status}`);

    layout = await response.json();

    pointByIndex = new Map(layout.indices.map((index, point) => [index, point]));

    document.querySelector("#datasetStats").textContent = `${number(layout.manifest.neurons)} neurons`;

    buildBrainPoints(layout);

    brainCanvasEl.classList.add("ready");

    loadMorphology();

  } catch (error) {

    setTimeout(loadLayout, 500);

  }

}



// Measured internal axon/dendrite segments, fetched independently of simulation.

const neuriteGroup = new THREE.Group();

const inspectedGroup = new THREE.Group();

brainScene.add(neuriteGroup, inspectedGroup);

let morphologyOffset = 0;

let morphologyMissing = 0;

let morphologyEdges = 0;

let morphologyLoading = false;

let morphologyStop = false;

let overviewCamera = null;

const morphologyStatus = document.querySelector("#morphologyStatus");

const clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 3);

const brainPreferenceIds = ["activeOnly", "showSomata", "showNeurites", "brainXray", "brainCut", "morphologyCount"];

try {

  const preferences = JSON.parse(localStorage.getItem("flybrain-view") || "{}");

  for (const id of brainPreferenceIds) {

    const element = document.getElementById(id);

    if (!(id in preferences)) continue;

    if (element.type === "checkbox") element.checked = Boolean(preferences[id]);

    else if (element.tagName !== "SELECT" || Array.from(element.options).some((option) => option.value === preferences[id])) {

      element.value = preferences[id];

    }

  }

} catch (error) { /* Use defaults if local preferences are unavailable. */ }



function saveBrainPreferences() {

  const preferences = {};

  for (const id of brainPreferenceIds) {

    const element = document.getElementById(id);

    preferences[id] = element.type === "checkbox" ? element.checked : element.value;

  }

  try { localStorage.setItem("flybrain-view", JSON.stringify(preferences)); } catch (error) { /* Optional persistence. */ }

}



function setBrainView() {

  activeOnlyUniform.value = document.querySelector("#activeOnly").checked ? 1 : 0;

  if (somaPoints) somaPoints.visible = document.querySelector("#showSomata").checked;

  neuriteGroup.visible = document.querySelector("#showNeurites").checked && !inspectedGroup.children.length;

  inspectedGroup.visible = document.querySelector("#showNeurites").checked;

  const xray = document.querySelector("#brainXray").checked;

  brainScene.traverse((object) => {

    if (!object.material) return;

    object.material.depthTest = !xray;

    object.material.depthWrite = !xray;

    object.material.opacity = object === somaPoints ? (xray ? 0.24 : 1) : (xray ? 0.22 : 0.65);

  });

  clipPlane.constant = Number(document.querySelector("#brainCut").value);

  brainRenderer.clippingPlanes = clipPlane.constant >= 3 ? [] : [clipPlane];

}

for (const id of ["activeOnly", "showSomata", "showNeurites", "brainXray", "brainCut"]) {

  document.querySelector(`#${id}`).addEventListener("input", () => { setBrainView(); saveBrainPreferences(); });

}



function addNeurites(buffer, group) {

  const rows = new Float32Array(buffer);

  if (rows.length % 7) throw new Error("Invalid morphology response");

  const n = rows.length / 7;

  const positions = new Float32Array(n * 6);

  const colors = new Float32Array(n * 6);

  const neuronUV = new Float32Array(n * 4);

  const color = new THREE.Color();

  for (let i = 0; i < n; i++) {

    positions.set(rows.subarray(i * 7, i * 7 + 6), i * 6);

    const index = rows[i * 7 + 6];

    const u = (index % GLOW_TEXTURE_WIDTH + 0.5) / GLOW_TEXTURE_WIDTH;

    const v = (Math.floor(index / GLOW_TEXTURE_WIDTH) + 0.5) / glowTexture.image.height;

    neuronUV.set([u, v, u, v], i * 4);

    color.setHex(paletteFor(layout.all_neuron_class[index], layout.all_superclass[index])[1]);

    colors.set([color.r, color.g, color.b, color.r, color.g, color.b], i * 6);

  }

  const geometry = new THREE.BufferGeometry();

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  geometry.setAttribute("neuronUV", new THREE.BufferAttribute(neuronUV, 2));

  const material = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,

    opacity: 0.22, depthWrite: false, depthTest: false });

  wireActivityMaterial(material);

  group.add(new THREE.LineSegments(geometry, material));

  setBrainView();

  return n;

}



async function loadMorphology() {

  if (!layout || morphologyLoading) return;

  morphologyLoading = true;

  morphologyStop = false;

  try {

    while (!morphologyStop) {

      const selected = document.querySelector("#morphologyCount").value;

      const target = selected === "all" ? layout.manifest.neurons : Number(selected);

      if (morphologyOffset >= target) break;

      const count = Math.min(32, target - morphologyOffset);

      const detail = selected === "all" ? 64 : 1024;

      morphologyStatus.textContent = `Loading internal branches: ${number(morphologyOffset)} / ${number(target)} neurons`;

      const response = await fetch(`/api/morphology?offset=${morphologyOffset}&count=${count}&detail=${detail}`);

      if (!response.ok) throw new Error(`Server returned ${response.status}`);

      morphologyEdges += addNeurites(await response.arrayBuffer(), neuriteGroup);

      morphologyMissing += Number(response.headers.get("X-Neurons-Missing"));

      morphologyOffset += Number(response.headers.get("X-Neurons-Attempted"));

    }

    if (!inspectedGroup.children.length) {

      morphologyStatus.textContent = `${number(morphologyOffset - morphologyMissing)} neurons, ${number(morphologyEdges)} measured branch segments. ${morphologyStop ? "Loading stopped." : "Overview ready."}` +

        (morphologyMissing ? ` ${number(morphologyMissing)} unavailable; change density to continue.` : "");

    }

  } catch (error) {

    morphologyStatus.textContent = `Branches: ${error.message}. Change density to retry.`;

  } finally {

    morphologyLoading = false;

  }

}

document.querySelector("#morphologyCount").addEventListener("change", () => { saveBrainPreferences(); loadMorphology(); });

document.querySelector("#stopMorphology").addEventListener("click", () => { morphologyStop = true; });

function clearInspected() {

  while (inspectedGroup.children.length) {

    const object = inspectedGroup.children[0];

    inspectedGroup.remove(object); object.geometry.dispose(); object.material.dispose();

  }

  document.querySelector("#clearNeuron").hidden = true;

  if (overviewCamera) {

    brainCamera.position.copy(overviewCamera.position);

    brainControls.target.copy(overviewCamera.target);

    brainControls.update();

    overviewCamera = null;

    brainControls.autoRotate = brainSpinToggle.checked;

  }

  setBrainView();

  morphologyStatus.textContent = `${number(morphologyOffset - morphologyMissing)} overview neurons, ${number(morphologyEdges)} sampled segments.`;

}

document.querySelector("#clearNeuron").addEventListener("click", clearInspected);

document.querySelector("#neuronForm").addEventListener("submit", async (event) => {

  event.preventDefault();

  const id = document.querySelector("#neuronId").value.trim();

  if (!/^\d+$/.test(id)) { morphologyStatus.textContent = "Enter a numeric MaleCNS body ID."; return; }

  morphologyStop = true;

  try {

    const response = await fetch(`/api/morphology?body_id=${encodeURIComponent(id)}`);

    if (!response.ok) throw new Error("Neuron ID is not in this graph or could not be loaded");

    if (Number(response.headers.get("X-Neurons-Missing"))) throw new Error("Skeleton unavailable from Janelia");

    clearInspected();

    const count = addNeurites(await response.arrayBuffer(), inspectedGroup);

    document.querySelector("#clearNeuron").hidden = false;

    overviewCamera = { position: brainCamera.position.clone(), target: brainControls.target.clone() };

    const bounds = new THREE.Box3().setFromObject(inspectedGroup);

    const center = bounds.getCenter(new THREE.Vector3());

    const radius = Math.max(0.1, bounds.getSize(new THREE.Vector3()).length() * 0.5);

    const direction = brainCamera.position.clone().sub(brainControls.target).normalize();

    brainControls.target.copy(center);

    brainCamera.position.copy(center).addScaledVector(direction, radius * 2.8);

    brainControls.autoRotate = false;

    brainControls.update();

    morphologyStatus.textContent = `Neuron ${id}: all ${number(count)} recorded branch segments.`;

  } catch (error) { morphologyStatus.textContent = error.message; }

});

brainCanvasEl.addEventListener("dblclick", (event) => {

  if (!somaPoints || !somaPoints.visible) return;

  const box = brainCanvasEl.getBoundingClientRect();

  const mouse = new THREE.Vector2((event.clientX - box.left) / box.width * 2 - 1,

    -(event.clientY - box.top) / box.height * 2 + 1);

  const raycaster = new THREE.Raycaster();

  raycaster.params.Points.threshold = 0.012;

  raycaster.setFromCamera(mouse, brainCamera);

  const hit = raycaster.intersectObject(somaPoints).find((hit) => !activeOnlyUniform.value || activeFrameNeurons.has(layout.indices[hit.index]));

  if (!hit) return;

  document.querySelector("#neuronId").value = layout.all_ids[layout.indices[hit.index]];

  document.querySelector("#neuronForm").requestSubmit();

});



// ===================== fly body: live posed 3D mesh =====================

const bodyRenderer = new THREE.WebGLRenderer({ canvas: bodyCanvasEl, antialias: true });

bodyRenderer.setClearColor(0x0b0e11, 1);

const bodyScene = new THREE.Scene();

const bodyCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 4000);

bodyCamera.up.set(0, 0, 1);

const bodyControls = new THREE.OrbitControls(bodyCamera, bodyRenderer.domElement);

bodyControls.enableDamping = true;

bodyControls.dampingFactor = 0.1;
// The fly walks tens of millimetres away from its start; without a distance bound it
// ends up a speck in the corner. Zoom stays free within these limits.
bodyControls.minDistance = 1.5;
bodyControls.maxDistance = 14;

bodyControls.autoRotate = true;

bodyControls.autoRotateSpeed = 0.5;

bodyControls.addEventListener("start", () => { bodyControls.autoRotate = false; });



const SKY_COLOR = 0xa9cdea;

bodyScene.background = new THREE.Color(SKY_COLOR);

bodyRenderer.shadowMap.enabled = true;

bodyRenderer.shadowMap.type = THREE.PCFSoftShadowMap;

bodyScene.add(new THREE.HemisphereLight(0xdcecff, 0x3d5a24, 0.95));

const sun = new THREE.DirectionalLight(0xfff3dd, 1.0);

sun.position.set(40, -60, 90);

sun.castShadow = true;

sun.shadow.mapSize.set(2048, 2048);

Object.assign(sun.shadow.camera, { left: -32, right: 32, top: 32, bottom: -32, near: 1, far: 260 });

sun.shadow.bias = -0.0005;

bodyScene.add(sun);

function canvasTexture(size, paint, repeat) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  paint(canvas.getContext("2d"), size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(...repeat);
  texture.anisotropy = 8;
  return texture;
}

function grassTexture() {
  return canvasTexture(256, (ctx, size) => {
    ctx.fillStyle = "#4f7d2c";
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 2600; i++) {
      const x = Math.random() * size, y = Math.random() * size;
      const shade = 35 + Math.random() * 30;
      ctx.strokeStyle = `hsl(${88 + Math.random() * 22}, ${45 + Math.random() * 20}%, ${shade}%)`;
      ctx.lineWidth = 1 + Math.random();
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + (Math.random() - 0.5) * 5, y - 4 - Math.random() * 7);
      ctx.stroke();
    }
  }, [60, 60]);
}

function stoneTexture(repeatX, repeatY) {
  return canvasTexture(256, (ctx, size) => {
    ctx.fillStyle = "#6b6a62";
    ctx.fillRect(0, 0, size, size);
    const rowH = size / 4;
    for (let row = 0; row < 4; row++) {
      const offset = row % 2 ? size / 4 : 0;
      for (let col = -1; col < 3; col++) {
        const x = col * size / 2 + offset, y = row * rowH;
        const tone = 92 + Math.random() * 40;
        ctx.fillStyle = `rgb(${tone}, ${tone - 4}, ${tone - 14})`;
        ctx.fillRect(x + 3, y + 3, size / 2 - 6, rowH - 6);
      }
    }
  }, [repeatX, repeatY]);
}

const WALL_VIEW_OPACITY = 0.18;

const wallMaterials = [];

function standardMaterial(color, options = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.8, ...options });
}

// Display models drawn over the physics shapes: each stays inside the collision volume the simulation uses.
function sceneObject(target) {
  const group = new THREE.Group();
  const [px, py, pz] = target.position;
  group.position.set(px, py, pz);
  const add = (geometry, material, position = [0, 0, 0]) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...position);
    mesh.castShadow = mesh.receiveShadow = true;
    group.add(mesh);
    return mesh;
  };

  if (target.name === "apple") {
    const r = target.size[0];
    const body = add(new THREE.SphereGeometry(r, 32, 24), standardMaterial(0xc4161c, { roughness: 0.45 }));
    body.scale.set(1, 1, 0.92);
    add(new THREE.CylinderGeometry(r * 0.05, r * 0.07, r * 0.4, 8), standardMaterial(0x4a2f17), [0, 0, r * 0.95])
      .rotation.x = Math.PI / 2;
    const leaf = add(new THREE.SphereGeometry(r * 0.3, 12, 8), standardMaterial(0x3f8f2a), [r * 0.22, 0, r * 1.02]);
    leaf.scale.set(1.2, 0.45, 0.12);
    leaf.rotation.y = -0.5;
  } else if (target.name === "banana") {
    const [lx, ly, lz] = target.size;
    const points = [];
    for (let i = 0; i <= 12; i++) {
      const t = i / 12 * 2 - 1;
      points.push(new THREE.Vector3(t * lx * 0.95, (1 - t * t) * ly * 0.9 - ly * 0.45, 0));
    }
    const curve = new THREE.CatmullRomCurve3(points);
    add(new THREE.TubeGeometry(curve, 40, Math.min(ly, lz) * 0.75, 12, false), standardMaterial(0xf2cf2c, { roughness: 0.55 }));
    for (const end of [points[0], points[points.length - 1]]) {
      add(new THREE.SphereGeometry(Math.min(ly, lz) * 0.42, 10, 8), standardMaterial(0x4a3418), [end.x, end.y, end.z]);
    }
  } else if (target.name.endsWith("_trunk")) {
    const [r, halfHeight] = target.size;
    add(new THREE.CylinderGeometry(r * 0.7, r * 1.15, halfHeight * 2, 14), standardMaterial(0x5a3a1c, { roughness: 0.95 }))
      .rotation.x = Math.PI / 2;
  } else if (target.name.endsWith("_crown")) {
    const r = target.size[0];
    const greens = [0x2f7a2a, 0x3b8c31, 0x276b25, 0x4a9a38];
    const blobs = [[0, 0, 0.15, 0.7], [0.4, 0.15, -0.1, 0.55], [-0.38, 0.2, -0.05, 0.55],
      [0.1, -0.42, -0.08, 0.55], [-0.15, -0.3, 0.35, 0.5], [0.25, 0.3, 0.35, 0.48], [0, 0, -0.35, 0.55]];
    blobs.forEach(([x, y, z, s], i) => {
      add(new THREE.IcosahedronGeometry(r * s, 1), standardMaterial(greens[i % greens.length], { flatShading: true }),
        [x * r, y * r, z * r]);
    });
  } else if (target.name === "orange") {
    const r = target.size[0];
    add(new THREE.SphereGeometry(r, 32, 24), standardMaterial(0xf08a12, { roughness: 0.7 }));
    add(new THREE.SphereGeometry(r * 0.12, 8, 6), standardMaterial(0x4f6b1f), [0, 0, r * 0.97]);
  } else if (target.name === "pear") {
    const [rx, , rz] = target.size;
    const profile = [];
    for (let i = 0; i <= 16; i++) {
      const t = i / 16;
      const radius = rx * (0.55 + 0.45 * Math.sin(Math.PI * Math.min(1, t * 1.25))) * Math.sin(Math.PI * t) ** 0.6;
      profile.push(new THREE.Vector2(Math.max(0.001, radius), -rz + t * 2 * rz));
    }
    add(new THREE.LatheGeometry(profile, 28), standardMaterial(0xb8c83a, { roughness: 0.6 })).rotation.x = Math.PI / 2;
    add(new THREE.CylinderGeometry(rx * 0.05, rx * 0.07, rz * 0.35, 8), standardMaterial(0x4a2f17), [0, 0, rz * 1.05])
      .rotation.x = Math.PI / 2;
  } else if (target.name === "mushroom_stem") {
    const [r, halfHeight] = target.size;
    add(new THREE.CylinderGeometry(r * 0.85, r, halfHeight * 2, 16), standardMaterial(0xeee6d2)).rotation.x = Math.PI / 2;
  } else if (target.name === "mushroom_cap") {
    const [rx, ry, rz] = target.size;
    const cap = add(new THREE.SphereGeometry(1, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2), standardMaterial(0xc81d14, { roughness: 0.5 }));
    cap.rotation.x = Math.PI / 2;
    cap.scale.set(rx, rz * 1.6, ry);
    cap.position.z = -rz * 0.6;
    for (let i = 0; i < 9; i++) {
      const a = i * 2.4, d = 0.35 + 0.45 * ((i * 37) % 10) / 10;
      const x = Math.cos(a) * d * rx, y = Math.sin(a) * d * ry;
      const z = -rz * 0.6 + rz * 1.6 * Math.sqrt(Math.max(0, 1 - d * d)) + 0.02;
      add(new THREE.SphereGeometry(rx * 0.09, 8, 6), standardMaterial(0xfaf6ea), [x, y, z]).scale.z = 0.4;
    }
  } else if (target.name.startsWith("rock_")) {
    const [rx, ry, rz] = target.size;
    const rock = add(new THREE.IcosahedronGeometry(1, 1), standardMaterial(0x8a8780, { flatShading: true, roughness: 1 }));
    rock.scale.set(rx, ry, rz);
  } else if (target.name.startsWith("wall_")) {
    const [sx, sy, sz] = target.size;
    const material = standardMaterial(0xffffff, { map: stoneTexture(Math.max(sx, sy) / 2, sz / 2), roughness: 0.9,
      transparent: true, opacity: WALL_VIEW_OPACITY, depthWrite: false });
    wallMaterials.push(material);
    add(new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2), material).castShadow = false;
  } else {
    const size = target.size.length === 1 ? [target.size[0], target.size[0], target.size[0]] : target.size;
    add(new THREE.SphereGeometry(1, 24, 16), standardMaterial(new THREE.Color(...target.color))).scale.set(...size);
  }
  return group;
}



// ===================== fly vision: compound-eye depiction =====================
// Display only: the arena is captured around the head and resampled into hexagonal facets.
// The simulated eye rays that feed the brain are computed on the server and are not used here.

const eyeCanvasEl = document.querySelector("#eyeView");
const eyeRenderer = new THREE.WebGLRenderer({ canvas: eyeCanvasEl, antialias: false });
eyeRenderer.setClearColor(0xa5c8eb, 1);
const eyeCubeTarget = new THREE.WebGLCubeRenderTarget(128, { generateMipmaps: false, minFilter: THREE.LinearFilter });
const eyeCubeCamera = new THREE.CubeCamera(0.02, 200, eyeCubeTarget);
const eyeHeadMatrix = new THREE.Matrix3();
const EYE_FACET_DEGREES = 4.5;
const EYE_FIELD_DEGREES = 350;
const eyeMaterial2d = new THREE.ShaderMaterial({
  uniforms: {
    env: { value: eyeCubeTarget.texture },
    headMatrix: { value: eyeHeadMatrix },
    resolution: { value: new THREE.Vector2(1, 1) },
  },
  vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
  fragmentShader: `
    uniform samplerCube env;
    uniform mat3 headMatrix;
    uniform vec2 resolution;
    const float FACET = ${EYE_FACET_DEGREES.toFixed(2)};
    const float FIELD = ${EYE_FIELD_DEGREES.toFixed(1)};
    const float S3 = 1.7320508;

    vec2 hexCenter(vec2 p, float size) {
      float q = (S3 / 3.0 * p.x - p.y / 3.0) / size;
      float r = (2.0 / 3.0 * p.y) / size;
      vec3 c = vec3(q, -q - r, r);
      vec3 rc = floor(c + 0.5);
      vec3 d = abs(rc - c);
      if (d.x > d.y && d.x > d.z) rc.x = -rc.y - rc.z;
      else if (d.y > d.z) rc.y = -rc.x - rc.z;
      else rc.z = -rc.x - rc.y;
      return vec2(size * (S3 * rc.x + S3 / 2.0 * rc.z), size * 1.5 * rc.z);
    }

    vec2 toAngles(vec2 pixel) {
      return vec2(FIELD / 2.0 - pixel.x / resolution.x * FIELD, -90.0 + pixel.y / resolution.y * 180.0);
    }

    float inEye(vec2 a, float side) {
      vec2 d = (a - vec2(side * 80.0, 0.0)) / vec2(97.0, 84.0);
      return step(dot(d, d), 1.0);
    }

    vec3 look(vec2 a) {
      vec2 r = radians(a);
      vec3 head = vec3(cos(r.y) * cos(r.x), cos(r.y) * sin(r.x), sin(r.y));
      return textureCube(env, headMatrix * head).rgb;
    }

    void main() {
      float facetPx = FACET * resolution.x / FIELD;
      float size = facetPx / S3;
      vec2 center = hexCenter(gl_FragCoord.xy, size);
      vec2 a = toAngles(center);
      float visible = max(inEye(a, 1.0), inEye(a, -1.0));
      if (visible < 0.5) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
      float spread = FACET * 0.35;
      vec3 color = look(a) * 2.0;
      for (int i = 0; i < 6; i++) {
        float t = float(i) * 1.0471976;
        color += look(a + spread * vec2(cos(t), sin(t)));
      }
      color /= 8.0;
      float edge = length(gl_FragCoord.xy - center) / size;
      color *= 1.0 - 0.65 * smoothstep(0.72, 1.0, edge);
      gl_FragColor = vec4(color, 1.0);
    }
  `,
  depthTest: false,
  depthWrite: false,
});
const eyeQuadScene = new THREE.Scene();
eyeQuadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), eyeMaterial2d));
const eyeQuadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
let eyeLastRender = 0;
let eyeHeadIndex = -1;

function renderCompoundEye(timestamp) {
  if (!bodyReady || !skeleton || timestamp - eyeLastRender < 50 || !eyeCanvasEl.offsetParent) return;
  if (eyeHeadIndex < 0) eyeHeadIndex = skeleton.segments.findIndex((segment) => segment.name === "c_head");
  if (eyeHeadIndex < 0) return;
  eyeLastRender = timestamp;
  const head = bodySegmentMeshes[eyeHeadIndex];
  eyeHeadMatrix.setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(head.quaternion));
  eyeCubeCamera.position.copy(head.position);
  // The fly sees solid walls; only the outside 3D view shows them see-through.
  for (const mesh of bodySegmentMeshes) mesh.visible = false;
  for (const material of wallMaterials) Object.assign(material, { opacity: 1, transparent: false, depthWrite: true });
  eyeCubeCamera.update(eyeRenderer, bodyScene);
  for (const material of wallMaterials) Object.assign(material, { opacity: WALL_VIEW_OPACITY, transparent: true, depthWrite: false });
  for (const mesh of bodySegmentMeshes) mesh.visible = true;
  const size = eyeRenderer.getDrawingBufferSize(new THREE.Vector2());
  eyeMaterial2d.uniforms.resolution.value.copy(size);
  eyeRenderer.render(eyeQuadScene, eyeQuadCamera);
  document.querySelector("#eyeViewStatus").hidden = true;
}

const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0xd79a4c, roughness: 0.7, metalness: 0.04 });

const eyeMaterial = new THREE.MeshStandardMaterial({ color: 0xab3620, roughness: 0.25 });

const wingMaterial = new THREE.MeshStandardMaterial({ color: 0xd9e2ea, roughness: 0.15, transparent: true, opacity: 0.32, side: THREE.DoubleSide });

const haltereMaterial = new THREE.MeshStandardMaterial({ color: 0x8f6c3c });



// Right-side segments reuse the left mesh. Mirroring with a negative scale would

// invert the normals and light every right limb inside-out (it rendered as dark

// gunk stuck to the leg), so mirror the geometry properly: flip Y, restore the

// triangle winding, and recompute normals.

function mirroredGeometry(geometry) {

  const g = geometry.clone();

  g.scale(1, -1, 1);

  const pos = g.attributes.position.array;

  for (let i = 0; i + 8 < pos.length; i += 9) {

    for (let k = 0; k < 3; k += 1) {

      const tmp = pos[i + 3 + k];

      pos[i + 3 + k] = pos[i + 6 + k];

      pos[i + 6 + k] = tmp;

    }

  }

  g.attributes.position.needsUpdate = true;

  g.deleteAttribute("normal");

  g.computeVertexNormals();

  return g;

}



function materialForSegment(name) {

  if (name.endsWith("_wing")) return wingMaterial;

  if (name.endsWith("_eye")) return eyeMaterial;

  if (name.endsWith("_haltere")) return haltereMaterial;

  return bodyMaterial;

}



const stlLoader = new THREE.STLLoader();

const meshGeometryCache = new Map();

function loadMeshGeometry(name) {

  if (meshGeometryCache.has(name)) return meshGeometryCache.get(name);

  const promise = new Promise((resolve, reject) => {

    stlLoader.load(`/meshes/${name}.stl`, resolve, undefined, reject);

  });

  meshGeometryCache.set(name, promise);

  return promise;

}



let bodySegmentMeshes = [];

// Poses arrive in discrete simulation frames; meshes ease toward the latest one every

// rendered frame so motion reads as continuous rather than a slideshow.

const targetPositions = [];

const targetQuaternions = [];



async function buildSkeleton(payload) {

  const uniqueNames = Array.from(new Set(payload.segments.map((s) => s.mesh)));

  const geometryByName = {};

  await Promise.all(

    uniqueNames.map(async (name) => {

      geometryByName[name] = await loadMeshGeometry(name);

    })

  );

  const scale = payload.mesh_scale;

  const mirroredByName = {};

  bodySegmentMeshes = payload.segments.map((segment) => {

    let geometry = geometryByName[segment.mesh];

    if (segment.mirror) {

      if (!mirroredByName[segment.mesh]) {

        mirroredByName[segment.mesh] = mirroredGeometry(geometry);

      }

      geometry = mirroredByName[segment.mesh];

    }

    const mesh = new THREE.Mesh(geometry, materialForSegment(segment.name));

    mesh.scale.setScalar(scale);

    mesh.castShadow = true;

    bodyScene.add(mesh);

    targetPositions.push(new THREE.Vector3());

    targetQuaternions.push(new THREE.Quaternion());

    return mesh;

  });



  const ground = new THREE.Mesh(

    new THREE.PlaneGeometry(240, 240),

    new THREE.MeshStandardMaterial({ map: grassTexture(), roughness: 1 })

  );

  ground.receiveShadow = true;

  bodyScene.add(ground);

  for (const target of payload.targets) bodyScene.add(sceneObject(target));



  bodyCamera.position.set(-5, -8, 4);

  bodyControls.target.set(0, 0, 1);

  bodyControls.update();

}



async function loadSkeleton() {

  try {

    const response = await fetch("/api/skeleton", { cache: "no-store" });

    if (!response.ok) throw new Error(`skeleton ${response.status}`);

    skeleton = await response.json();

    eyeView.setLayout(skeleton);

    await buildSkeleton(skeleton);

  } catch (error) {

    setTimeout(loadSkeleton, 500);

  }

}



function applyBodyPose(pose) {

  if (!bodySegmentMeshes.length || !pose) return;

  for (let i = 0; i < bodySegmentMeshes.length; i += 1) {

    const p = pose.positions[i];

    const q = pose.quaternions[i]; // server sends (w, x, y, z)

    targetPositions[i].set(p[0], p[1], p[2]);

    targetQuaternions[i].set(q[1], q[2], q[3], q[0]);

    if (!bodyReady) {

      bodySegmentMeshes[i].position.copy(targetPositions[i]);

      bodySegmentMeshes[i].quaternion.copy(targetQuaternions[i]);

    }

  }

  if (!bodyReady) {

    bodyReady = true;

    bodyCanvasEl.classList.add("ready");

    bodyWaiting.hidden = true;

  }

}



function interpolateBody() {

  if (!bodyReady) return;

  for (let i = 0; i < bodySegmentMeshes.length; i += 1) {

    bodySegmentMeshes[i].position.lerp(targetPositions[i], 0.22);

    if (skeleton.segments[i].name.endsWith("_wing")) {
      // Display the received physical wing pose without averaging away flaps.
      bodySegmentMeshes[i].quaternion.copy(targetQuaternions[i]);
    } else {
      bodySegmentMeshes[i].quaternion.slerp(targetQuaternions[i], 0.22);
    }

  }

  if (skeleton) {

    const thorax = bodySegmentMeshes[skeleton.thorax_index].position;

    const previousTarget = bodyControls.target.clone();

    bodyControls.target.lerp(new THREE.Vector3(thorax.x, thorax.y, thorax.z + 0.3), 0.25);

    bodyCamera.position.add(bodyControls.target.clone().sub(previousTarget));

  }

}



// ===================== control panel =====================

async function postControl(payload) {

  const response = await fetch("/api/control", {

    method: "POST",

    headers: { "Content-Type": "application/json" },

    body: JSON.stringify(payload),

  });

  if (!response.ok) throw new Error(await response.text());

  if (payload.paused !== undefined) {

    paused = payload.paused;

    if (paused) clearActiveFrame();

    pauseButton.textContent = paused ? "Resume" : "Pause";

  }

  return response.json();

}



function updateState(next) {

  if (Number.isFinite(next.control?.assist)) {
    confirmedAssist = next.control.assist;
    assist.disabled = false;
    if (!assistEditing) {
      assist.value = confirmedAssist;
      assistValue.textContent = confirmedAssist.toFixed(2);
    }
  }

  const mode = next.status || "loading";

  statusText.textContent = mode.toUpperCase();

  statusDot.className = mode;

  paused = mode === "paused";

  pauseButton.textContent = paused ? "Resume" : "Pause";

  pauseButton.classList.toggle("active", paused);



  if (mode !== "running") clearActiveFrame();

  if (!next.brain) return;

  liveTraces.append(next);

  document.querySelector("#activeNeurons").textContent = number(next.brain.active_neurons);

  document.querySelector("#frameSpikes").textContent = number(next.brain.frame_spikes);

  document.querySelector("#brainTime").textContent = `${number(next.brain.simulated_ms)} ms`;

  document.querySelector("#bodyTime").textContent = `${next.body.simulated_seconds.toFixed(2)} s`;

  const d = next.body.displacement;

  document.querySelector("#distance").textContent = `${Math.hypot(...d).toFixed(3)} mm`;

  document.querySelector("#loopRate").textContent = `${next.loop.wall_ms ? (1000 / next.loop.wall_ms).toFixed(1) : "0.0"} Hz`;

  document.querySelector("#brainBackend").textContent = `${(next.brain.backend || "CPU").split(" / ")[0]} · ${next.brain.step_wall_ms.toFixed(1)} ms`;

  document.querySelector("#brainBackend").title = next.brain.backend_error || next.brain.backend || "";

  document.querySelector("#simulationSpeed").textContent = `${(next.loop.realtime_factor || 0).toFixed(2)}× real time`;

  document.querySelector("#datasetStats").textContent = `${number(next.neurons)} neurons`;

  document.querySelector("#contactStats").textContent = `${number(next.edges)} edges · ${number(next.synaptic_contacts)} contacts`;

  document.querySelector("#disclaimer").textContent = next.disclaimer;

  const vision = next.vision || {};
  eyeView.update(vision);
  const seen = Object.entries(vision.visible_objects || {})
    .filter(([name]) => !name.startsWith("wall_"))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([name, count]) => `${name.replace(/_/g, " ")} ${count}`)
    .join(", ");
  document.querySelector("#visionReadout").textContent =
    `Eye luminance L ${(vision.left_mean || 0).toFixed(2)} R ${(vision.right_mean || 0).toFixed(2)}` +
    ` | Visible samples: ${seen || "none"}`;



  wingStimulus = Boolean(next.control?.wing_stimulus);

  const wingButton = document.querySelector("#wingButton");

  wingButton.textContent = wingStimulus ? "Stop wing stimulus" : "Flap wings";

  wingButton.setAttribute("aria-pressed", String(wingStimulus));

  wingButton.classList.toggle("active", wingStimulus);
  const wings = next.body.wings;
  if (wings) {
    const angles = wings.joint_angles || [0, 0, 0, 0];
    document.querySelector("#wingReadout").textContent =
      `DNg02 L ${wings.left_rate.toFixed(1)} / R ${wings.right_rate.toFixed(1)} Hz | ` +
      `Wing drive ${wings.frequency.toFixed(0)} Hz | ` +
      `Stroke L ${(angles[1] * 180 / Math.PI).toFixed(0)} / R ${(angles[3] * 180 / Math.PI).toFixed(0)} deg`;
  }


  const stim = next.control && (next.control.queued_stimulus || next.control.stimulus);

  const m = next.motor || {};

  const held = stim ? `${stim.toUpperCase()} (${Math.round(next.control.stimulus_ms_left)} ms left)` : "none";

  document.querySelector("#motorReadout").textContent =

    `Assist ${Number(next.control?.assist || 0).toFixed(2)} · Manual drive: ${held}${paused && stim ? " (queued while paused)" : ""} · fwd ${(m.forward_rate || 0).toFixed(1)} Hz · ` +

    `back ${(m.backward_rate || 0).toFixed(1)} Hz · turn ${(m.turn_rate || 0).toFixed(1)} · ` +

    `${paused ? "last applied legs" : "legs"} L ${(m.left || 0).toFixed(2)} R ${(m.right || 0).toFixed(2)}`;

  document.querySelectorAll("[data-stimulus]").forEach((button) => {

    button.classList.toggle("active", button.dataset.stimulus === stim);

  });

  if (next.error) statusText.textContent = `ERROR: ${next.error.split("\n").filter(Boolean).pop()}`;



  const activityFrame = `${next.loop.iteration}:${next.brain.simulated_ms}`;

  if (mode === "running" && activityFrame !== lastActivityFrame && neuralGlow) {

    boostBrainActivity(next.brain.activity, next.brain.activity_binary);

  }

  lastActivityFrame = activityFrame;

  applyBodyPose(next.body.pose);

}



function connectStream() {

  let socket;

  try {

    socket = new WebSocket(`ws://${location.host}/ws/state`);

  } catch (error) {

    setTimeout(connectStream, 800);

    return;

  }

  socket.onmessage = (event) => updateState(JSON.parse(event.data));

  socket.onerror = () => socket.close();

  socket.onclose = () => {

    assist.disabled = true;
    assistValue.textContent = "Reconnecting";

    clearActiveFrame();

    statusText.textContent = "RECONNECTING";

    statusDot.className = "error";

    setTimeout(connectStream, 800);

  };

}



document.querySelector("#wingButton").addEventListener("click", async () => {

  try { await postControl({ wing_stimulus: !wingStimulus }); }

  catch (error) { document.querySelector("#motorReadout").textContent = error.message; }

});

pauseButton.addEventListener("click", async () => postControl({ paused: !paused }));

document.querySelector("#resetButton").addEventListener("click", async () => postControl({ reset: true }));

document.querySelectorAll("[data-stimulus]").forEach((button) => {

  button.addEventListener("click", async () => {

    try {

      await postControl({ stimulus: button.dataset.stimulus });

    } catch (error) { document.querySelector("#motorReadout").textContent = error.message; }

  });

});

const RANDOM_STIMULI = ["forward", "left", "right", "loom", "backward", "wings"];
const RANDOM_WING_MS = 1500;
const randomButton = document.querySelector("#randomButton");
const randomRate = document.querySelector("#randomRate");
let randomTimer = null;

function scheduleRandomEvent() {
  const perMinute = Math.min(120, Math.max(1, Number(randomRate.value) || 20));
  const meanMs = 60000 / perMinute;
  randomTimer = setTimeout(async () => {
    if (!paused) {
      const stimulus = RANDOM_STIMULI[Math.floor(Math.random() * RANDOM_STIMULI.length)];
      try {
        if (stimulus !== "wings") await postControl({ stimulus });
        else if (!wingStimulus) {
          await postControl({ wing_stimulus: true });
          setTimeout(() => postControl({ wing_stimulus: false }).catch(() => {}), RANDOM_WING_MS);
        }
      } catch (error) { document.querySelector("#motorReadout").textContent = error.message; }
    }
    if (randomTimer !== null) scheduleRandomEvent();
  }, meanMs * (0.4 + Math.random() * 1.2));
}

randomButton.addEventListener("click", () => {
  const on = randomTimer === null;
  if (on) scheduleRandomEvent();
  else { clearTimeout(randomTimer); randomTimer = null; }
  randomButton.classList.toggle("active", on);
  randomButton.setAttribute("aria-pressed", String(on));
});

randomRate.addEventListener("change", () => {
  if (randomTimer === null) return;
  clearTimeout(randomTimer);
  scheduleRandomEvent();
});

assist.addEventListener("input", () => {

  assistEditing = true;
  assistEditSequence += 1;

  assistValue.textContent = Number(assist.value).toFixed(2);

});

assist.addEventListener("change", async () => {
  const sequence = assistEditSequence;
  try {
    const control = await postControl({ assist: Number(assist.value) });
    if (sequence === assistEditSequence) confirmedAssist = control.assist;
  } catch (error) {
    document.querySelector("#motorReadout").textContent = error.message;
  } finally {
    if (sequence === assistEditSequence) {
      assistEditing = false;
      if (Number.isFinite(confirmedAssist)) {
        assist.value = confirmedAssist;
        assistValue.textContent = confirmedAssist.toFixed(2);
      }
    }
  }
});



function resizeAll() {

  fitRenderer(brainRenderer, brainCamera, brainCanvasEl);

  fitRenderer(bodyRenderer, bodyCamera, bodyCanvasEl);

  fitRenderer(eyeRenderer, eyeQuadCamera, eyeCanvasEl);

  eyeLastRender = 0;

}

window.addEventListener("resize", resizeAll);

new ResizeObserver(resizeAll).observe(document.querySelector("main"));

resizeAll();



function animate(timestamp) {

  requestAnimationFrame(animate);

  decayAndPaintBrainActivity();

  paintNeuriteActivity();

  liveTraces.draw(timestamp);

  interpolateBody();

  brainControls.update();

  bodyControls.update();

  brainRenderer.render(brainScene, brainCamera);

  bodyRenderer.render(bodyScene, bodyCamera);

  renderCompoundEye(timestamp);

}

requestAnimationFrame(animate);



loadLayout();

loadSkeleton();

connectStream();



let focusView = null;

let ownsFullscreen = false;

function setFocusView(view) {

  focusView = view;

  document.body.classList.toggle("focus-brain", view === "brain");

  document.body.classList.toggle("focus-physics", view === "physics");

  document.body.classList.toggle("focus-vision", view === "vision");

  document.body.classList.toggle("focus-bodybrain", view === "bodybrain");

  document.querySelectorAll("[data-fullscreen]").forEach((button) => {

    const focused = button.dataset.fullscreen === view;

    const label = focused ? (button.dataset.exitLabel || "Exit full screen") : (button.dataset.label || "Full screen");

    button.textContent = label;

    button.setAttribute("aria-pressed", String(focused));

    button.setAttribute("aria-label", `${label} ${button.dataset.fullscreen} view`);

  });

  requestAnimationFrame(resizeAll);

}

async function exitFocusView() {

  setFocusView(null);

  if (ownsFullscreen && document.fullscreenElement) await document.exitFullscreen();

  ownsFullscreen = false;

}

document.querySelectorAll("[data-fullscreen]").forEach((button) => {

  button.addEventListener("click", async () => {

    if (focusView === button.dataset.fullscreen) return exitFocusView();

    setFocusView(button.dataset.fullscreen);

    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {

      try { await document.documentElement.requestFullscreen(); ownsFullscreen = true; }

      catch (error) { /* Keep the full-window view if browser fullscreen is unavailable. */ }

    }

  });

});

document.addEventListener("fullscreenchange", () => {

  if (!document.fullscreenElement && ownsFullscreen) { ownsFullscreen = false; setFocusView(null); }

});

document.addEventListener("keydown", (event) => {

  if (event.key === "Escape" && focusView) exitFocusView();

});

