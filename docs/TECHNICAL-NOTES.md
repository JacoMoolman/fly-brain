# Technical notes

Design notes and model assumptions for the MaleCNS and FlyGym closed-loop simulation. See the [README](../README.md) for setup.

## Layout

- FlyGym 2.1.0 source: `vendor/flygym` (you clone it, it is not in this repo).
- MaleCNS v1.0 tables, the prepared graph, the Python environment and caches live under the asset root set in `scripts/Set-FlyBrainEnv.ps1`.

The runtime retains all 166,700 annotated MaleCNS neurons and 25,582,938 directed edges. Rays from the fly's two eye positions sample the actual MuJoCo scene into 3,377 mapped R1-R6 neurons. Unknown receptor hemispheres are inferred from their strongest mapped lamina partners. DNp09, oDN1, MDN, and bilateral DNa02 activity is decoded into FlyGym's existing walking controller. The eye optics, brightness to current conversion, neural dynamics, and motor readout are explicit model assumptions, not measured physiology.

The arena has physical walls (12 mm high, drawn see-through in the 3D view), an apple, a banana, an orange, a pear, a mushroom, two rocks and four trees. The same shapes are the collision geometry and what the eye rays hit. The browser shows the same scene geometry and reports eye luminance and object sample counts. The default walking assist is zero. Vision enters the existing photoreceptor population only; there are no visual steering rules or added connections to motor neurons. The control buttons remain available for manual stimulation. The `Controller assist` slider can still be changed explicitly.

Run `scripts/verify_vision.py --iterations 300` in the project Python environment for a logged unassisted observation. It saves `vision-observation.json` under the configured results directory. In the initial 1.8 seconds of simulated time, the retina and lamina responded to a changed view, and descending turn activity briefly appeared. The fly settled and did not sustain walking. Early body displacement includes physics settling and is not evidence of visually guided movement. This is an observation of the current model, not a guarantee of autonomous behavior.

The subsequent [visual pathway diagnosis](../VISUAL-PATHWAY-DIAGNOSIS.md) traces this
silence to inhibition of the selected walking readout, documents active neural
outputs that the body does not read, and tests the effect of the existing lamina
background current. It includes isolated causal controls and CPU confirmation.

## Official FlyGym

Run the body check:

```powershell
.\scripts\Verify-FlyGym.ps1
```

Open FlyGym's own native viewer separately with:

```powershell
.\scripts\Start-FlyGymViewer.ps1
```

Or double-click `START-FLYGYM.cmd`. This launcher opens only the official FlyGym/MuJoCo 3D arena. It does not start the MaleCNS viewer.

## Integrated verification

```powershell
.\scripts\Verify-Integrated.ps1
```

The body now preserves static MuJoCo segments and validates their qualified body
names. This keeps the head at the neck instead of accidentally using the hind-foot
transform when a missing body ID resolves to -1.

Movement buttons provide a 300 ms interval of neural stimulation, in simulation
time, to the corresponding descending population. The existing motor readout then
drives the walking controller. Release control removes the stimulation. Escape
walk is fast walking, not a simulated jump or flight.
The live footer identifies queued/active commands and left/right leg drive. Reset
also works while paused and clears the active command. Run `scripts/verify_walking.py`
in the project Python environment to check actual translation and turn directions.

The previous brain display contained only measured cell-body positions. Internal
axon/dendrite segments now come from the public
[MaleCNS skeleton dataset](https://male-cns.janelia.org/download/) (Janelia, CC-BY),
cached under the dataset's `skeletons` directory in the asset root. The source's nanometer
coordinates are converted to 8 nm annotation units and normalized identically to
the soma coordinates. No synthetic interior points or soma-to-soma chords are drawn.

The default overview progressively loads 512 neurons, up to 1,024 recorded segments
per neuron, balanced across annotation groups. Higher density options include all
166,700 retained neurons with 64 segments per newly loaded neuron to bound graphics
memory. Loading all requires a large download and takes time. The overview is a
sample of branching geometry, not every branch or every synapse. Full neuron
inspection accepts a MaleCNS body ID and displays every edge in its published
skeleton. Missing skeletons are counted explicitly. All retained neurons and graph
edges remain in the simulation regardless of display density. Cell-body visibility,
X-ray transparency and a movable cutaway expose internal geometry.

The neural backend now defaults to CUDA through NVIDIA Warp when available. All
166,700 neurons, all 25,582,938 edges, and the delayed spike queue stay on the GPU.
The neural integration step remains 0.1 ms. CUDA graphs advance a 6 ms closed-loop
slice and return spike counts for motor decoding and visualization. The panel
header identifies the backend and its measured step time; a CPU fallback reports
its initialization error in the header tooltip. Set `FLYBRAIN_BRAIN_BACKEND=cuda`
to require CUDA, or `cpu` to run the reference implementation explicitly.

Body physics uses a validated 0.5 ms step with the walking controller updated every
1 ms. Cached anatomy mappings and spline/gain inputs preserve the upstream
controller rules without rebuilding them on every step. The previous 0.1 ms physics
step is available via `FLYBRAIN_PHYSICS_DT=0.0001`. A 1 ms physics step failed the
walking checks and is not the default. The speed readout reports actual simulation
time per wall second; the viewer does not claim real-time operation.

Live activity is sent as compact little-endian uint32 `(neuron index, spike count)`
pairs encoded in base64. Every spiking neuron is included. Both cell bodies and
loaded internal branches brighten from measured simulation spikes, with a short
visual decay. Branch colors sample a shared GPU activity texture, avoiding a CPU
rewrite of every branch vertex on every frame. Double-click a cell body to inspect
its full published skeleton.

Additional regression checks are `scripts/verify_gpu.py` (CPU/CUDA spike, delay,
activation and reset parity), `scripts/verify_controller.py` (upstream observation
and controller parity), and `scripts/verify_morphology.py` (published skeleton
coordinate and edge-count fidelity, against the running server).

The live graphs below the 3D views show a combined brain activity trace (mean firing rate in Hz per neuron, and the percentage of neurons firing in the current slice, on separate axes) and left/right walking drive.
Choose a 30 second, 1, 2 or 5 minute (default) wall-clock history and hover for sampled values.
Firing rates use simulated time, so they are not distorted by changes in computer
speed. Pausing freezes the history, reset clears it, and gaps are not filled with
invented samples. Hide graphs restores the larger 3D view. Canvas plotting runs at
most 10 times per second to keep its cost small.

Each 3D view has a Full screen button. Escape or Exit full screen restores the
split layout; simulation controls remain available below the selected view.
Active pathways only hides neurons that did not spike in the latest simulation
slice, including their loaded branches. Pause and disconnection clear this view.
This shows activity per neuron, not propagation timing within individual branches.
The checkbox is saved with the other brain view preferences.

The Flap wings button injects current into the 29 annotated DNg02 neurons
(DNg02_a through DNg02_g) in the retained MaleCNS graph. Their simulated spike
rates, averaged separately by soma side and filtered over 100 ms, drive an
engineered wing oscillator. Each physical wing has spread and stroke joints with
MuJoCo actuators. All displayed wing poses come from MuJoCo, not browser animation.
The readout displays the actual DNg02 rates, oscillator frequency and joint angles.
Stop wing stimulus removes the injected current; any remaining activity decays
through the neural model. Pause freezes physics; reset clears stimulation and rates.

The transfer gains, oscillator and actuator parameters are engineered. This is a
direct descending-neuron-to-actuator bridge, not a reconstruction of the intervening
VNC circuitry or asynchronous flight muscles. DLM/DVM motor neurons are present in
the graph but are not the actuator readout in this implementation. No aerodynamic
lift is modeled. While the wings beat strongly the body is simply placed 3 mm higher
and held level, and when they stop it drops under gravity; this is a display
convenience, not flight physics, and is capped far below the wall height.
Wing simulation uses 0.1 ms physics steps while active, so it costs more than walking.
The browser samples physical poses and cannot display every 180-220 Hz wingbeat.

DNg02's role in wing amplitude regulation is described in:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9206711/

Run scripts/verify_neural_wings.py to check real graph stimulation, physical joint
motion, neural silencing, left/right specificity and reset (it predates the 3 mm hover placement
and may need updating).
