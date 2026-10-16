# Why visual input does not sustain walking

The visual input propagates through the graph, but the eight neurons used by the
walking decoder stop spiking. DNp09 (forward) and DNa02 (turning) are strongly
hyperpolarized by their incoming inhibitory connections. Meanwhile, many other
descending and VNC motor neurons are active but have no direct body actuator
mapping in this implementation.

This is a measured limitation of the current simulation. It does not establish
that an actual fly would remain still in this scene.

## Reproduction

Run `scripts/diagnose_visual_pathway.py` in the project Python environment. Each
case resets an independent brain instance and advances 1,200 ms with no manual
stimulus, zero walking assist, and no body movement commands. The late analysis
window is 600 to 1,200 ms. Frames come from the same eye ray sampler as the app.
The changing-view control cycles through twelve yaw orientations every 24 ms.

The diagnostic saves full traces, voltages, population counts, and presynaptic
contributions to the configured results directory:

- `verification/visual-pathway-diagnosis.json`
- `verification/visual-pathway-diagnosis-cpu.json` for the CPU confirmation

Use `--backend cpu --cases scene readout_inhibition_removed` for the CPU check.
The ablations affect only the diagnostic instance in memory. They do not edit
the prepared graph, application dynamics, scene, or motor interface.

## Direct cause of the zero walking command

The decoder uses two DNp09 cells, two DNa02 cells, and four MDN cells. It also
requests an `oDN1` population, but there are no cells with that label in this
prepared MaleCNS graph, so that term silently contributes zero.

In the normal scene, all eight actual walking readout neurons emit zero spikes
in the late window. The two DNp09 and two DNa02 neurons sit at or near the
model's -75 mV inhibitory floor; the firing threshold is -45 mV. MDN neurons
are also below threshold. This is upstream of the leg controller, so increasing
the display rate or changing the wall geometry cannot restore walking output.

For the two forward neurons, incoming signed synaptic weight multiplied by the
presynaptic spike rate shows inhibitory input approximately five to eight times
larger than excitatory input. This metric describes offered input, not exact
integrated current: the engine also applies delay, decay, reset, and refractory
filtering. The actual membrane and synaptic state measurements independently
confirm hyperpolarization.

The strongest forward inhibitors include CL366 cells 10323 and 10184, followed
by LAL206 cells. Strong turning inhibitors include PS059 cells 11495, 11142,
514988, and 519399, with MBON31 and MBON32 also contributing. The checked CL366,
PS059, MBON31, and MBON32 examples are classified as GABA in the downloaded
neurotransmitter table, consistent with their negative prepared weights.

## Causal controls

The diagnostic compares the ordinary scene, complete darkness, changing views,
disconnected retinal outputs, and removed inhibition into the selected readouts.

- Changing views changes sensory and network activity but does not sustain
  spikes in the walking readouts.
- Darkness also leaves those readouts silent after the transient.
- Disconnecting outgoing retinal edges changes lamina activity, confirming that
  retinal input is actually reaching the network. Walking readouts remain silent.
- Removing only the inhibitory edges entering the selected readout populations
  restores late spikes in DNp09, DNa02, and MDN without direct stimulation of
  those cells. This demonstrates that their silence depends on inhibition in
  this model. It is a diagnostic ablation, not a proposed biological correction.

Measured spike totals in the final 600 ms of the September 16 diagnostic:

| Condition | Backend | Forward DNp09 | Turning DNa02 | Backward MDN |
| --- | --- | ---: | ---: | ---: |
| Ordinary scene | CUDA | 0 | 0 | 0 |
| Darkness | CUDA | 0 | 0 | 0 |
| Changing views | CUDA | 0 | 0 | 0 |
| Retinal outputs disconnected | CUDA | 0 | 0 | 0 |
| Inhibition into readouts removed | CUDA | 36 | 184 | 239 |
| Lamina background removed | CUDA | 0 | 0 | 0 |
| Ordinary scene | CPU | 0 | 0 | 0 |
| Inhibition into readouts removed | CPU | 39 | 185 | 234 |

The CPU confirmation reproduces the suppression and its release. Exact spike
totals differ between backends and some repeated runs; the causal conclusion
holds in both implementations.

## Activity hidden by the body interface

The complete neural graph is stepped, but the walking controller consumes only
the selected eight-neuron readout. Around 200 other descending neurons and more
than 150 VNC motor neurons spike in the normal scene's late window. Their activity
does not directly actuate the legs. The wing actuator uses a separate engineered
DNg02 readout rather than directly reading the active flight motor neurons.

Consequently, a stationary displayed body is not a reliable indicator that the
rest of the neural graph is inactive. This finding does not justify assigning
arbitrary body actions to those active neurons.

## A pre-existing background input matters

`MaleCNSBrain.step` supplies a constant modeled drive of 12 to all 8,884 L1-L5
lamina neurons, in addition to the eye-derived photoreceptor input. That current
is present even in complete darkness. It is an artificial background excitability
assumption, not an environmental measurement.

When that background current is removed in the diagnostic, the retina still
spikes, but no other neurons spike in the late window. Every retained outgoing
R1-R6 edge has a negative weight. With the remaining cells starting below
threshold and no background drive, this spiking model cannot start an excitatory
cascade from those inhibitory retinal outputs alone. With the background current
enabled, activity propagates broadly and the selected walking cells become
inhibited.

The current photoreceptor/lamina model and its background state therefore require
validation as a coupled system. Simply increasing eye brightness, deleting
inhibitory wiring, or adding walking bias would alter the experiment rather than
establish a correct sensory response.

## Scope of the conclusion

The tests locate the immediate failure to produce walking and identify its
numerical cause. They do not determine what physiological values the unmeasured
background currents, synaptic strengths, or cell-specific dynamics should have.
Those require independent evidence, not tuning until the fly moves.

The published model cited by this project's parameterization explicitly notes
limitations from treating neurons identically, omitting receptor dynamics and
non-spiking neurons, and assuming zero basal firing. Its demonstrated feeding
and grooming results do not validate this project's photoreceptor-to-walking
model. Source: [Shiu et al., 2024](https://doi.org/10.1038/s41586-024-07763-9).
