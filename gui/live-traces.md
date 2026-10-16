# Live trace contract

These are live instruments inside the existing simulation UI, not a separate
analytical report. Source: the same `/ws/state` snapshots used by the 3D views.
Question: how do population activity and walking commands change as the fly runs?
No causal or biological interpretation is implied by these modeled signals.

| Chart | Definition | Units / scale | Encoding |
| --- | --- | --- | --- |
| Mean firing rate | Change in total spikes / change in simulated seconds / retained neurons | Hz per neuron, zero-based automatic upper bound | Blue solid line |
| Neurons firing | Unique indices in the current binary activity packet / retained neurons | Percent firing in the latest neural slice, zero-based automatic upper bound | Blue solid line |
| Walking drive | Current motor.left and motor.right | Controller input, fixed -1.4 to +1.4 with zero reference | Left blue solid, right gold dashed, direct legend |

All three use a selectable 30, 60 or 120 second wall-clock window and at least two
observations before drawing a line. Rates use elapsed simulated time, not wall
time, and cumulative spike counts handle skipped WebSocket frames. Duplicate
simulation timestamps are ignored; a reset clears the history. A wall-clock gap
over one second breaks the line. Pause retains the most recent window. Hover
shows exact sampled values and simulated time. No smoothing or invented samples.

Canvas drawing is capped at 10 Hz and history at 8,000 samples, independent of
the 3D animation rate. The containing app is dark, so charts use its dark theme,
quiet grid lines and existing font. Three equally sized panels sit below the
3D views on a desktop and stack at narrow widths. QA checks include formulas,
reset/pause/gap behavior, bounded history, labels at desktop/mobile widths,
browser errors, and live rendering in the user's existing tab.
