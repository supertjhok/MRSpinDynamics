# Bipolar 13-Interval PGSTE

`spin_dynamics.workflows.bipolar` implements the Cotts 13-interval alternating
pulsed-gradient stimulated-echo (APGSTE) sequence -- the sequence in the Bruker
`diff_stebp.gp` program -- for diffusion measurements that suppress a constant
background gradient. The background gradient is often the internal gradient from
magnetic-susceptibility contrast in porous media (see
[Internal / Susceptibility Gradients](internal_gradients.md)); without
suppression it biases the apparent diffusion coefficient through the cross-term
between the applied and background gradients.

## Sequence and phase cycle

The sequence is a stimulated echo (three 90 degree pulses) with one 180 degree
refocusing pulse in each of the two encoding periods, a gradient lobe on each
side of every 180, and the applied gradient polarity inverted between the two
encoding periods. The two 180 pulses refocus the continuously present background
gradient within each period; the polarity alternation cancels the
applied-times-background cross-term.

The phase cycle is the 16-step Bruker `diff_stebp` table, built on the package's
`PhaseCycle` machinery:

```python
from spin_dynamics.phase_cycling import diff_stebp_phase_cycle

cycle = diff_stebp_phase_cycle()   # 16 steps; pulses excitation_90, refocus_180, store_90, read_90
```

The receiver follows `ph31 = -(ph1 + ph2 + ph3) mod 4` (with `ph4 = 0` on both
180 pulses), which selects the stimulated-echo coherence-transfer pathway
`Delta p = (+1, -2, +1, +1, -2)` with detection at `p = -1`, and rejects the
anti-echo, axial (FID), and double-quantum-like pathways. The coherence order
`p(t)` selected by this cycle is exactly the `sign` carried by the toggling-frame
intervals returned by `cotts_thirteen_interval_intervals`.

## Cross-term suppression

The toggling-frame moments make the suppression explicit. The diffusion
attenuation in a constant background gradient `g0` is

```text
ln(E) = -D * (b_applied + g0 * cross_coefficient + g0**2 * background_coefficient)
```

`toggling_frame_moments` returns all three coefficients. For the 13-interval
sequence the `cross_coefficient` is zero; for an ordinary monopolar stimulated
echo it is not, and the apparent diffusion coefficient then drifts with `g0`.

```python
from spin_dynamics.workflows import (
    run_cotts_thirteen_interval_moment,
    run_monopolar_pgste_moment,
)

c = run_cotts_thirteen_interval_moment(gradient_amplitude=0.1, background_gradient=0.05)
m = run_monopolar_pgste_moment(gradient_amplitude=0.1, background_gradient=0.05)
print(c.cross_term_bias)  # ~0: the slope bias the background imposes
print(m.cross_term_bias)  # non-zero: monopolar is biased
print(c.phase_cycle.name)  # "diff_stebp"
```

The apparent diffusion coefficient is recovered from the slope of `ln(E)` versus
the applied b-value over a gradient sweep. Over that sweep the background
*self*-term (`g0**2 * background_coefficient`) is a gradient-independent offset
that does not bias the slope, so only the cross-term matters;
`cross_term_bias = g0 * cross_coefficient / b_applied` reports it per run.

## Scope and limits

- The moment model is the free-diffusion (Gaussian-propagator) b-value for ideal
  rectangular lobes. Because the applied wavevector refocuses within each
  encoding period, the free-diffusion b-value comes from the bipolar lobe pairs;
  the storage interval sets the diffusion *time* that matters for restricted
  media, which needs the walker pipeline.
- Trapezoidal ramps (the Bruker `diff_ramp`) and finite-pulse effects are not in
  the moment model; they shift the b-value slightly but not the cross-term
  cancellation, which is a timing-symmetry property.
- See `examples/plot_bipolar_pgste.py` for the wavevector trajectories and the
  apparent-diffusion suppression curve versus background gradient.
