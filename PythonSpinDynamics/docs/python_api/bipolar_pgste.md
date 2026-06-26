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

## Explicit random-walker runner

`run_cotts_thirteen_interval_walkers` runs the real five-pulse sequence -- the
three 90 degree pulses, the two 180 refocusing pulses, and the four gradient
lobes -- with moving walkers, so it captures restricted-geometry and finite-pulse
effects the moment model omits. A constant `background_gradient` (T/m) is applied
as a linear off-resonance map, so the suppression appears directly in the
simulated signal.

```python
import numpy as np
from spin_dynamics.workflows import run_cotts_thirteen_interval_walkers

result = run_cotts_thirteen_interval_walkers(
    gradient_amplitude=0.1,
    diffusion_coefficient=2.3e-9,
    background_gradient=0.04,     # T/m along the gradient axis
    walkers_per_cell=4000,
    seed=7,
)
echo = np.abs(result.signal[-1])
print(result.b_value, result.phase_cycle.name)  # moment b-value, "diff_stebp"
```

For free diffusion the walker echo reproduces `exp(-b D)` with the moment-model
`b_value` (validated to ~1% in the test suite), and the apparent diffusion
coefficient from a b-value sweep is unbiased by the background gradient. Pass a
restricted-geometry `boundary` (e.g. `spin_dynamics.motion.make_circular_reflector`)
and supply `fields`, `rho`, and `x_axis`/`z_axis` for pore-scale studies; the
storage interval then sets the diffusion time that matters for restricted media.
Unlike a monopolar stimulated echo, the bipolar pair refocuses the encoding phase
before storage, so a fully refocused component is stored rather than half.

## Scope and limits

- The moment model is the free-diffusion (Gaussian-propagator) b-value for ideal
  rectangular lobes. Because the applied wavevector refocuses within each
  encoding period, the free-diffusion b-value comes from the bipolar lobe pairs;
  the storage interval sets the diffusion *time* that matters for restricted
  media, which the walker runner resolves.
- Trapezoidal ramps (the Bruker `diff_ramp`) are not modeled; the walker runner
  uses rectangular lobes. Ramps shift the b-value slightly but not the cross-term
  cancellation, which is a timing-symmetry property.
- See `examples/plot_bipolar_pgste.py` for the wavevector trajectories, the
  moment-model apparent-diffusion suppression curve, and the walker runner
  validated against `exp(-b D)`.
