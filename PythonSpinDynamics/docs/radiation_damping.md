# Radiation Damping

The radiation-damping implementation follows the rotating-frame back-action
model in Section 10.2.5 of the local Measurements textbook. The probe is
described by the same tuned or matched parameter sets used by the regular pulse
sequence workflows, and the nonlinear feedback strength is set by

```text
Trd = 2 / (gamma * mu0 * eta * M0 * Q)
```

where `eta` is the magnetic-energy fill factor, `M0` is the equilibrium
magnetization density in A/m, and `Q` is the loaded probe quality factor.

## Models

`spin_dynamics.radiation_damping` provides two deterministic back-action
models:

- `model="instant"` uses the high-Q on-resonance feedback field directly
  proportional to the conjugate transverse magnetization.
- `model="circuit"` adds a first-order probe ringdown state with time constant
  `2 Q / omega0`, optional feedback phase, and optional probe detuning.

Magnetization is propagated in normalized units, so `mth=1` corresponds to the
sample magnetization density used to build the probe coupling.

## Workflows

For analytic checks and quick experiments, use:

```python
from spin_dynamics.workflows import run_radiation_damping_fid

result = run_radiation_damping_fid(
    probe="matched",
    fill_factor=0.7,
    equilibrium_magnetization=0.8,
    flip_angle=1.0,
    model="instant",
)
```

For finite tuned or matched CPMG trains, pass an opt-in `radiation_damping`
mapping:

```python
from spin_dynamics.workflows import run_tuned_cpmg_train

train = run_tuned_cpmg_train(
    numpts=51,
    num_echoes=8,
    radiation_damping={
        "fill_factor": 0.7,
        "equilibrium_magnetization": 0.8,
        "model": "circuit",
        "detuning": 2.0e4,
        "apply_during_pulses": True,
    },
)
```

By default, finite-sequence damping is applied during free-precession and
acquisition windows. Set `apply_during_pulses=True` to use an operator-split
approximation during RF pulse matrices as well. This keeps the regular
MATLAB-compatible matrix workflow intact while exposing the nonlinear feedback
path where it is needed.

Use `water_proton_sample`, `hyperpolarized_proton_sample`, or
`proton_thermal_magnetization_density` to estimate `M0`. Use
`normalized_radiation_damping_weights(density, sensitivity)` when an ensemble
should damp through receive/transmit sensitivity weighting instead of equal
isochromat weights.

## NMR Maser Example

An inverted sample changes the same feedback loop from damping to gain. The
helper `simulate_nmr_maser` represents optical/RF pumping as longitudinal
relaxation toward an inverted `pump_mz`. In the instant high-Q limit, the
small-signal threshold is approximately:

```text
-pump_mz / Trd > 1 / T2
```

The example scripts show this transition:

```powershell
python examples\nmr_maser.py
python examples\plot_nmr_maser.py --output results\nmr_maser.png
```

The default plot uses pump levels at `0.5x`, `2x`, and `16x` threshold so the
strongest trace reaches nonlinear saturation and depletes the inversion. Use
`--pump-multipliers 0.5,2,8,16` to choose other threshold multiples. The
default examples use `model="instant"` for a compact threshold plot; add
`--model circuit` to include finite probe ringdown and optional detuning.

## Validation

The on-resonance hard-pulse FID is validated against the analytic Section
10.2.5 envelope:

```text
|mxy(t)| = M0 / cosh(t / Trd - log(tan(theta / 2)))
```

The focused test suite checks this envelope, conservation of normalized
magnetization for the no-relaxation instant model, circuit lag relative to the
instant model, CPMG workflow coupling, RF-pulse damping mode, sample presets,
sensitivity-squared weighting, and NMR maser threshold growth.

## Noise Boundary

Radiation damping here is deterministic probe back-action. The existing
`spin_dynamics.noise` helpers remain the received-signal noise layer for white
or probe-colored receiver output noise. A future source-level spin-noise model
can couple stochastic magnetization fluctuations into the same probe feedback
state, but that is intentionally separate from the validated deterministic RD
path.
