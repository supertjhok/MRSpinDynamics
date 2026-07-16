# Radiation Damping

Radiation damping is feedback from the detected sample magnetization through
the resonant probe. A large transverse magnetization drives current in the
coil; the coil field then acts back on the same spins. The result can be a
non-exponential FID, rapid return toward equilibrium, altered echo trains, or an
idealized maser-like instability.

Use this page when sample-probe back-action is strong enough that receive
coupling cannot be treated as passive. The implementation reuses the tuned or
matched probe parameters from ordinary workflows and offers an instantaneous
high-Q limit and a finite-ringdown circuit model. It is deterministic and does
not by itself include stochastic spin noise.

The model follows the rotating-frame treatment in Section 10.2.5 of the local
Measurements textbook, with the feedback-phase convention explained below.
Its nonlinear strength is set by

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

## Feedback Phase Convention

The integrators drive the feedback with the phase-independent (Bloom 1957)
target `-Mxy / Trd`, so the damping rate is the same for every transverse
azimuth, `|M|^2` is conserved by the feedback (energy flows from the spins to
the circuit for `Mz > 0`), and `Mz` recovers monotonically after a pulse of
any phase.

Textbook Eq. 10.29/10.30 instead writes the feedback field as proportional to
the *conjugate* `M*_xy`. That conjugate traces to a sideband bookkeeping slip:
Eqs. 10.26-10.28 derive the feedback as the coefficient of the `e^{+i w0 t}`
sideband of the real lab-frame signal, but the book's own rotating-frame
convention (`M_lab = M e^{-i w0 t}`, demodulation by `e^{-i w0 t}`) lives on
the negative sideband, so converting to the rotating frame requires one more
conjugation that Eq. 10.29 drops. Taken literally, the conjugate form is a
phase-sensitive amplifier with no energy source: it anti-damps one transverse
quadrature and drives `Mz` *downward* for magnetization along that quadrature.
The two forms coincide exactly on the `+/-y` axis, which is where the book's
own sech validation (Eq. 10.32) and all single-phase FID/maser trajectories
live -- so the distinction only matters for multi-phase or stochastic inputs
(CPMG with off-axis components, spin noise).

The validated sech envelope, maser threshold, and circuit-lag behavior are
identical in both conventions on their validation trajectories; the
phase-independence itself is pinned by
`test_damping_is_independent_of_transverse_phase`.

## Noise Boundary

Radiation damping here is deterministic probe back-action. The existing
`spin_dynamics.noise` helpers remain the received-signal noise layer for white
or probe-colored receiver output noise. The source-level spin-noise model now
exists in `spin_dynamics.spin_noise` (see [spin_noise.md](spin_noise.md)): it
couples stochastic magnetization fluctuations into the same probe feedback
state and shares the coupling constant through
`R_n0 = R_coil * T2 / (2 * Trd)`, but remains intentionally separate from the
validated deterministic RD path. Both modules use the same phase-independent
feedback form `-mxy / Trd` (see "Feedback Phase Convention" above).
