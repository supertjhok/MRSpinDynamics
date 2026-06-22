# ESR Models

The `spin_dynamics.esr` package provides a first ESR/EPR surface for
single-electron spin-1/2 systems. Hamiltonians are represented in radians per
second, matching the dense Hamiltonian conventions used elsewhere in the Python
package.

The initial model uses the electron Zeeman Hamiltonian

```text
H = 2*pi*(mu_B/h) * B0^T g S
```

where `g` may be supplied as a scalar isotropic value, a three-component
principal-axis vector, or a full `3x3` tensor. Static-field and microwave-field
directions are expressed in the `g`-tensor principal-axis frame.

## Basic Resonance

```python
from spin_dynamics.esr import ESRSpinSystem, resonance_frequency_hz

system = ESRSpinSystem(g_tensor=2.0023)
frequency_hz = resonance_frequency_hz(system, [0.0, 0.0, 0.35])
```

For field-swept experiments, use the inverse helper:

```python
from spin_dynamics.esr import resonance_field_tesla

field_tesla = resonance_field_tesla(system, microwave_frequency_hz=9.5e9)
```

## Anisotropic g Tensors

```python
from spin_dynamics.esr import ESRSpinSystem, effective_g_value

system = ESRSpinSystem(g_tensor=[2.0, 2.1, 2.2])
g_parallel_z = effective_g_value(system, [0.0, 0.0, 1.0])
```

For a direction `n`, the effective value is `|g^T n|`. This convention also
supports non-diagonal tensors.

## Spectra

Frequency-swept spectra are useful at fixed field:

```python
from spin_dynamics.esr import simulate_frequency_spectrum

result = simulate_frequency_spectrum(
    system,
    b0_tesla=0.35,
    orientations="single",
    broadening_hz=1e6,
)
```

Field-swept spectra are useful for conventional continuous-wave ESR:

```python
from spin_dynamics.esr import simulate_field_sweep

result = simulate_field_sweep(
    system,
    microwave_frequency_hz=9.5e9,
    orientations="powder",
    broadening_tesla=1e-4,
)
```

Both helpers return broadened Gaussian spectra plus orientation-resolved
`ESRLine` metadata. Intensities include the microwave `B1` projection onto the
transition dipole, so a `B1` direction parallel to the spin quantization axis is
dark.

For conventional CW-style displays, the spectrum helpers also accept
`lineshape="gaussian"` or `"lorentzian"` and
`detection_mode="absorption"` or `"derivative"`.

```python
result = simulate_field_sweep(
    system,
    microwave_frequency_hz=9.5e9,
    orientations="single",
    broadening_tesla=1e-4,
    lineshape="lorentzian",
    detection_mode="derivative",
)
```

## Static Disorder and Strain

Static distributions are represented by weighted `ESRDistributionSample`
objects. The convenience `static_disorder_grid` samples diagonal `g` strain and
applied-field offsets with Gaussian weights, then the distribution spectrum
helpers combine those samples with the ordinary single-crystal or powder
orientation grid.

```python
from spin_dynamics.esr import (
    static_disorder_grid,
    simulate_field_sweep_distribution,
)

disorder = static_disorder_grid(
    system,
    g_std=[0.0, 0.0, 0.005],
    field_std_tesla=0.1e-3,
    g_points=5,
    field_points=5,
)

result = simulate_field_sweep_distribution(
    disorder,
    microwave_frequency_hz=9.5e9,
    orientations="powder",
    broadening_tesla=0.05e-3,
)
```

## Pulsed ESR

The pulsed helpers use a single rotating frame with the rotating-wave
approximation. The `nutation_hz` parameter is the on-resonance spin-1/2 Rabi
rate for the selected microwave-field direction, so a rectangular 90-degree
pulse has duration `1 / (4 * nutation_hz)`.

```python
import numpy as np

from spin_dynamics.esr import (
    ESRSpinSystem,
    flip_angle_duration,
    resonance_frequency_hz,
    simulate_fid,
    simulate_hahn_echo,
)

system = ESRSpinSystem(g_tensor=2.0023)
b0 = [0.0, 0.0, 0.339]
carrier = resonance_frequency_hz(system, b0)
nutation_hz = 5e6
t90 = flip_angle_duration(np.pi / 2, nutation_hz)
t180 = flip_angle_duration(np.pi, nutation_hz)

fid = simulate_fid(
    system,
    b0,
    nutation_hz=nutation_hz,
    pulse_duration_seconds=t90,
    times_seconds=np.linspace(0.0, 5e-6, 256),
    rf_frequency_hz=carrier,
)

echo = simulate_hahn_echo(
    system,
    b0,
    nutation_hz=nutation_hz,
    excitation_duration_seconds=t90,
    refocus_duration_seconds=t180,
    tau_seconds=2e-6,
    times_seconds=np.linspace(0.0, 4e-6, 256),
    rf_frequency_hz=carrier,
)
```

The Hahn-echo helper simulates one isochromat. To reproduce a visible echo
envelope, sum several detuned isochromats as shown in
`examples/plot_esr_pulsed_echo.py`.

## Pulsed Relaxation

For physical relaxation during pulses, free evolution, and acquisition, pass an
`ESRRelaxationModel` to the pulsed helpers. The model uses Liouville-space
propagation in the energy eigenbasis: `t1_seconds` damps population differences
and `t2_seconds` damps coherences while preserving trace.

```python
from spin_dynamics.esr import ESRRelaxationModel, simulate_fid

fid = simulate_fid(
    system,
    b0,
    nutation_hz=nutation_hz,
    pulse_duration_seconds=t90,
    times_seconds=np.linspace(0.0, 5e-6, 256),
    rf_frequency_hz=carrier,
    relaxation=ESRRelaxationModel(t1_seconds=50e-6, t2_seconds=3e-6),
)
```

The older `t2_seconds` argument is a simple post-propagation envelope retained
for quick demos. When using `ESRRelaxationModel`, leave `t2_seconds=inf` to
avoid double-counting coherence decay. See `examples/plot_esr_relaxation.py`
for a commented comparison of FID T2 decay, Hahn-echo T2 decay, and T1
population relaxation.

## Hyperfine Coupling

The first hyperfine layer models one electron spin-1/2 coupled isotropically to
one or more nuclei. It builds a dense electron-first product Hilbert space and
uses

```text
H = H_eZ + H_nZ + 2*pi * sum_k A_k S . I_k
```

with `A_k` in hertz. The field-swept helper scans the applied field, exactly
diagonalizes the Hamiltonian at each field point, and sums ESR-active
transitions according to their microwave `B1` matrix elements.

```python
from spin_dynamics.esr import (
    NuclearSite,
    diagonalize_hyperfine_system,
    electron_nuclear_system,
    simulate_hyperfine_field_sweep,
)

system = electron_nuclear_system(
    [20e6],
    nuclei=[NuclearSite("H1", gamma_hz_per_t=42.577e6)],
    g_tensor=2.0023,
)

eigensystem = diagonalize_hyperfine_system(system, [0.0, 0.0, 0.34])
sweep = simulate_hyperfine_field_sweep(
    system,
    microwave_frequency_hz=9.5e9,
    broadening_hz=0.5e6,
)
```

For a one-nucleus spin-1/2 example, see
`examples/plot_esr_hyperfine_doublet.py`.

## Current Scope

The first ESR surface intentionally covers:

- single-electron spin-1/2 Zeeman Hamiltonians;
- scalar, diagonal, or full `g` tensors;
- exact dense diagonalization for one electron;
- single-crystal and powder orientation grids;
- fixed-field frequency spectra and fixed-frequency field sweeps.
- Gaussian/Lorentzian absorption or first-derivative CW spectra;
- diagonal `g` strain and applied-field offset distributions;
- rectangular-pulse FID and Hahn-echo density-matrix simulations for one
  orientation/isochromat.
- Liouville-space T1/T2 relaxation for pulsed ESR.
- isotropic electron-nuclear hyperfine Hamiltonians and field-swept spectra.

Not yet included:

- anisotropic hyperfine tensors, exchange, or dipolar coupling;
- higher-spin zero-field splitting;
- saturation, temperature-dependent equilibrium magnetization, or microwave
  resonator effects.
