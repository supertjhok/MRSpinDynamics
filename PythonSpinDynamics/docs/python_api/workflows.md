# Workflows

Workflow helpers live under `spin_dynamics.workflows`.

Application code should prefer the public workflow runners below. The
lower-level examples remain useful for validation, porting, and debugging
individual MATLAB reference paths.

## Prepolarization

Use `spin_dynamics.prepolarization` when a sample relaxes in a polarizing
magnet before an NMR sequence begins. The helpers return prepared `m0` values
in the same normalized units used by the kernels, plus the sequence equilibrium
`mth`:

```python
from spin_dynamics.prepolarization import prepolarized_state

prepared = prepolarized_state(
    polarizing_field_tesla=0.1,
    detection_field_tesla=50e-6,
    prepolarization_time_seconds=2.0,
    t1_seconds=1.5,
)

params = prepared.as_parameters()
```

For transport and flow experiments, `prepolarized_flow_state` computes the
residence time from path length and speed before applying the same T1 recovery
model.

## BPP Relaxation

Use `spin_dynamics.relaxation` to estimate scalar `T1` and `T2` values from a
rotational correlation time. The default BPP coefficients are proportional to
`J(w0) + 4 J(2w0)` for `R1` and
`1.5 J(0) + 2.5 J(w0) + J(2w0)` for `R2`; the coupling scale absorbs the
dipolar constants and convention-specific prefactors:

```python
import numpy as np

from spin_dynamics.relaxation import BPPRelaxationModel

model = BPPRelaxationModel(
    angular_frequency_rad_per_s=2 * np.pi * 2.0e6,
    tau_ref_seconds=2.0e-9,
    coupling_scale_per_second2=3.0e9,
    activation_energy_j_per_mol=12_000.0,
)

rates = model.rates([280.0, 300.0, 320.0])
params = rates.as_parameters()
```

For other dipolar conventions or phenomenological fits, provide custom
`r1_coefficients` and `r2_coefficients` in the order
`(J(0), J(w0), J(2w0))`.

## Public CPMG Runners

Use these for application code and examples:

```python
from spin_dynamics.workflows import (
    run_ideal_cpmg,
    run_ideal_cpmg_train,
    run_matched_cpmg,
    run_tuned_cpmg,
    run_untuned_cpmg,
)

result = run_tuned_cpmg(numpts=101, maxoffs=10)
train = run_ideal_cpmg_train(
    numpts=101,
    maxoffs=10,
    num_echoes=8,
    auto_refine_grid=True,
    num_workers=None,
)
```

Each runner returns a `CPMGResult` with:

- `del_w`: normalized offset grid;
- `masy`: asymptotic magnetization before receive filtering;
- `mrx`: received spectrum, equal to `masy` for the ideal workflow;
- `echo`, `tvect`: time-domain echo and time vector;
- `snr`: probe SNR where available, otherwise `None`;
- `probe`: one of `ideal`, `tuned`, `untuned`, or `matched`.

Finite train runners return a `CPMGTrainResult` with:

- `del_w`: normalized offset grid;
- `mrx`: acquired spectra with shape `(num_echoes, numpts)`;
- `echo`, `tvect`: direct-summed time-domain echoes and acquisition vector;
- `echo_integrals`: trapezoidal integrals of each echo;
- `sequence_time`: physical echo-center times in seconds;
- `probe`: one of `ideal`, `tuned`, `untuned`, or `matched`.

Finite train runners also accept `rephase_action`, `rephase_safety_factor`,
`auto_refine_grid`, and `num_workers`. By default they warn when the normalized
angular offset grid may rephase before the train finishes. Set
`auto_refine_grid=True` to increase `numpts` before pulse matrices are built,
and set `num_workers=None` to use the available CPU count for chunked
isochromat propagation.

Default CPMG branch subtraction is represented by
`spin_dynamics.phase_cycling.cpmg_two_step_phase_cycle()`. A phase cycle is a
scan table: rows are cycle steps, columns are named logical RF pulse roles, and
each row also carries a receiver phase and branch weight. The default CPMG/PAP
table has one pulse column, `excitation`, with rows `pi/2` and `3*pi/2`; the
weights `+1` and `-1` reproduce the historical `(branch1 - branch2) / 2`
combination. Results expose this table as `result.phase_cycle`. Arbitrary
user-supplied phase-cycle tables are not yet accepted by the public CPMG
runners.

PGSTE walker results also expose `result.phase_cycle`, but with a different
meaning: it is a one-row selected-pathway table for the stimulated echo, not a
multi-branch receiver-weighted simulation.

Finite CPMG train runners also accept an opt-in `absolute_phase` mapping from
`spin_dynamics.absolute_phase`. With only `rf_frequency_hz`, the ideal workflow
tracks the laboratory-frame RF phase at the excitation and refocusing pulses
while preserving the validated default pulse shapes. Tuned, untuned, and
matched probe train workflows use that phase to solve the probe waveform for
each finite-train pulse and build per-pulse rotation matrices from the
discretized rotating-frame shape:

```python
train = run_tuned_cpmg_train(
    numpts=21,
    num_echoes=32,
    absolute_phase={
        "rf_frequency_hz": 0.25 / 200e-6,
    },
)

phase_step = train.absolute_phase.delta_refocus_phase_cycles
```

Long finite trains can reuse pulse solves by adding `phase_bins`. Refocusing
absolute phases are snapped to the nearest uniform bin around the RF cycle,
while `train.absolute_phase.refocus_absolute_phase_rad` still records the
unsnapped schedule. The metadata also reports the matrix phase used for each
echo, the phase-bin index, and a `refocus_pulse_library` containing the unique
resolved shapes:

```python
train = run_tuned_cpmg_train(
    num_echoes=256,
    absolute_phase={
        "rf_frequency_hz": 0.03125 / 200e-6,
        "phase_bins": 32,
    },
)

library = train.absolute_phase.refocus_pulse_library
matrix_phases = train.absolute_phase.refocus_matrix_phase_rad
```

For direct debugging or teaching plots, solve the same probe pulse shapes without
running a full echo train:

```python
import numpy as np

from spin_dynamics.pulse_diagnostics import solve_probe_pulse_shape_sweep

sweep = solve_probe_pulse_shape_sweep(
    probe="tuned",
    absolute_phase_rad=2 * np.pi * np.array([0.0, 0.25, 0.5, 0.75]),
    numpts=17,
)

shape = sweep.shapes[1]
drive = shape.drive
quadrature_fraction = shape.quadrature_energy_fraction
library = sweep.pulse_shape_library
```

For ideal trains and for custom reduced models, add a `transient_model` mapping
to perturb pulse phase or amplitude as a function of absolute RF phase:

```python
train = run_ideal_cpmg_train(
    numpts=51,
    num_echoes=128,
    absolute_phase={
        "rf_frequency_hz": 0.125 / 200e-6,
        "transient_model": {
            "kind": "longitudinal_phase_kick",
            "phase_amplitude_rad": 0.043,
        },
    },
)
```

The simple transient models are phenomenological. They are intended for
synchronization sweeps and for plumbing sequence timing through the simulator;
measured or circuit-derived pulse-shape libraries can use the same
`absolute_phase` submodule.
For measured or precomputed pulse shapes, pass `kind="library"` with
`absolute_phase_rad` samples and one or more pulse-kind entries. Interpolation
is performed on the complex RF drive, so wrapped phases remain continuous:

```python
import numpy as np

train = run_ideal_cpmg_train(
    num_echoes=8,
    absolute_phase={
        "rf_frequency_hz": 0.25 / 200e-6,
        "transient_model": {
            "kind": "library",
            "absolute_phase_rad": [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
            "shapes": {
                "refocusing": {
                    "duration": [np.pi],
                    "phase": [[0.0], [0.15], [0.0], [-0.15]],
                    "amplitude": [[1.0], [0.8], [1.0], [1.2]],
                }
            },
        },
    },
)
```

The same library interface can be generated from low-order circuit models:

```python
import numpy as np

from spin_dynamics.absolute_phase import (
    AbsolutePhaseSpec,
    InterpolatedPulseShapeModel,
    build_nonresonant_circuit_pulse_library,
)

rf_frequency_hz = 0.25 / 200e-6
library = build_nonresonant_circuit_pulse_library(
    absolute_phase_rad=np.linspace(0.0, 2 * np.pi, 16, endpoint=False),
    rf_frequency_hz=rf_frequency_hz,
    pulse_duration_seconds=50e-6,
    time_scale_rad_per_s=(np.pi / 2) / 25e-6,
    tau_seconds=12e-6,
)

train = run_ideal_cpmg_train(
    num_echoes=8,
    absolute_phase=AbsolutePhaseSpec(
        rf_frequency_hz=rf_frequency_hz,
        transient_model=InterpolatedPulseShapeModel(library),
    ),
)
```

`build_tuned_resonator_pulse_library` provides a second-order tuned-resonator
analogue for reduced ideal-workflow studies. Prefer the finite tuned, untuned,
or matched probe workflows when the goal is to reproduce absolute-phase
transients from solved probe waveforms.

## Radiation Damping

Radiation damping is available as an opt-in nonlinear probe back-action model.
The FID workflow is the analytic validation anchor:

```python
from spin_dynamics.workflows import run_radiation_damping_fid

fid = run_radiation_damping_fid(
    probe="matched",
    fill_factor=0.7,
    equilibrium_magnetization=0.8,
    flip_angle=1.0,
)
```

Finite tuned and matched CPMG train workflows accept a `radiation_damping`
mapping. This preserves the ordinary pulse-sequence API while adding nonlinear
feedback during free windows, acquisition, and optionally RF pulse matrices:

```python
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

See `docs/radiation_damping.md` for equations, sample presets, detuning
controls, sensitivity weighting, validation notes, and the boundary with the
existing received-noise layer.

## Probe Parameter Sweeps

```python
from spin_dynamics.workflows import (
    run_matched_mistuning_sweep,
    run_matched_q_sweep,
    run_matched_z_magnetization_q_sweep,
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
)

tuned_q = run_tuned_q_sweep(q_values=[20, 50, 80], numpts=101)
matched_detune = run_matched_mistuning_sweep(offsets=[-2, 0, 2], numpts=101)
matched_z = run_matched_z_magnetization_q_sweep(q_values=[20, 50, 80], numpts=101)
```

Sweep runners return `CPMGParameterSweepResult` with:

- `values` and `value_label`: the swept Q values or frequency-error offsets;
- `del_w`: normalized offset grid;
- `mrx`: received spectra with shape `(num_values, numpts)`;
- `echo`, `tvect`: direct-summed echoes and common echo time vector;
- `snr`: matched-filter SNR for each sweep point;
- `probe` and `sweep`: metadata labels.

The mistuning offsets are in units of `fin / Q`, matching the MATLAB scripts.
The sweep-level `num_workers` option parallelizes independent sweep points.

`run_matched_z_magnetization_q_sweep` returns `ZMagnetizationSweepResult` with:

- `values` and `value_label`: swept Q values;
- `del_w`: normalized offset grid;
- `mz`: final z magnetization with shape `(num_values, numpts)`;
- `tvect`: matched-probe excitation pulse time samples;
- `probe` and `sweep`: metadata labels.

MATLAB references:

- `CompareQ/sim_tuned_probe_coil_Q.m`
- `CompareQ/sim_matched_probe_coil_Q.m`
- `CompareMistuned/tuned_probe/sim_tuned_probe_mistuned.m`
- `CompareMistuned/matched_probe/sim_matched_probe_mistuned.m`
- `z_mag/z_Mag_Q.m`
- `calc_masy/calc_masy_matched_nut.m`

## Ideal CPMG

```python
from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.parameters import set_params_ideal
from spin_dynamics.workflows.cpmg import calc_masy_ideal

sp, pp = set_params_ideal(numpts=101)
masy = calc_masy_ideal(sp, pp)
echo, tvect = calc_time_domain_echo(masy, sp.del_w)
```

MATLAB references:

- `Params/set_params_ideal.m`
- `calc_masy/calc_masy_ideal.m`
- `calc_echo/calc_time_domain_echo.m`

## Finite Ideal CPMG Train

```python
from spin_dynamics.workflows import run_ideal_cpmg_train

result = run_ideal_cpmg_train(
    numpts=101,
    maxoffs=10,
    num_echoes=8,
    auto_refine_grid=True,
    num_workers=None,
)
```

This public workflow assembles a no-probe PAP phase-cycled CPMG echo train,
uses `calc_macq_ideal_probe_relax4` for finite acquisition with relaxation, and
direct-sums each acquired spectrum into a time-domain echo.

MATLAB references:

- `time_varying_field/sim_cpmg_ideal_tv.m`
- `calc_macq/calc_macq_ideal_probe_relax4.m`

## Time-Varying-Field CPMG

```python
from spin_dynamics.workflows import (
    run_ideal_time_varying_amplitude_sweep,
    run_matched_time_varying_cpmg_final,
    sinusoidal_field_waveform,
)

waveform = sinusoidal_field_waveform(num_echoes=16)
result = run_ideal_time_varying_amplitude_sweep(
    amplitudes=[0, 0.5, 1.0, 2.0],
    waveform=waveform,
    numpts=101,
    auto_refine_grid=True,
)
matched_final = run_matched_time_varying_cpmg_final(
    0.5 * waveform,
    numpts=51,
    q_value=50,
    auto_refine_grid=True,
)
```

`run_ideal_time_varying_cpmg_final` returns the final echo for a supplied
per-echo normalized B0 offset waveform. The amplitude sweep wrapper returns
`IdealTimeVaryingSweepResult` with echoes, echo integrals, and matched-filter
signals versus fluctuation amplitude. Field offsets use MATLAB's normalized
`w_0t = gamma * B_0t / w_1n` convention.

Probe-aware variants are available as `run_tuned_time_varying_cpmg_final`,
`run_untuned_time_varying_cpmg_final`, and
`run_matched_time_varying_cpmg_final`, with matching
`run_*_time_varying_amplitude_sweep` wrappers. These use the same per-echo B0
offset assembly as the ideal path, but build the refocusing rotations from the
tuned, untuned, or matched pulse responses and apply the corresponding receiver
transfer functions. The probe-aware result containers include `probe`,
`q_value`, and `mistuning_offset` metadata.

These fixed-grid final-echo workflows support the same rephasing controls as
the finite CPMG train APIs: `rephase_action`, `rephase_safety_factor`, and
`auto_refine_grid`. Enable `auto_refine_grid=True` for long echo trains or
large offset spans so the grid is refined before the per-offset RF matrices are
built.

MATLAB references:

- `time_varying_field/sim_cpmg_ideal_tv_final.m`
- `time_varying_field/compare_cpmg_results_ideal_tv.m`
- `time_varying_field/compare_cpmg_results_ideal_v0crit.m`

## CPMG-IR Finite Trains

```python
from spin_dynamics.workflows import run_matched_cpmg_ir_train

result = run_matched_cpmg_ir_train(
    num_echoes=4,
    echo_spacing_seconds=0.5e-3,
    tauvect=[0.5e-3, 1.0e-3, 2.0e-3],
    numpts=21,
    tau_workers=2,
    num_workers=1,
    rephase_action="ignore",
)
```

`run_ideal_cpmg_ir_train`, `run_tuned_cpmg_ir_train`,
`run_untuned_cpmg_ir_train`, and `run_matched_cpmg_ir_train` run homogeneous
inversion-recovery CPMG trains over `tauvect`. They return CPMG-IR result
containers with:

- `tauvect`: inversion-delay vector in seconds;
- `del_w`: normalized offset grid;
- `mrx`: received spectra with shape `(num_tau, num_echoes, numpts)`;
- `echo`, `tvect`: direct-summed echoes and common acquisition vector;
- `echo_integrals`: trapezoidal echo integrals with shape
  `(num_tau, num_echoes)`;
- `sequence_time`: echo-center times in seconds.

The `tau_workers` option parallelizes independent inversion delays. The
`num_workers` option is passed through to the chunked isochromat backend inside
each finite acquisition. For long `tauvect` values, use the same rephasing
checks or `auto_refine_grid=True` strategy as the other finite-train workflows.

MATLAB references:

- `Sim_CPMG_IR/sim_cpmg_ir_matched_probe_relax4.m`
- `Sim_CPMG_IR/sim_cpmg_ir_matched_probe_compare.m`

## Finite-Train Probe Parameter Sweeps

```python
from spin_dynamics.workflows import (
    run_matched_finite_q_sweep,
    run_tuned_finite_mistuning_sweep,
)

matched_q = run_matched_finite_q_sweep(
    q_values=[20, 50, 80],
    num_echoes=8,
    numpts=101,
    auto_refine_grid=True,
    num_workers=None,
    sweep_workers=3,
)
tuned_detune = run_tuned_finite_mistuning_sweep(
    offsets=[-1, 0, 1],
    num_echoes=8,
    numpts=101,
    auto_refine_grid=True,
)
```

Finite sweep runners return `CPMGFiniteParameterSweepResult` with:

- `values` and `value_label`: swept Q values or frequency-error offsets;
- `del_w`: normalized offset grid, after optional refinement;
- `mrx`: received spectra with shape `(num_values, num_echoes, numpts)`;
- `echo`, `tvect`: direct-summed finite-train echoes and common acquisition
  vector;
- `echo_integrals`: trapezoidal integrals with shape
  `(num_values, num_echoes)`;
- `sequence_time`: echo-center times in seconds.

Available wrappers:

- `run_tuned_finite_q_sweep`
- `run_untuned_finite_q_sweep`
- `run_matched_finite_q_sweep`
- `run_tuned_finite_mistuning_sweep`
- `run_untuned_finite_mistuning_sweep`
- `run_matched_finite_mistuning_sweep`

These are Python-native extensions around `run_tuned_cpmg_train`,
`run_untuned_cpmg_train`, and `run_matched_cpmg_train`. They preserve the
finite-train options for rephasing checks, `auto_refine_grid`, and chunked
isochromat propagation through `num_workers`; `sweep_workers` parallelizes
independent sweep points.

## Matched Diffusion CPMG

```python
from spin_dynamics.workflows import run_matched_diffusion_q_sweep

result = run_matched_diffusion_q_sweep(
    q_values=[20, 50],
    num_echoes=3,
    numpts=21,
    dz=50e-6,
    auto_refine_grid=True,
    sweep_workers=2,
)
```

For a tuned-probe counterpart with stronger probe-solved absolute-phase
contrast, use `run_tuned_diffusion_cpmg`:

```python
from spin_dynamics.workflows import run_tuned_diffusion_cpmg

tuned = run_tuned_diffusion_cpmg(
    num_echoes=8,
    dz=50e-6,
    auto_refine_grid=True,
    absolute_phase={
        "rf_frequency_hz": 0.25 / 1.0e-3,
        "phase_bins": 16,
    },
)
```

`run_matched_diffusion_cpmg` returns `MatchedDiffusionCPMGResult` with:

- `del_w`: normalized offset grid derived from `gamma * gradient * dz / w1`;
- `mrx`: diffusion-aware acquired spectra with shape `(num_echoes, numpts)`;
- `echo`, `tvect`: direct-summed echoes and common acquisition vector;
- `echo_integrals`: trapezoidal integrals for each echo;
- `q_value`, `diffusion_coefficient`, `diffusion_time`, `gradient`, and `dz`.

`run_matched_diffusion_q_sweep` returns `MatchedDiffusionQSweepResult` with
echo arrays and echo integrals stacked over Q. This path uses
`sim_spin_dynamics_arb10_diffusion`, an `arb10`-style modernization of
MATLAB's diffusion-aware kernel that keeps precomputed RF matrices and omits
the older acquisition-window convolution. Very high-Q diffusion sweeps should
be treated as a validation target because the current NumPy matched-probe
transient solver can become stiff; the compact matched-diffusion workflow is
currently solver-validated through Q=2000.

The matched diffusion workflow also accepts the finite-train `absolute_phase`
mapping. The diffusion-encoding pi pulse and each CPMG refocusing pulse are
solved at their laboratory-frame RF phase, optionally quantized with
`phase_bins`, while free-precession intervals continue to carry the diffusion
attenuation:

```python
from spin_dynamics.workflows import run_matched_diffusion_cpmg

combined = run_matched_diffusion_cpmg(
    num_echoes=16,
    echo_spacing_seconds=1.0e-3,
    diffusion_time=1.0e-3,
    q_value=50,
    absolute_phase={
        "rf_frequency_hz": 0.25 / 1.0e-3,
        "phase_bins": 16,
    },
)

metadata = combined.absolute_phase
encoding_phase = metadata.encoding_absolute_phase_rad
refocus_phases = metadata.refocus_absolute_phase_rad
```

`metadata.refocus_*` fields describe the echo-train refocusing pulses, while
`metadata.encoding_*` fields describe the diffusion-encoding pi pulse. The
full RF event list is available through `pulse_kind`, `pulse_start_seconds`,
`pulse_absolute_phase_rad`, and `pulse_matrix_phase_rad`.

**Diffusion model and assumptions.** The diffusion term models *free*
(unrestricted) diffusion in a *constant* background gradient. Each
free-precession interval attenuates the transverse coherence by
`exp(-(1/3) gamma**2 G**2 D t**3)` on that interval's own physical duration
`t`; over a CPMG train this reproduces the textbook Carr-Purcell law
`exp(-(1/12) gamma**2 G**2 D t_E**3 N)`. It is **not** a narrow-pulse PGSE
model: despite the name, `diffusion_time` is the encoding-block duration that
sets the gradient/refocusing geometry, not a Stejskal-Tanner `Delta`. The
analytic correctness of this law is covered by
`tests/test_diffusion_physics.py`.

Because the diffusion workflow also uses a uniform fixed offset grid, it
accepts `rephase_action`, `rephase_safety_factor`, and `auto_refine_grid`.
The Q sweep forwards those options to each matched-diffusion CPMG run.
For examples and exploratory plots, prefer either a physically narrow `dz` or
`auto_refine_grid=True`; otherwise a compact `numpts` can discretely rephase
inside the echo train and produce misleading echo integrals.

MATLAB references:

- `DIffusion_Example/Diff_Echo_Q.m`
- `Sim_Diffusion/sim_dif_matched_CPMG_noRx.m`
- `calc_macq_diff/calc_macq_matched_probe_relax_diff_noRx.m`
- `sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m`

## PGSE and PGSE-Prepared CPMG

The PGSE workflow layer covers rectangular Stejskal-Tanner diffusion encoding
and PGSE-prepared CPMG echo trains. It has two complementary backends:

- `run_pgse_moment`: deterministic gradient-moment tracking. It computes
  \(b = \int q(t)^2 dt\) for the effective gradient waveform and is the fast
  path for homogeneous unrestricted diffusion.
- `run_pgse_walkers`: explicit random walkers through the motion sequence
  machinery. It is slower and stochastic, but it can handle motion through B0/B1
  maps, boundaries, and future inhomogeneous-gradient extensions.

For a rectangular PGSE pair, `pgse_b_value` and the moment backend reduce to the
standard Stejskal-Tanner result:

```text
b = (gamma * G * delta)^2 * (Delta - delta / 3)
```

where `Delta` is the leading-edge separation between the two gradient lobes.
The physical gradient lobes are scheduled with the same polarity across the
180-degree refocusing pulse; the refocusing pulse flips the coherence-frame
sign, so stationary spins refocus and diffusion produces the attenuation.

```python
from spin_dynamics.workflows import run_pgse_moment

pgse = run_pgse_moment(
    num_echoes=8,
    gradient_amplitude=0.12,      # T/m
    gradient_duration=2.5e-3,     # delta
    diffusion_time=28.0e-3,       # Delta
    diffusion_coefficient=1.2e-9,
    first_echo_time_seconds=56e-3,
    echo_spacing_seconds=8e-3,
    t2_seconds=80e-3,
)

print(pgse.b_value, pgse.signal.shape)
```

`num_echoes=1` is the ordinary spin-echo PGSE case. Larger `num_echoes` use the
same PGSE preparation followed by a compact CPMG-style echo train in the result
time axis.

Use the walker backend when spatial motion matters:

```python
from spin_dynamics.workflows import run_pgse_walkers

walkers = run_pgse_walkers(
    gradient_amplitude=0.05,
    gradient_duration=2.0e-3,
    diffusion_time=16.0e-3,
    diffusion_coefficient=2.3e-9,
    walkers_per_cell=12000,
    seed=123,
)

echo = walkers.signal[0]
```

The walker tests compare the diffusing signal against a zero-diffusion walker
reference and the moment-backend Stejskal-Tanner attenuation. Increase
`walkers_per_cell` and substeps for production convergence studies.

### Restricted diffusion and pore geometry

Because the walker backend integrates explicit displacements, it can model
diffusion restricted by hard walls or exchanged through semi-permeable
membranes -- physics the moment backend cannot express. The `boundary` argument
accepts the rectangular modes `"reflect"`, `"periodic"`, and `"clip"`, or any
callable mapping `(N, 2)` positions to confined positions.
`spin_dynamics.motion.make_circular_reflector` supplies a reflecting circular
wall for a pore:

```python
from spin_dynamics.motion import make_circular_reflector
from spin_dynamics.workflows import run_pgse_walkers
import numpy as np

radius = 5.0e-6
axis = np.linspace(-radius, radius, 21)
xx, zz = np.meshgrid(axis, axis, indexing="ij")
rho = (xx**2 + zz**2 <= radius**2).astype(float)  # uniform disc

walkers = run_pgse_walkers(
    rho=rho,
    x_axis=axis,
    z_axis=axis,
    gradient_amplitude=2.0,        # large G to reach the q-space regime
    gradient_duration=0.4e-3,      # short delta (narrow-pulse / SGP limit)
    diffusion_time=80.0e-3,        # long Delta >> a^2/D (full pore sampling)
    diffusion_coefficient=2.3e-9,
    boundary=make_circular_reflector((0.0, 0.0), radius),
    walkers_per_cell=28,
    substeps_per_interval=80,
    seed=2026,
)
```

In the narrow-pulse, long-mixing limit the echo follows the pore form factor and
develops *diffusive diffraction* minima at `q_ang a = 3.83, 7.02, ...` for a disc
of radius `a`, where `q_ang = gamma * G * delta`. Keep the per-substep hop
`sqrt(2 D dt)` well below the pore size (raise `substeps_per_interval`) for
accurate reflection. See the slab-pore and circular-pore diffraction examples.

For slow exchange between two compartments, use
`spin_dynamics.motion.make_semipermeable_plane`. The membrane is an internal
line (`x = interface` or `z = interface`) inside the rectangular bounds. A
walker that crosses the line transmits with probability
`1 - exp(-exchange_rate * dt)`; otherwise it reflects from the membrane. This
exchange-rate form remains stable when the motion interval is split into more
substeps:

```python
from spin_dynamics.motion import make_semipermeable_plane
from spin_dynamics.workflows import run_pgse_walkers
import numpy as np

x = np.linspace(-10e-6, 10e-6, 41)
z = np.array([-0.5e-6, 0.5e-6])
rho = np.ones((x.size, z.size))
membrane = make_semipermeable_plane(
    0.0,
    exchange_rate=25.0,  # s^-1 in this seconds-based workflow
    axis="x",
)

walkers = run_pgse_walkers(
    rho=rho,
    x_axis=x,
    z_axis=z,
    gradient_amplitude=0.2,
    gradient_duration=1.0e-3,
    diffusion_time=40.0e-3,
    diffusion_coefficient=2.0e-9,
    boundary=membrane,
    walkers_per_cell=64,
    substeps_per_interval=16,
    seed=2026,
    jitter=True,
)
```

Set `exchange_rate=0` for an impermeable internal wall and `np.inf` for a
freely transmitting interface. The outer rectangular boundary still defaults to
reflection, so the two exchanging compartments remain confined by the sample
box.

### Stimulated-echo PGSE (PGSTE)

`run_pgste_walkers` splits the diffusion encoding across three 90-degree pulses:
``90 - G(delta) - 90 - [storage] - 90 - G(delta) - echo``. The second pulse
stores one quadrature along the longitudinal axis, so during the storage
interval the encoded magnetization decays with `T1` rather than `T2`. This is
the standard way to reach long diffusion times in short-`T2` samples (porous
media, low field, internal gradients). A spoiler gradient applied during storage
crushes the residual transverse coherences, and the workflow records an
effective selected-pathway phase table while suppressing equilibrium regrowth
into a contaminating FID. The surviving stimulated echo carries **half** the
spin-echo amplitude:

```python
from spin_dynamics.workflows import run_pgste_walkers
import numpy as np

axis = np.linspace(-1.0e-3, 1.0e-3, 64)          # wide slab (see note below)
rho = np.ones((axis.size, 2))

ste = run_pgste_walkers(
    rho=rho,
    x_axis=axis,
    z_axis=np.array([-1e-6, 1e-6]),
    gradient_amplitude=0.1,
    gradient_duration=1.0e-3,        # delta
    diffusion_time=60.0e-3,          # Delta = lobe leading-edge separation
    diffusion_coefficient=2.3e-9,
    t1_seconds=0.4,                  # storage decays with T1, not T2
    t2_seconds=15.0e-3,
    walkers_per_cell=64,
    seed=2026,
    jitter=True,
)
ste.phase_cycle.name  # "pgste_stimulated_echo"
# attenuation E(b) ~ 0.5 * exp(-Ts/T1) * exp(-b D)
```

`diffusion_time` is the leading-edge lobe separation (the Stejskal-Tanner
`b = (gamma G delta)^2 (Delta - delta/3)` still applies), and the storage
interval is `Delta - delta - 2*encode_delay - 2*excitation_duration`. Use a
sample wide compared with the gradient phase wavelength `pi / (gamma G delta)`
so the unwanted stimulated anti-echo dephases spatially; with diffusion present
it is additionally suppressed. See the PGSTE stimulated-echo example.

### Double diffusion encoding (DDE / double-PGSE)

`run_dde_walkers` applies two refocused PGSE blocks separated by a mixing time,
``90 - [G1 block] - mixing - [G2 block] - echo``, with the blocks encoding along
``angle1`` and ``angle2``. Sweeping the relative angle ``psi = angle2 - angle1``
probes microscopic anisotropy: in a restricted anisotropic pore the echo gains a
``cos 2*psi`` modulation whose amplitude reports the local anisotropy. Because it
depends only on the relative angle, it survives powder averaging and so reveals
shape anisotropy even when the single-PGSE diffusion tensor is macroscopically
isotropic. Pair it with `make_elliptical_reflector` for an anisotropic pore:

```python
from spin_dynamics.motion import make_elliptical_reflector
from spin_dynamics.workflows import run_dde_walkers
import numpy as np

semi_axes = (8.0e-6, 3.0e-6)
x = np.linspace(-semi_axes[0], semi_axes[0], 15)
z = np.linspace(-semi_axes[1], semi_axes[1], 15)
xx, zz = np.meshgrid(x, z, indexing="ij")
rho = ((xx / semi_axes[0])**2 + (zz / semi_axes[1])**2 <= 1.0).astype(float)

dde = run_dde_walkers(
    rho=rho, x_axis=x, z_axis=z,
    gradient_amplitude=1.0, gradient_duration=1.0e-3,
    diffusion_time=12.0e-3, mixing_time=1.0e-3,
    angle1=0.0, angle2=np.pi / 2,          # vary to trace E(psi)
    diffusion_coefficient=2.0e-9,
    boundary=make_elliptical_reflector((0.0, 0.0), semi_axes),
    walkers_per_cell=64, substeps_per_interval=10, seed=2026, jitter=True,
)
```

The reported `b_value` is per block, and the `cos 2*psi` powder term is a
higher-order (`q^4`) effect, so strong diffusion weighting is needed to resolve
it. See the elliptical-pore DDE example.

### Oscillating-gradient spin echo (OGSE)

`run_ogse_walkers` replaces the two rectangular PGSE lobes with cosine-modulated
gradient waveforms of `num_periods` whole periods around the refocusing pulse,
``90 - cos lobe - 180 - cos lobe - echo``. The encoding power sits at the angular
frequency `omega = 2*pi*oscillation_frequency`, so sweeping the frequency maps
the diffusion spectrum `D(omega)` and reaches the short-diffusion-time regime
that ordinary PGSE cannot. The waveform is discretized into
`samples_per_period` constant-gradient steps, and the b-value is computed from
the effective gradient spectrum (it falls steeply with frequency,
`b = (gamma G / omega)^2 * N / f`).

```python
from spin_dynamics.motion import make_motion_field_maps_2d
from spin_dynamics.workflows import run_ogse_walkers
import numpy as np

x = np.linspace(-2.5e-6, 2.5e-6, 15)        # 5 um reflecting slab along x
z = np.array([-0.5e-6, 0.5e-6])
rho = np.ones((x.size, z.size))

ogse = run_ogse_walkers(
    rho=rho, x_axis=x, z_axis=z,
    fields=make_motion_field_maps_2d(x, z),  # walls = the slab
    gradient_amplitude=0.5, oscillation_frequency=200.0,  # sweep this
    num_periods=2, diffusion_coefficient=2.0e-9,
    walkers_per_cell=160, substeps_per_interval=6, seed=2026, jitter=True,
)
d_app = -np.log(abs(ogse.signal[0]) / rho.sum()) / ogse.b_value  # D(omega)
```

In restricted geometry `D_app` rises from the long-time value toward the bulk
value as the frequency increases. Keep the per-substep hop small compared with
the pore size. See the OGSE frequency-diffusion example.

## WURST Inversion and CPMG

```python
from spin_dynamics.workflows import (
    run_ideal_wurst_inversion,
    run_matched_wurst_cpmg,
    run_matched_wurst_inversion,
)

ideal = run_ideal_wurst_inversion(numpts=101)
matched = run_matched_wurst_inversion(numpts=51, num_steps=64)
echoes = run_matched_wurst_cpmg(
    num_echoes=2,
    numpts=51,
    num_steps=64,
    rephase_action="ignore",
)
```

`run_ideal_wurst_inversion` is a fast reference path that propagates the WURST
amplitude envelope and integrated chirp phase directly on an ideal RF channel.
`run_matched_wurst_inversion` routes the same pulse through the matched-probe
transmit response, returning the demodulated coil current and receiver transfer
function along with the final magnetization. `run_matched_wurst_cpmg` uses the
matched WURST pulse as the excitation block before a finite rectangular
matched-probe CPMG train and returns received spectra, direct-summed echoes,
and echo integrals.

The WURST pulse parameters use explicit physical units where possible:
`duration_seconds`, `sweep_width_normalized` relative to nominal `w1`,
`num_steps`, envelope `order`, and peak `amplitude`. The matched CPMG workflow
also accepts the finite-grid rephasing controls `rephase_action`,
`rephase_safety_factor`, and `auto_refine_grid`.

MATLAB references:

- `Wurst_Inversion/create_WURST.m`
- `Wurst_Inversion/calc_macq_matched_probe_WURST.m`
- `Wurst_Inversion/sim_inv_matched_probe_WURST*.m`
- `circuit_simulation/matched_probe/find_coil_current_WURST.m`
- `calc_masy/calc_masy_matched_probe_WURST.m`

## Moving-Isochromat Sequences

```python
from spin_dynamics.motion import initialize_ensemble_from_density, make_motion_field_maps_2d
from spin_dynamics.sequences.motion import run_motion_cpmg_sequence, run_motion_udd_sequence
import numpy as np

fields = make_motion_field_maps_2d([-1, 1], [-1, 1])
ensemble = initialize_ensemble_from_density([[1.0]], [0.0], [0.0])
result = run_motion_cpmg_sequence(
    ensemble,
    fields,
    num_echoes=4,
    echo_spacing=0.08,
    excitation_duration=0.002,
    refocusing_duration=0.004,
    gradient=(35.0, 0.0),
    substeps_per_interval=8,
)

udd = run_motion_udd_sequence(
    ensemble,
    fields,
    num_pulses=4,
    total_duration=0.32,
    excitation_duration=0.002,
    refocusing_duration=0.004,
    gradient=(35.0, 0.0),
    t1=5.0,
    t2=1.0,
    detuning_waveform=lambda time, positions: (
        1500.0 * positions[:, 0] * np.cos(2 * np.pi * 0.35 * time)
    ),
    substeps_per_interval=8,
)
```

`spin_dynamics.sequences.motion` is the first workflow-oriented layer for
Lagrangian advection/diffusion physics. It accepts a moving `ParticleEnsemble`,
samples B0/B1 maps at the current particle positions, substeps RF and
free-precession intervals, and records receive samples during acquisition
windows. The generic `run_motion_sequence` driver takes explicit
`MotionSequenceStep` intervals; `run_motion_cpmg_sequence` builds a compact
rectangular-pulse CPMG train with one receive sample per echo, while
`run_motion_udd_sequence` places the same finite refocusing pulses at Uhrig
times and records a final UDD signal at the end of the evolution window.
Both runners accept `detuning_waveform` for time-dependent B0 fluctuations. It
may return a scalar uniform detuning or one detuning per particle, so slow
fluctuating gradients can be modeled as a callable of `(time, positions)`.

This path is not a MATLAB fixture-parity replacement for the fixed-grid
`arb10` kernels. It is intended for physical motion studies where spins move
through field maps, such as advection through inside-out B1 profiles or
diffusion-driven CPMG attenuation in static gradients.

Deterministic flow uses the same interface: pass a velocity callback that
returns one velocity vector per particle. `plot_cpmg_pipe_flow.py` demonstrates
an axisymmetric cylindrical pipe with Poiseuille flow, upstream residence-time
prepolarization in a static magnet, and downstream CPMG detection by separate
transmit/receive coil maps. Sweeping the mean velocity exposes both incomplete
polarization and motion-induced echo dephasing.

## CPMG Imaging

```python
import numpy as np

from spin_dynamics.workflows import (
    fit_imaging_echo_decay,
    form_imaging_image,
    load_imaging_field_maps_npz,
    make_imaging_field_maps,
    run_ideal_phase_encoded_cpmg_imaging,
    run_t1_encoded_phase_encoded_cpmg_imaging,
    run_tuned_phase_encoded_cpmg_imaging,
)

rho = np.eye(4)
result = run_ideal_phase_encoded_cpmg_imaging(
    rho,
    num_echoes=2,
    ny=7,
    phase_workers=2,
)
tuned = run_tuned_phase_encoded_cpmg_imaging(rho, num_echoes=1, ny=5)

field_maps = make_imaging_field_maps(
    rho,
    b0_map=np.zeros_like(rho),
    b1_tx_map=np.ones_like(rho),
    b1_rx_map=np.ones_like(rho),
)
custom = run_tuned_phase_encoded_cpmg_imaging(
    field_maps,
    num_echoes=4,
    ny=5,
    receive_mode="weighted",
)
rho_weighted = form_imaging_image(custom, mode="single", echo_index=0)
t2_weighted = form_imaging_image(custom, mode="echo_sum")
fit = fit_imaging_echo_decay(custom)
t1_prepared = run_t1_encoded_phase_encoded_cpmg_imaging(
    field_maps,
    inversion_time_seconds=0.5e-3,
    num_echoes=4,
    ny=5,
)
t1_t2_weighted = form_imaging_image(t1_prepared, mode="echo_sum")
from_npz = run_tuned_phase_encoded_cpmg_imaging(load_imaging_field_maps_npz("field_maps.npz"))
```

`run_ideal_phase_encoded_cpmg_imaging`,
`run_tuned_phase_encoded_cpmg_imaging`, and
`run_matched_phase_encoded_cpmg_imaging` return imaging result containers with:

- `rho`, `t1_map`, `t2_map`: sample maps;
- `b0_map`, `b1_tx_map`, `b1_rx_map`: off-resonance, transmit, and receive
  field maps used by the run;
- `kspace`: complex phase-encoded echo integrals with shape
  `(px, pz, num_echoes)`;
- `image` and `magnitude`: reconstructed image arrays from each echo;
- `gradx`, `gradz`: phase-encoding gradient steps;
- `del_w`: flattened normalized offset grid;
- `sequence_time`: echo-center times in seconds.

The raw reconstruction stack is one image per echo. Image formation is a
separate post-processing step:

- `form_imaging_image(result, mode="single", echo_index=0)` returns one
  reconstructed echo image. Echo 1 is the closest rho-weighted display, while
  later echoes include stronger T2 attenuation.
- `form_imaging_image(result, mode="echo_sum")` sums echo magnitudes. This can
  improve SNR, but produces a rho-plus-T2-weighted image of the form
  `sum_n A_n exp(-t_n / T2)`.
- `fit_imaging_echo_decay(result)` fits each voxel magnitude to
  `rho_app * exp(-t / T2)` and returns apparent `rho_map` and `t2_map`
  arrays. `rho_app` includes B1, receive, and probe scaling unless those are
  separately calibrated out.
- When an imaging result was generated with `noise=...`, use
  `form_imaging_image(..., use_noisy=True)` or
  `fit_imaging_echo_decay(..., use_noisy=True)` to process the noisy image
  stack. The default remains the deterministic stack for backwards-compatible
  fixture parity.
- `summarize_imaging_noise_trials([...])` compares repeated noisy imaging
  results in image space and reports the mean noisy image, per-pixel noise
  standard deviation, background noise RMS, signal mean, and image SNR.

`run_t1_encoded_phase_encoded_cpmg_imaging` adds an ideal 180-degree inversion
pulse and an inversion delay before the usual phase encoding and CPMG train.
The approximate preparation factor is
`rho * (1 - 2 exp(-TI / T1))` before the excitation pulse, so the same image
formation modes can produce T1-weighted selected-echo images, T1-plus-T2
weighted echo sums, or fitted apparent-rho/T2 maps from a T1-prepared echo
stack. The shorter `run_t1_encoded_cpmg_imaging` name is a compatibility alias.
Probe-shaped inversion preparation for tuned or matched receive models is not
part of this ideal T1-prep helper.

Each runner accepts either a spin-density array or an `ImagingFieldMaps`
container. `make_imaging_field_maps` validates arbitrary two-dimensional
`rho`, `t1_map`, `t2_map`, normalized `b0_map`, relative `b1_tx_map`,
`b1_rx_map`, `del_wx`, and `del_wz` arrays. Missing relaxation maps default to
5 ms, missing B0 defaults to zero off-resonance, and missing B1 maps use the
legacy single-sided synthetic sensitivity map. `load_imaging_field_maps_npz`
loads the same fields from a NumPy archive using keys such as `rho`, `b0_map`,
`b1_tx_map`, and `b1_rx_map`.

Scalar `b1_tx_map` and `b1_rx_map` inputs are interpreted as already-transverse
relative sensitivities. Archives or constructors may instead provide
`b0_vector_map` plus `b1_tx_vector_map` and/or `b1_rx_vector_map`, each with
shape `(..., 3)`. In that case the scalar B1 maps are computed as the magnitude
of the B1 component perpendicular to the local B0 direction. This is the
preferred path for field exports that contain laboratory-frame vector fields.

Imaging workflows and
`ImagingFieldMaps.kernel_maps(..., density_normalization="legacy")` preserve
the MATLAB-parity convention where each auxiliary off-resonance sample receives
the full voxel density. Use `density_normalization="preserve"` for physical
map-to-isochromat conversion: voxel density is divided over the auxiliary
samples so the total represented spin density remains equal to `sum(rho)`.

`b1_tx_map` scales the pulse field used by the isochromat kernels. The matched
probe imaging path applies `b1_rx_map` through its receiver calculation. The
tuned compact imaging path defaults to `receive_mode="raw"`, which preserves
the MATLAB fixture's raw-current display convention and ignores receive-map
weighting in the final k-space composition. Use `receive_mode="weighted"` to
compose tuned k-space from the receiver-filtered spectra, including the tuned
transfer function and `b1_rx_map`.

The shorter names `run_ideal_cpmg_imaging`, `run_tuned_cpmg_imaging`, and
`run_matched_cpmg_imaging` remain compatibility aliases for these phase-encoded
workflows. New imaging code should include the encoding mode in the function
name so later frequency-encoded or hybrid acquisitions can be added without
overloading "imaging".

All three imaging runners are checked against compact MATLAB-generated k-space
fixtures in `validation/fixtures`.

### Frequency-encoded imaging: spin-warp and RARE

The phase-encoded workflows above fill k-space one point per phase-encode step.
`run_spin_warp_imaging` and `run_rare_imaging` add a *readout* (frequency-encode)
gradient applied during acquisition, so each spin echo samples a whole k-space
line. They are built on the Lagrangian motion engine (one static isochromat per
voxel), which supports the two-axis gradient waveforms that the scalar-gradient
arbitrary-pulse kernels cannot express; they are ideal-probe (no probe transfer
function) and reuse `reconstruct_image_from_kspace`.

```python
import numpy as np
from spin_dynamics.workflows import run_spin_warp_imaging, run_rare_imaging

rho = np.zeros((32, 32)); rho[8:24, 10:18] = 1.0
t2 = np.full((32, 32), 60e-3)

# Spin-warp: one spin echo per phase-encode line (pz excitations, no blurring).
ref = run_spin_warp_imaging(rho, fov=(0.02, 0.02), t2_map=t2)

# RARE / fast spin echo: each echo reads a different line, so an echo train of
# length 8 needs only ceil(pz / 8) excitations.
fse = run_rare_imaging(rho, fov=(0.02, 0.02), t2_map=t2, echo_train_length=8)

image = ref.image[:, :, 0]          # complex reconstruction
print(fse.num_shots, fse.line_echo_time.max())
```

Readout is along x (frequency encode) and the phase encode is along z. Each echo
is gradient-balanced (an x pre-dephase and rewind return k-space to the origin
before each 180), so the train stays a clean Meiboom-Gill CPMG. The T2 decay
across the echo train weights the phase-encode lines (`line_echo_time` records
the echo time of each k_z line), which broadens the point-spread function -- the
characteristic RARE blurring, strongest for short T2. `echo_train_length=1`
recovers the spin-warp reference. See the RARE imaging example.

Both workflows accept the same inputs as the phase-encoded path so inhomogeneity
can be evaluated: pass `rho` plus per-voxel `b0_map` (absolute angular
off-resonance, rad/s), `b1_tx_map`, `b1_rx_map`, `t1_map`, and `t2_map`, or pass
an `ImagingFieldMaps` container directly (in which case the map keywords must be
omitted). B0 inhomogeneity produces geometric distortion along the readout axis
(`1 / readout gradient`); B1 maps shade the image and perturb the flip angles.
An unresolved *sub-voxel* B0 spread is modelled with `num_offsets > 1`
isochromats per voxel evenly spaced over `+/- offset_spread` (rad/s) -- the
counterpart of the phase-encoded path's `ny` / `maxoffs` off-resonance samples.
Because a spin echo refocuses the static spread at each echo, the spread blurs
the image along the readout axis (the T2* point-spread function) without decaying
the echo train. See the imaging-inhomogeneity example.

A related diagnostic is `imaging_slice_sensitivity`, which maps the real-space
*sensitive slice* of an excitation in a non-uniform field. The excited
transverse magnetization is computed from a rectangular RF pulse for every voxel
at its own off-resonance (`b0_map - center_frequency`, rad/s) and transmit-B1,
so the slice profile (bandwidth ~ 1/`excitation_duration`) and the curvature
(set by the B0 contours) emerge directly; the result is weighted by the receive
B1, and `refocusing=True` also applies the 180-degree refocusing efficiency (the
spin-echo sensitive volume). The returned `sensitivity` follows the curved
iso-B0 contours and is shaded by B1 -- it is "neither flat nor uniform", the
practical reality of imaging in inhomogeneous fields. It accepts the same `rho`
array plus maps, or an `ImagingFieldMaps` container. See the sensitive-slice
example.

### Slice-selective excitation and 3D multi-slice imaging

`imaging_slice_sensitivity` is passive -- a hard pulse reading out the existing
inhomogeneity. Real slice selection plays a *shaped* RF while a gradient is on,
localizing a plane even in a uniform field. `make_slice_selective_excitation`
builds this pulse (a windowed-sinc RF train carrying the slice gradient plus a
rephasing lobe) as motion-engine steps, and `simulate_slice_profile` returns the
through-slice profile -- a sharply bounded band whose thickness scales as the
excitation bandwidth over the gradient.

`run_multislice_imaging` is the **true-3D** workflow. A single 3D walker ensemble
lives in a 3D `MotionFieldMaps` carrying the actual `(B0, B1)` field; each slice
is excited (carrier offset to position it) and read out (spin-warp: readout on
one in-plane axis, phase encode on the other) through the engine, filling that
slice's 2D k-space. Because the slice is selected by *total* off-resonance
(slice gradient plus local B0), a nonuniform B0 curves and displaces the excited
slice and warps the readout -- the real-magnet behavior, not a flat-slice
cartoon.

```python
import numpy as np
from spin_dynamics.workflows import run_multislice_imaging, simulate_slice_profile

profile = simulate_slice_profile(duration=1e-3, slice_gradient=1.5e7)

rho = np.zeros((16, 5, 16)); rho[6, 2, 9] = 1.0
volume = run_multislice_imaging(
    rho, slice_gradient=1.5e7, fov=(0.02, 0.02, 0.02),
    b0_map=b0_volume, b1_tx_map=b1_volume, b1_rx_map=b1_volume,  # 3D, shape of rho
)
print(volume.magnitude.shape)   # (nx, n_slices, nz)
```

`run_multislice_imaging_separable` is a fast approximation: it reduces the slice
pulse to a 1D through-plane weight `w(y)` and forms each slice with the validated
2D spin-warp workflow on the `w`-weighted density. It ignores in-plane field
variation (the slice stays flat), trading that fidelity for speed on large
volumes -- the engine path costs roughly `n_slices x n_phase_encode` full-ensemble
runs. Genuinely Fourier-encoded 3D (slab-select plus a second phase encode with
`ifftn`) is left for later. The example `plot_multislice_halbach_imaging.py`
acquires a structured 3D phantom in a mild Halbach `(B0, B1)` saddle and shows the
acquired slices and the 3D reconstruction.

All three paths -- the slice pulse, the motion engine, and the phase-encoded
kernels -- read the same per-voxel physics through the dimension-agnostic
`spin_dynamics.fields` layer (`SpatialDomain`, `SpatialFieldMaps`), so the 1D, 2D,
and 3D cases share one field representation and one gradient-coupling rule
(`del_w_local = del_w_static + sum_d g_d * r_d`).

Noise-aware workflows can pass `NoiseSpec(domain="time")` to CPMG echo
workflows to add white noise directly to the time-domain echo samples instead
of the received spectrum. Probe-colored noise remains spectrum-domain because
it is defined from receiver output noise density. Noise metadata records the
requested model/domain, clean signal RMS, realized noise RMS, and realized SNR.

Use `spin_dynamics.noise.estimate_matched_filter_snr` to compare repeated noisy
spectra against the analytical matched-filter SNR predicted by a probe
`pnoise` density. Pass the same offset axis used for the received spectrum as
`sample_axis` when generating probe noise and as `offsets` when estimating SNR
so the discrete noise variance matches the numerical matched-filter integral.

`phase_workers` parallelizes independent phase-encoding points. `num_workers`
is passed through to the chunked isochromat backend inside each acquisition.
The plotting example `examples/plot_ideal_imaging.py` loads the MATLAB
`flower.png` phantom and saves a three-panel phantom/k-space/reconstruction
image. Use `--probe ideal`, `--probe tuned`, or `--probe matched` to choose the
acquisition model.

The default flower phantom is a bright-background bitmap, so
`examples/plot_ideal_imaging.py` inverts it before using it as the spin-density
map. This makes the compact demo behave like a bright object in a darker field
of view, which is less prone to bright-background aliasing when the FOV is
small. Use `--raw-image` to preserve the source bitmap contrast.

MATLAB references:

- `Imaging_demo/imaging_example_ideal.m`
- `Sim_CPMG/sim_cpmg_ideal_probe_img.m`
- `Sim_CPMG/sim_cpmg_tuned_probe_img.m`
- `Sim_CPMG/sim_cpmg_matched_probe_img.m`
- `create_fields/create_fields_single_sided.m`

## Magnet Field Sources and Single-Sided NMR

The imaging and motion workflows consume `(B0, B1)` maps; `spin_dynamics.fields.magnetostatics`
*generates* them from first-principles magnet and coil models (pure NumPy, mesh-free), so a
real device field can drive the spin dynamics instead of a synthetic field.

- **B0** is the closed-form field of uniformly magnetized rectangular bars (the magnetic
  charge-sheets on their faces), exact for the nearly linear rare-earth magnets used in
  single-sided NMR. A soft-iron **return yoke** is added by the method of images across a
  `mu -> infinity` plane, which enforces `B_tangential = 0` at the iron surface.
- **B1** is the Biot-Savart field of a coil (`circular_loop` segments); its transverse
  (imaging-relevant) component is the part perpendicular to the local B0.
- **Finite Halbach dipoles** use the lowest-order four-rod approximation:
  `halbach_dipole_magnets` builds four diametrically magnetized cylindrical or square
  rods, and `sample_halbach_dipole_field` samples the 3D bore/fringe field by summing a
  finite-volume point-dipole cubature. This is intended for quick bore-field and
  end-effect studies, not precision magnet design inside the magnet material.

```python
import numpy as np
from spin_dynamics.fields.magnetostatics import (
    nmr_mouse_magnets, circular_loop, sample_magnet_field,
)

bars, yoke = nmr_mouse_magnets(gap=0.012, remanence=1.30)   # two antiparallel bars on a yoke
coil = circular_loop((0.0, 0.030, 0.0), 0.008, axis="y")
x = np.linspace(-0.02, 0.02, 121); y = np.linspace(0.021, 0.045, 121)
fm = sample_magnet_field(x, y, bars, yoke_y=yoke, coil_segments=coil)
# fm.b0_magnitude (T), fm.b0_gradient (T/m, the static gradient), fm.larmor_hz, fm.b1_transverse
```

For the canonical NMR-MOUSE (two antiparallel bars on a yoke) this reproduces the device's
hallmarks: `|B0| ~ 0.25-0.54 T` over the gap, a proton Larmor frequency of `~5-23 MHz`, and a
strong static gradient `G ~ 7-28 T/m`. The example `plot_nmr_mouse_fields.py` shows the field,
the gradient, the coil B1, and the depth-resolved sensitive slice from `imaging_slice_sensitivity`.

### Depth-resolved relaxation and diffusion

Single-sided NMR profiles a sample by depth: an excitation frequency selects the iso-B0
sensitive slice, and the strong static gradient encodes diffusion. The defining feature is
that the spins *move through a spatially structured field* -- as a molecule diffuses it
samples a changing off-resonance set by the real gradient -- which is irreducibly spatial and
cannot be reduced to a fixed off-resonance distribution. `spin_dynamics.workflows.single_sided`
therefore drives the moving-isochromat engine directly with the magnet's own B0 map.

```python
from spin_dynamics.fields.magnetostatics import nmr_mouse_magnets
from spin_dynamics.workflows.single_sided import (
    SampleLayer, LayeredSample, mouse_depth_profile, measure_diffusion_at_depth,
)

bars, yoke = nmr_mouse_magnets(gap=0.012, remanence=1.30)
sample = LayeredSample([
    SampleLayer(0.022, 0.030, rho=1.0, t2=0.060, diffusion=2.3e-9),  # water
    SampleLayer(0.030, 0.034, rho=0.0),                              # gap
    SampleLayer(0.034, 0.044, rho=1.0, t2=0.015, diffusion=0.5e-9),  # gel
])
profile = mouse_depth_profile(bars, sample, frequencies_hz, yoke_y=yoke)  # signal/T2 vs depth
d = measure_diffusion_at_depth(bars, sample, frequency_hz, yoke_y=yoke)    # D at one depth
```

`mouse_depth_profile` sweeps the carrier frequency: the excited signal traces the depth
profile (a `rho = 0` gap shows up as a hole) and the echo decay gives the apparent T2, which is
diffusion-shortened where the gradient is strong. `measure_diffusion_at_depth` runs the CPMG
twice with identical initial walker positions -- diffusion on and off -- so the messy
inhomogeneous-field echo envelope cancels in the ratio, leaving the diffusion attenuation in
the real gradient; the rate `k = (1/12) gamma^2 G^2 D tE^2` then gives D using the slice's local
gradient. Because this is a moving-walker Monte-Carlo it is stochastic (average several seeds),
and the diffusion sensitivity honestly falls off with depth as the gradient weakens. The
moving-walker engine is validated against the exact constant-gradient Carr-Purcell law in
`tests/test_single_sided.py`, so deviations here are the real-field physics, not numerics. The
example `plot_nmr_mouse_depth_profile.py` profiles a layered phantom end to end.

### Finite Halbach dipole field maps

The same magnetostatics layer can generate the finite-length Halbach field used by
pre-polarizing magnets and compact imaging magnets:

```python
import numpy as np
from spin_dynamics.fields.magnetostatics import sample_halbach_dipole_field

x = y = np.linspace(-14e-3, 14e-3, 31)
z = np.linspace(-50e-3, 50e-3, 41)
fm = sample_halbach_dipole_field(
    x, y, z,
    center_radius=30e-3,
    rod_shape="square",
    rod_width=16e-3,
    length=80e-3,
    remanence=1.30,
)
```

The result contains `b0_vector`, `b0_magnitude`, `b0_gradient`, and `larmor_hz`
on the full `(x, y, z)` grid. `examples/plot_halbach_dipole_field.py` plots the
mid-plane field, field uniformity, transverse centerline, and finite-length axial
falloff.

## Ideal FID

```python
from spin_dynamics.parameters import set_params_ideal_fid
from spin_dynamics.workflows.fid import sim_fid_ideal

sp, pp = set_params_ideal_fid(numpts=101)
macq, fid, tvect = sim_fid_ideal(sp, pp)
```

MATLAB references:

- `Params/set_params_ideal_FID.m`
- `calc_macq/calc_macq_fid.m`
- `Sim_FID/simFID_ideal.m`
- `calc_FID_decay/calc_FID_time_domain.m`

## Ideal Finite Acquisition

```python
from spin_dynamics.workflows import (
    calc_macq_ideal_probe_relax4,
    calc_macq_matched_probe_relax4,
    calc_macq_tuned_probe_relax4,
    calc_macq_untuned_probe_relax4,
)

macq = calc_macq_ideal_probe_relax4(sp, pp)
macq, mrx = calc_macq_tuned_probe_relax4(sp, pp)
```

This lower-level workflow mirrors MATLAB
`calc_macq/calc_macq_ideal_probe_relax4.m`. It accepts a fully assembled
arbitrary sequence with precomputed pulse matrices in `pp.Rtot`, returns one
acquired spectrum per acquisition segment, and uses relaxation during
free-precession intervals. The tuned and matched wrappers mirror
`calc_macq_tuned_probe_relax4.m` and `calc_macq_matched_probe_relax4.m` by
applying receiver transfer functions after acquisition. The untuned wrapper is
the Python analogue using the same receiver-map contract.

MATLAB references:

- `calc_macq/calc_macq_ideal_probe_relax4.m`
- `calc_macq/calc_macq_tuned_probe_relax4.m`
- `calc_macq/calc_macq_matched_probe_relax4.m`
- `sim_spin_dynamics_arb/sim_spin_dynamics_arb9.m`

## Tuned-Probe CPMG

```python
from dataclasses import replace

import numpy as np

from spin_dynamics.parameters import set_params_tuned_orig
from spin_dynamics.probes.tuned import calc_masy_tuned_probe_lp_orig

params, sp, pp = set_params_tuned_orig(numpts=21)
sp = replace(sp, del_w=np.linspace(-5, 5, 21), plt_tx=0, plt_rx=0)
mrx, masy, snr = calc_masy_tuned_probe_lp_orig(params, sp, pp)
```

MATLAB references:

- `Params/set_params_tuned_Orig.m`
- `circuit_simulation/tuned_probe/tuned_probe_lp_Orig.m`
- `circuit_simulation/tuned_probe/tuned_probe_rx.m`
- `calc_rot/calc_rot_axis_tuned_probe_lp_Orig2.m`
- `calc_masy/calc_masy_tuned_probe_lp_Orig.m`

## Untuned-Probe CPMG

```python
from dataclasses import replace

import numpy as np

from spin_dynamics.parameters import set_params_untuned_orig
from spin_dynamics.probes.untuned import calc_masy_untuned_probe_lp

params, sp, pp = set_params_untuned_orig(numpts=21)
sp = replace(sp, del_w=np.linspace(-5, 5, 21), plt_tx=0, plt_rx=0)
mrx, masy, snr = calc_masy_untuned_probe_lp(params, sp, pp)
```

MATLAB references:

- `Params/set_params_untuned_Orig.m`
- `circuit_simulation/untuned_probe/untuned_probe_lp.m`
- `circuit_simulation/untuned_probe/untuned_probe_rx.m`
- `calc_rot/calc_rot_axis_untuned_probe_lp.m`
- `calc_masy/calc_masy_untuned_probe_lp.m`

## Matched-Probe CPMG

```python
from dataclasses import replace

import numpy as np

from spin_dynamics.parameters import set_params_matched_orig
from spin_dynamics.probes.matched import calc_masy_matched_probe_orig

sp, pp = set_params_matched_orig(numpts=11)
sp = replace(sp, del_w=np.linspace(-4, 4, 11), plt_tx=0, plt_rx=0)
mrx, masy, snr = calc_masy_matched_probe_orig(sp, pp)
```

MATLAB references:

- `Params/set_params_matched_Orig.m`
- `circuit_simulation/matched_probe/matching_network_design2.m`
- `circuit_simulation/matched_probe/find_coil_current.m`
- `circuit_simulation/matched_probe/matched_probe_rx.m`
- `calc_rot/calc_rot_axis_matched_probe.m`
- `calc_masy/calc_masy_matched_probe_Orig.m`

The matched-probe Python workflow uses a NumPy-only nonlinear solve and
fixed-step RK4 response calculation. It is validated against MATLAB fixtures,
but small differences from MATLAB's optimization and ODE solver stack are
expected at tighter-than-practical tolerances.
