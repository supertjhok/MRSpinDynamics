# Workflows

Workflow helpers live under `spin_dynamics.workflows`.

Application code should prefer the public workflow runners below. The
lower-level examples remain useful for validation, porting, and debugging
individual MATLAB reference paths.

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
    auto_refine_grid=True,
    sweep_workers=2,
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

MATLAB references:

- `DIffusion_Example/Diff_Echo_Q.m`
- `Sim_Diffusion/sim_dif_matched_CPMG_noRx.m`
- `calc_macq_diff/calc_macq_matched_probe_relax_diff_noRx.m`
- `sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m`

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
from spin_dynamics.sequences.motion import run_motion_cpmg_sequence

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
```

`spin_dynamics.sequences.motion` is the first workflow-oriented layer for
Lagrangian advection/diffusion physics. It accepts a moving `ParticleEnsemble`,
samples B0/B1 maps at the current particle positions, substeps RF and
free-precession intervals, and records receive samples during acquisition
windows. The generic `run_motion_sequence` driver takes explicit
`MotionSequenceStep` intervals; `run_motion_cpmg_sequence` builds a compact
rectangular-pulse CPMG train with one receive sample per echo.

This path is not a MATLAB fixture-parity replacement for the fixed-grid
`arb10` kernels. It is intended for physical motion studies where spins move
through field maps, such as advection through inside-out B1 profiles or
diffusion-driven CPMG attenuation in static gradients.

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
