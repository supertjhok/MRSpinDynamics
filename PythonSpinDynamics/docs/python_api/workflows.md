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

## Ideal Time-Varying-Field CPMG

```python
from spin_dynamics.workflows import (
    run_ideal_time_varying_amplitude_sweep,
    sinusoidal_field_waveform,
)

waveform = sinusoidal_field_waveform(num_echoes=16)
result = run_ideal_time_varying_amplitude_sweep(
    amplitudes=[0, 0.5, 1.0, 2.0],
    waveform=waveform,
    numpts=101,
)
```

`run_ideal_time_varying_cpmg_final` returns the final echo for a supplied
per-echo normalized B0 offset waveform. The amplitude sweep wrapper returns
`IdealTimeVaryingSweepResult` with echoes, echo integrals, and matched-filter
signals versus fluctuation amplitude. Field offsets use MATLAB's normalized
`w_0t = gamma * B_0t / w_1n` convention.

MATLAB references:

- `time_varying_field/sim_cpmg_ideal_tv_final.m`
- `time_varying_field/compare_cpmg_results_ideal_tv.m`
- `time_varying_field/compare_cpmg_results_ideal_v0crit.m`

## Matched CPMG-IR Finite Train

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

`run_matched_cpmg_ir_train` returns `MatchedCPMGIRTrainResult` with:

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
transient solver can become stiff.

MATLAB references:

- `DIffusion_Example/Diff_Echo_Q.m`
- `Sim_Diffusion/sim_dif_matched_CPMG_noRx.m`
- `calc_macq_diff/calc_macq_matched_probe_relax_diff_noRx.m`
- `sim_spin_dynamics_arb/sim_spin_dynamics_arb_relax_diff.m`

## CPMG Imaging

```python
import numpy as np

from spin_dynamics.workflows import run_ideal_cpmg_imaging, run_tuned_cpmg_imaging

rho = np.eye(4)
result = run_ideal_cpmg_imaging(
    rho,
    num_echoes=2,
    ny=7,
    phase_workers=2,
)
tuned = run_tuned_cpmg_imaging(rho, num_echoes=1, ny=5)
```

`run_ideal_cpmg_imaging`, `run_tuned_cpmg_imaging`, and
`run_matched_cpmg_imaging` return imaging result containers with:

- `rho`, `t1_map`, `t2_map`: sample maps;
- `kspace`: complex phase-encoded echo integrals with shape
  `(px, pz, num_echoes)`;
- `image` and `magnitude`: reconstructed image arrays from each echo;
- `gradx`, `gradz`: phase-encoding gradient steps;
- `del_w`: flattened normalized offset grid;
- `sequence_time`: echo-center times in seconds.

All three imaging runners are checked against compact MATLAB-generated k-space
fixtures in `validation/fixtures`.

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
