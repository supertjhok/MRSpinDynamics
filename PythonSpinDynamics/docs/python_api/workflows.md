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
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
)

tuned_q = run_tuned_q_sweep(q_values=[20, 50, 80], numpts=101)
matched_detune = run_matched_mistuning_sweep(offsets=[-2, 0, 2], numpts=101)
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

MATLAB references:

- `CompareQ/sim_tuned_probe_coil_Q.m`
- `CompareQ/sim_matched_probe_coil_Q.m`
- `CompareMistuned/tuned_probe/sim_tuned_probe_mistuned.m`
- `CompareMistuned/matched_probe/sim_matched_probe_mistuned.m`

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
