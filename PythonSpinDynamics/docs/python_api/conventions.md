# Conventions Profiles

Read this page when results must be physically consistent rather than match
the MATLAB reference implementation, or when you need to know why two runs
of the same experiment differ in scale.

Several defaults reproduce MATLAB/Octave results even though a physically
consistent alternative exists. Each alternative is still a per-call argument,
and `spin_dynamics.conventions` selects them together:

| Setting | `"matlab"` profile (default) | `"physics"` profile | Effect of the physical choice |
|---|---|---|---|
| `relax_during_pulses` | `False`: pulses act instantly | `True`: Strang-split T1/T2 relaxation around every arb10 pulse | Removes the `TE / (TE - T180)` inflation of fitted T2 (1.333 → 1.000 for the default ideal train) |
| `echo_registration` | `"matlab"` layout | `"exact"` DFT time grid | Removes the MATLAB layout's echo-integral deficit (about 9% at `numpts = 101`) and its time-axis stretch of about `1/n` |
| `imaging_signal_convention` | `"matlab"`: `mrx1 - mrx2` | `"cpmg"`: `(mrx1 - mrx2) / 2` | Imaging k-space on the CPMG-train scale (×½; tuned imagers unaffected) |
| `imaging_density_normalization` | `"legacy"`: each auxiliary offset sample carries the full voxel density | `"preserve"`: density divided by `ny` | Total represented spin density conserved |

```python
from spin_dynamics.conventions import conventions, get_conventions, set_conventions
from spin_dynamics.workflows import run_ideal_cpmg_train

with conventions("physics"):
    physical = run_ideal_cpmg_train(num_echoes=8, t2_seconds=0.05, auto_refine_grid=True)

set_conventions({"echo_registration": "exact"})  # one setting, others unchanged
print(get_conventions())
set_conventions("matlab")
```

An explicit argument always wins over the profile: `calc_time_domain_echo(...,
registration="matlab")` keeps the MATLAB layout inside a `"physics"` block,
and a `relax_during_pulses` key in kernel parameters overrides the profile.
`spin_dynamics.core.kernels.set_arb10_pulse_relaxation` and the profile share
one setting, so either reports the other's change.

The setting is process-global, like `set_arb10_backend`, so it also reaches
worker threads (`num_workers`, `tau_workers`). The context manager is not safe
for concurrent callers that want different conventions.

## Recorded with every facade run

`Experiment.run()` records the active conventions in the run's provenance
(`record.provenance["conventions"]`), and `LoadedRun.verify_reproduction()`
reruns under the recorded profile, so an archive made under `"physics"`
reproduces even when the default profile is active. The facade's
`CPMGImaging.signal_convention` and `density_normalization` fields default to
`None`, which follows the profile.

## Not part of a profile

- **CPMG-IR combine.** `run_*_cpmg_ir_train(signal_convention=...)` already
  defaults to the physical `"cpmg"` combine; `"matlab"` gives the relax4 sign
  (−2× CPMG).
- **M+/M− detection.** `PhaseCycle.combine(detection=...)` describes the data
  being combined (the arb10 kernels acquire M−), not a modeling choice, so a
  global switch would be wrong for data from other sources.
- **Tuned-probe imaging receive mode** (`receive_mode="raw"` or
  `"weighted"`) selects what the receiver measures rather than correcting a
  bias.

The default remains `"matlab"` so existing results, fixtures, and the Octave
parity tests are unchanged. The [validation guide](validation.md) lists the
MATLAB-parity evidence each default preserves.
