# Unified Experiment Workflow

`spin_dynamics.experiment` is the recommended entry point for new code. It is a
thin, declarative layer over the validated `run_*` / `simulate_*` workflows: you
describe an experiment with frozen-dataclass specs, front-load validation with
`plan()`, execute with `run()`, and save the result with provenance. The facade
never re-implements any dynamics — a default `Experiment` reproduces the direct
workflow call bit for bit — so it costs nothing in fidelity and adds
compatibility checking, runtime estimation, automatic hardware wiring, and
serialization on top.

If you already know which `run_*` function you want, calling it directly (see
[Workflows](workflows.md)) remains fully supported. Reach for the facade when
you want the surrounding scaffolding, a config-file-driven run, or a single
interface across the NMR, imaging, NQR, and ESR engines.

## The eight-step workflow

The facade organizes an experiment into the same stages every example follows:

1. **define the sample** — relaxation, a phantom, an NQR site, or an ESR system;
2. **define transmitters/receivers** — probe, and optionally coil geometry;
3. **define the sequence** — CPMG, imaging, SLSE, ESR FID/echo, …;
4. **define the acquisition** — offset grid, noise, rephasing policy;
5. **plan** — resolve the workflow, check compatibility, estimate cost;
6. **solve fields** — done automatically inside `plan()`/`run()` when coils are given;
7. **run** — delegate to the validated workflow and capture provenance;
8. **view / analyze / save** — inspect the native result, persist it with its spec.

## A first run

```python
from spin_dynamics.experiment import Experiment, CPMGTrain, Sample, Hardware, Acquisition

study = Experiment(
    sequence=CPMGTrain(num_echoes=8),
    sample=Sample(t1_seconds=2.0, t2_seconds=2.0),
    hardware=Hardware(probe="tuned"),
    acquisition=Acquisition(numpts=501, maxoffs=10.0),
)

plan = study.plan()
print(plan.report())
record = study.run()
record.save("run1.npz")
```

`plan.report()` prints the resolved workflow, an advisory runtime/memory
estimate, and any warnings or errors:

```
sequence: CPMGTrain
probe: tuned
workflow: run_tuned_cpmg_train
estimate: ~6 ms on 'numpy' backend (advisory); memory ~3 MB
checks: ok
```

Runnable version: [`examples/experiment_facade_quickstart.py`](../../examples/experiment_facade_quickstart.py).

## `plan()` before `run()`

`plan()` is the front-loaded validation stage. It resolves which workflow will
run and returns an `ExperimentPlan` whose `errors` and `warnings` are also
available structurally as `plan.findings` (each finding has a `rule`,
`severity`, `message`, and `details`). `run()` refuses to execute a plan with
errors; warnings are surfaced but do not block. Current rules:

- **rephasing** — checks the isochromat offset grid is fine enough for the
  sequence length *before* running, and reports the recommended `numpts`.
  Respects `Acquisition.rephase_action` (`"warn"` / `"raise"` / `"ignore"`).
- **noise_spec** — rejects malformed or unsupported noise at plan time.
- **hardware_wiring** — solves requested coil fields and reports a
  transmit-efficiency diagnostic (see below).
- **nqr_model** — runs `select_nqr_model` and reports the reduced-vs-full
  recommendation with its reasons.

Set fields the resolved workflow does not consume and `plan()` warns that they
will be ignored, so mismatched specs never fail silently.

## Automatic hardware wiring (imaging)

For imaging sequences, describe the transmit coil as geometry in SI meters and
the facade solves its Biot-Savart B1 on the phantom grid, projects it transverse
to B0, normalizes it to a sample-weighted mean of one, and passes it to the
workflow in place of the synthetic default:

```python
from spin_dynamics.experiment import (
    Experiment, CPMGImaging, Sample, Phantom, Hardware,
    TxCoil, SolenoidCoil, UniformB0, ImagingPlane,
)

study = Experiment(
    sequence=CPMGImaging(ny=9),
    sample=Sample(phantom=Phantom(rho=disc), t1_seconds=5e-3, t2_seconds=5e-3),
    hardware=Hardware(
        b0=UniformB0(direction=(0, 0, 1)),
        tx_coil=TxCoil(geometry=SolenoidCoil(radius_m=0.015, length_m=0.03,
                                             turns=10, axis="x")),
        plane=ImagingPlane(extent_m=(0.02, 0.02)),
    ),
)
```

The solve happens once at plan time (cached by a geometry hash and reused by
`run()`). `plan()` reports the transverse fraction of the coil's B1 over the
sample and warns when most of it is parallel to B0 — an inefficiency the
normalization would otherwise hide. Runnable version:
[`examples/experiment_imaging_with_coil.py`](../../examples/experiment_imaging_with_coil.py).

## Other engines

The same interface drives the NQR and ESR engines. For NQR, `plan()` picks the
reduced two-level or full density-matrix engine via `select_nqr_model` and
reports why:

```python
from spin_dynamics.experiment import Experiment, NQRSLSE, Sample
from spin_dynamics.nqr import QuadrupolarSite

study = Experiment(
    sequence=NQRSLSE(pulse_duration_seconds=100e-6, nutation_hz=2.5e3,
                     echo_spacing_seconds=1e-3, num_echoes=4),
    sample=Sample(site=QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)),
)
study.plan()   # → workflow: simulate_slse (reduced), with reasons
```

`NQRSLSE.nutation_hz` uses the reduced engine's effective two-level Rabi
convention; the adapter converts to the bare `gamma*B1/(2*pi)` the full engine
needs. Runnable version:
[`examples/experiment_nqr_auto_model.py`](../../examples/experiment_nqr_auto_model.py).

For pulsed ESR, set `Sample.esr_system` (an `ESRSpinSystem`) and a
`Hardware.b0 = UniformB0(field_tesla=...)` so the electron Larmor frequency is
fixed, then use an `ESRFID` or `ESRHahnEcho` sequence.

## Saving and reloading

`record.save(path)` writes an NPZ archive holding every array field of the
native result plus a JSON document with the full experiment spec, provenance
(package/NumPy/Python versions, platform, timestamp, elapsed time), and scalar
result fields. `load_run(path)` returns a `LoadedRun` exposing the raw arrays,
the reconstructed native result, and — via `loaded.experiment` — the original
spec, so a saved run can be re-run or tweaked:

```python
from spin_dynamics.experiment import load_run

loaded = load_run("run1.npz")
loaded.experiment.run()          # reproduce
```

Specs are JSON-serializable directly, too (`Experiment.to_json()` /
`Experiment.from_json(...)`), which is the basis for config-file-driven runs.

## Scope

The facade currently wraps the CPMG family (asymptotic, finite train, and
inversion-recovery train for ideal/tuned/untuned/matched probes), phase-encoded
CPMG imaging, pulsed NQR (SLSE and SORC), and pulsed ESR (FID and Hahn echo).
Design notes and the milestone roadmap are in
[`../unified_workflow_plan.md`](../unified_workflow_plan.md).
