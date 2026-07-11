# Unified Experiment Workflow — Architecture Analysis and Plan

Status: facade milestones 1--7 implemented; the backend-neutral sequence IR
follow-on is underway. Companion planning docs: `nqr_module_plan.md`,
`non_inductive_detection.md`, `optimal_control_hardware_response.md`.

## 1. Motivation

PythonSpinDynamics has grown feature-first: each new capability (probes, NQR,
ESR, imaging, diffusion, GRAPE, field solvers, detectors, RFI cancellation)
arrived as a well-tested vertical slice, but there is no horizontal layer that
tells a user *how the pieces compose*. The proposed remedy is a default
workflow of the form:

> define geometry → define transmitters/receivers → define sample → define
> experiment → predict runtime → solve fields (if needed) → run → view/analyze
> → save,

where each step offers pre-built options or user plug-ins, and the system
warns about incompatible combinations. This document analyzes the current
architecture, refines that concept, and lays out feasible implementation
paths.

## 2. Architecture as it stands

### 2.1 Six engine families, no apex abstractions

| Engine | Sample object | Entry points | Results |
|---|---|---|---|
| Classical Bloch/isochromat (`workflows/`, `core/`) | arrays + `CoupledSpinSystem` | ~50 stable + ~100 extended `run_*` functions | per-module frozen dataclasses (`CPMGResult`, `PGSEWalkerResult`, …) |
| Reduced NQR (`nqr/simulation.py`) | `QuadrupolarSite` | `simulate_slse`, `simulate_sorc`, … | `SLSEResult`, `SORCResult`, … |
| Full density-matrix NQR (`nqr/full_dynamics.py`) | `QuadrupolarSite` (shared) | `simulate_full_fid/echo/slse` | `FullNQR*Result` |
| ESR (`esr/`) | `ESRSpinSystem` | per-submodule (DEER, ESEEM, …) | per-submodule |
| GRAPE (`optimal_control/`) | `ControlHamiltonianModel` | `grape_optimize*` | `GrapeOptimizationResult` |
| Nonresonant field reversal (`nonresonant/`) | `NonresonantFieldModel` | `simulate_field_reversal_echoes` | `FieldReversalResult` |

Concepts (sample, sequence, detection, result) are re-invented per engine.
There is no shared base class for any of them; the only cross-engine reuse is
`QuadrupolarSite` (reduced ↔ full NQR ↔ GRAPE NQR builder).

### 2.2 The workflow layer is convention-unified, not code-unified

The ~150 `run_*` functions follow strong *conventions*:

- signature: `run_<probe>_<sequence>(numpts, maxoffs, ..., *, num_workers, noise, absolute_phase, ...)`;
- frozen-dataclass results carrying `del_w`, `mrx`, `echo`, `tvect`, plus metadata (`NoiseMetadata`, `PhaseCycle`, `AbsolutePhaseMetadata`);
- shared spec objects for cross-cutting physics (`NoiseSpec`, `RadiationDampingSpec`, `AbsolutePhaseSpec`, `PhaseCycle`);
- shared guards (`check_rephasing` with warn/raise actions, grid auto-refinement).

But each workflow re-implements a ~13-step procedural assembly (parameter
constructors → grid → phase cycle → pulse shapes → rotation matrices →
sequence encoding → kernel call → echo reconstruction → noise → result), ~50–200
LoC each, with the probe variants (ideal/tuned/untuned/matched) largely
duplicated. There is no registry, base class, or declarative description of an
experiment. `STABLE_WORKFLOW_API` / `EXTENDED_WORKFLOW_API` tuples in
`workflows/__init__.py` are the closest thing to a curated catalog.

### 2.3 Hardware chain: good pieces, manual glue

- **Geometry** — `fields/coils.py` generators emit a unified `list[Segment]`
  format; magnet arrays, shields, spirals, Maxwell pairs.
- **Field solvers** — Biot-Savart B1, charge-sheet/dipole B0,
  (non)linear magnetostatics, quasistatic eddy solvers. Output: bare arrays.
- **Field-map container** — `SpatialFieldMaps` (`fields/maps.py`) is the
  canonical hardware→simulation interface (domain, rho, T1/T2, B0, B1_tx,
  B1_rx, diffusion), consumed by imaging (Eulerian `flatten()`) and motion
  (Lagrangian `sample()`).
- **RX** — clean `Detector` ABC (field-referred `H(f)` + `N²(f)`), inductive/
  SQUID/OPM implementations, gradiometer pickups that reuse Biot-Savart via
  reciprocity.
- **TX** — fragmented across three orthogonal stacks that never meet:
  spatial B1 maps (amplitude shading only; imaging defaults to a *synthetic
  Gaussian* if none supplied), circuit-level probe models (`probes/tuned.py`
  etc., consume circuit + pulse params, never field maps), and command-level
  LTI envelope filters for GRAPE (`optimal_control/control_response.py`).

Every arrow in geometry → solver → `SpatialFieldMaps` → workflow is manual:
the user calls the solver, projects the transverse component
(`transverse_b1_magnitude`), builds the container, passes it in. Only two
chains are already wired end-to-end: `EddyModes → GradientDriverResponse`
(pre-emphasis) and `Gradiometer → detected SNR`.

### 2.4 Compatibility checking exists — but engine-locally

The "warn about incompatible options" idea is already half-built, scattered:

- `nqr/model_selection.py::select_nqr_model()` — reduced vs. full engine
  recommendation with machine-readable reasons;
- RWA fundamental-line guard (`nqr/full_dynamics.py`, also called from the
  GRAPE NQR builder);
- `check_rephasing` (offset-grid adequacy, warn/raise/auto-refine);
- EFG-distribution rephasing check, weak-Zeeman perturbation guard;
- spin-value constraints in `QuadrupolarSite.__post_init__` and
  `ESRSpinSystem.__post_init__`;
- ValueError guards inside workflows (e.g. "radiation damping is currently
  wired for tuned/matched probes").

What is missing is a *front-loaded* validation pass: today most guards fire
mid-run or not at all (a user can happily pass a `SQUIDMagnetometer` where the
workflow ignores it, or pair `simulate_slse` with a spin-5/2 site and get a
late error).

### 2.5 User surface

- 163 example scripts, mostly following the same implicit workflow
  (args → params → run → plot → optional `--save-npz`). No hierarchy.
- Docs are feature-by-feature (`docs/python_api/*.md`, ~6.5k lines) plus a
  LaTeX user manual; no workflow-first narrative or tutorial.
- No CLI entry points, no GUI, no notebooks, no config-file experiment
  definitions, no runtime estimator. Saving is ad-hoc NPZ. (Flask experience
  exists in the workspace via NQRDatabase's explorer app.)

## 3. Assessment of the proposed concept

The proposed pipeline is the right *mental model* — it is literally the shape
of all 145 examples — but three refinements are needed before it becomes an
API:

**(a) Declarative spec, not a wizard.** A rigid step-order (geometry first,
sample third…) fights the reality that most experiments skip geometry/field
solving entirely, and that the sample+experiment choice determines which
engine (and therefore which options) exist. The unifying object should be a
declarative `Experiment` description whose fields can be filled in any order,
with a single `plan()` step that resolves engine choice, validates
compatibility, and estimates cost *before* `run()`. "Define geometry → … →
save" then becomes the documented default *narrative* over that spec, not an
enforced state machine.

**(b) Facade over engines, not a new engine.** The `run_*` functions are
validated against MATLAB/Octave fixtures and are the package's scientific
anchor. A unified layer must delegate to them (a registry mapping spec →
existing callable), never re-implement dynamics. Rewriting toward apex base
classes across six engines would put ~23k LoC and the fixture parity at risk
for zero physics gain.

**(c) Compatibility knowledge as data, not scattered code.** Generalize the
existing guards into a rule set the planner runs: each registered workflow
declares what it accepts (probe types, spin ranges, detectors honored,
physics options wired), and each rule can return ok/warn/error with the same
"reasons" style `select_nqr_model` already uses. This is also exactly the
metadata a GUI or doc generator needs.

Runtime prediction is feasible and cheap: kernel cost is near-linear in
`numpts × num_pulse_segments × num_echoes` (isochromat) or
`(2I+1)² × steps` (density matrix), and `benchmarks/` already maintains
commit-indexed golden timings per backend to calibrate the constants. A
`plan()` that reports "engine=full-NQR, backend=numba, est. 40 s, memory
~1.2 GB, warnings: [...]" covers the user need without promising exactness.

## 4. Refined concept

```python
from spin_dynamics import experiment as ex

study = ex.Experiment(
    sample=ex.Sample(sites=[QuadrupolarSite(...)], t1=..., t2=..., geometry=phantom),
    hardware=ex.Hardware(                      # every field optional
        b0=ex.B0FromMagnetArray(...),          # or ex.UniformB0(2.0)
        tx=ex.TxCoil(solenoid(...), probe="tuned", circuit=...),
        rx=ex.RxChain(detector=SQUIDMagnetometer(...), pickup=gradiometer),
    ),
    sequence=ex.CPMGTrain(num_echoes=8, ...),  # thin spec mapping to run_* kwargs
    acquisition=ex.Acquisition(noise=NoiseSpec(...), phase_cycle=...),
)

plan = study.plan()      # engine + backend selection, all guards, runtime/memory estimate
print(plan.report())     # human-readable; machine-readable .warnings / .errors / .estimate
result = study.run()     # solves fields lazily if hardware demands it, then delegates
result.plot()            # engine-appropriate default figures
result.save("run1.npz")  # arrays + JSON provenance (spec, versions, git sha, timings)
study2 = ex.load("run1.npz").experiment      # round-trip: re-run / tweak
```

Key properties:

1. **Additive.** All existing `run_*`/`simulate_*` functions remain the
   supported low-level API; the facade is sugar plus safety, and its
   registry entries point at them.
2. **Plan/run split.** `plan()` is the home for compatibility checking and
   runtime prediction; `run()` refuses only on `errors`, not `warnings`.
3. **Lazy field solving.** `B0FromMagnetArray`/`TxCoil` carry geometry; the
   planner inserts a field-solve stage and builds `SpatialFieldMaps`
   automatically (closing today's manual glue), with caching keyed on the
   geometry hash so repeated runs skip the solve.
4. **Serializable spec.** Specs are frozen dataclasses → JSON/TOML
   round-trip. This gives config-file-driven runs, provenance inside saved
   results, and the substrate any GUI/CLI needs — for free.
5. **Extension points.** `ex.register_sequence(...)`, custom `Detector`
   subclasses, custom field maps: "pick from pre-existing options or write
   your own" becomes registering into the same catalog the built-ins use.

## 5. Feasible paths forward

**Path A — documentation-only unification (1–2 sessions).** Write the
workflow chapter (manual + `docs/python_api/getting_started.md`) presenting
the 8-step narrative over existing APIs; restructure the examples index by
workflow stage. Zero code risk; does not deliver validation, runtime
prediction, or serialization. Worth doing regardless, but insufficient alone.

**Path B — incremental facade (recommended).** Build
`spin_dynamics.experiment` as in §4, in PR-sized milestones (§6). Moderate
effort, additive, each milestone independently useful, existing API and
fixtures untouched.

**Path C — deep re-architecture.** Apex `Sample`/`Sequence`/`Detector`/
`Result` base classes and migration of all engines. Rejected: months of work,
high regression risk against MATLAB parity fixtures, and the audience
(scientists scripting in Python) gains little over Path B's facade.

**GUI.** Defer until the spec layer exists; then it is a thin client.
Escalation ladder: (1) CLI `python -m spin_dynamics.experiment run exp.toml`
— nearly free once specs serialize; (2) Jupyter `ipywidgets` builder that
edits a spec and shows `plan.report()` live; (3) small local web app
(NiceGUI/Streamlit, or Flask as already used in NQRDatabase) with dropdowns
populated from the registry and compatibility greying-out driven by the rule
set. A hand-built Qt/desktop GUI is not warranted.

## 6. Milestone plan (Path B)

| PR | Content | Notes |
|---|---|---|
| 1 ✅ | `experiment/` package: spec dataclasses, registry, `run()` delegation for the CPMG family (ideal/tuned/untuned/matched, train + IR); `Result.save/load` with JSON provenance | proves the shape end-to-end |
| 2 ✅ | `plan()`: rule engine (`rules.py`) with a front-loaded rephasing pre-check + noise-spec validation, structured `ExperimentPlan.findings`, `plan.report()`. Probe×option compatibility is already covered by PR-1's `honors` "ignored-field" warnings; detector-honored checks defer to PR-4 when detectors enter the facade. `max_time` formulas extracted into shared workflow helpers so plan- and run-time verdicts agree | front-loads today's scattered warnings |
| 3 ✅ | Runtime/memory estimator (`estimate.py`, `ExperimentPlan.estimate`): per-workflow work-unit cost models, `seconds = a + b·units`. Calibrated once per process from two sub-second on-host ideal-train dry runs (per active kernel backend) rather than from the `benchmarks/` host constants — more accurate on the user's machine and self-updating. `plan(estimate=False)` / `set_calibration()` to opt out or pin | advisory, order-of-magnitude accuracy |
| 4 ✅ | Hardware wiring (`hardware.py` + `wiring.py`): `TxCoil`/`RxCoil`/`UniformB0`/`ImagingPlane` specs (SI meters) + `Phantom` + `CPMGImaging` sequence → automatic Biot-Savart solve onto the phantom grid at plan time, transverse-B1 projection, rho-weighted-mean normalization, geometry-hash caching; transmit-efficiency diagnostic warns on coils mostly parallel to B0. Scope notes: targets the 2-D `ImagingFieldMaps` container (the imaging workflows' interface) rather than `SpatialFieldMaps`; magnet-array B0 solves and non-inductive detectors (RxChain) defer to PR-5+ | closes the biggest manual-glue gap (§2.3) |
| 5 ✅ | NQR adapter (`nqr_adapter.py`): `Sample.site` + `NQRSLSE`/`NQRSORC` specs; reduced-vs-full dispatch via `select_nqr_model` inside `plan()` (recommendation + reasons surfaced as findings; forced overrides warn; spin-1 engine constraint errors). The adapter owns the effective↔bare nutation conversion (`bare = eff/(2·strength)`) and `transition="auto"` picks the line most coupled to the drive polarization. ESR adapters split into PR-5a | reuses the existing selector verbatim |
| 5a ✅ | ESR adapter (`esr_adapter.py`): `Sample.esr_system` + `ESRFID`/`ESRHahnEcho` specs over the pulsed core; `UniformB0.field_tesla` fixes the electron Larmor frequency (plan error when missing); 90-180 Hahn defaults; fixed the previously-raising `ESRSpinSystem.__eq__`. | pulsed core |
| 5b ✅ | Measurement-level spectroscopy facade: full-model `NQRFID`, reduced spin-1 `NQRPopulationTransfer`, `ESRCWSweep`, and `ESRDEER`; DEER distance distributions serialize with the experiment and its array result is wrapped for normal NPZ provenance. | closes primary 1-D NQR/ESR gaps |
| 5c ✅ | Electron-nuclear correlation facade: two-/three-pulse ESEEM with analytic/full selection, 2-D HYSCORE time and frequency planes, and Davies/Mims ENDOR over serialized `HyperfineCoupling`; HYSCORE planning includes grid/FFT cost. | completes ESEEM/HYSCORE/ENDOR coverage |
| 6 ✅ | Docs (Path A): getting-started guide (`docs/python_api/experiment_workflow.md`) + user-manual chapter "Unified Experiment Workflow" (both the 8-step narrative); docs index lists the facade as the recommended entry point; examples index leads with a facade section. Three flagship facade examples (`experiment_facade_quickstart.py`, `experiment_imaging_with_coil.py`, `experiment_nqr_auto_model.py`) in the example smoke tests | Path A folded in here |
| 7 ✅ | CLI runner (`python -m spin_dynamics.experiment` with plan/run/show/convert) over a human-friendly `type`-tagged TOML/JSON config (`experiment/config.py` + `cli.py`, dependency-free TOML writer + `tomllib` reader); shipped `examples/experiment_config_cpmg.toml`. Round-trips every engine family incl. 2-D phantoms + nested coils | free by-product of the spec |
| 8 (opt) | Notebook/web GUI prototype on the registry | only after 1–7 prove out |
| 9 ✅ | Reproducible-result provenance: canonical experiment/result SHA-256 identities, resolved callable/module hashes, Git revision/dirty state, numerical environment/build/thread capture, randomness classification, archive integrity properties, and rerun verification in Python and the CLI; reads legacy v1 archives. | distinguishes reproducibility from physical validation |
| 10 ✅ | General `SequenceIRExecution` facade target: explicit 1--3-D `SequenceDomain`, native/Pulseq compilation, gradient-axis mapping, RF/gradient moving-isochromat execution, ADC demodulation, white noise, planning/cost checks, persistence, and exact rerun support. Probe-required policies fail closed pending a probe-aware target. | makes the standard interchange IR executable without inventing a second sequence format |

Each PR keeps the pre-submit gates (ruff + regenerated `api_reference.md`) and
adds fixture-style tests asserting the facade reproduces the direct `run_*`
call bit-for-bit.

## 7. Risks and non-goals

- **Framework-itis.** The facade must stay optional and thin; if a spec field
  just forwards a kwarg, it should look like that kwarg. Success metric: the
  flagship examples get *shorter*, not longer.
- **Two APIs drifting.** Registry entries should be defined next to the
  workflows they wrap, and a test should assert every `STABLE_WORKFLOW_API`
  entry has a registry mapping (or an explicit exclusion), so new workflows
  can't silently bypass the catalog.
- **GRAPE and analysis stay adjacent.** Pulse optimization consumes/produces
  experiment pieces (a sequence spec in, an optimized waveform out) but is a
  design-time loop, not a ninth pipeline stage; same for ILT/fitting, which
  operate on `Result`. Model them as tools that accept/return spec objects.
- **Not a goal:** unifying the six engines' internals, changing any numerics,
  or per-package PyPI packaging (release model stays the unified workspace).
