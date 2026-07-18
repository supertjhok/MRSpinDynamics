# Python API Documentation

This is the capability map for PythonSpinDynamics. Use it when you know the
scientific area but not yet the relevant guide, workflow, or module. For a
first simulation, follow [How to Use This Documentation](../documentation_map.md)
and the [Unified Experiment Workflow](experiment_workflow.md) instead of
reading the API pages in directory order.

The package began as a Python port of the active MATLAB Version 3 workflows,
but it now also contains substantial Python-native physics, field, hardware,
and analysis layers. The MATLAB tree remains a parity reference for the
original workflows; it is not the organizing principle for new user code.

## Choose an entry point

### Run and configure experiments

- [Installation](installation.md)
- [Concepts and Units](concepts.md)
- [Unified Experiment Workflow](experiment_workflow.md) — recommended entry point
- [Examples](examples.md)
- [Workflows](workflows.md)

### Choose a physical model

- [J-Coupling Models](j_coupling.md)
- [Hyperpolarization and Singlet States](hyperpolarization.md)
- [NQR Models](nqr.md)
- [ESR Models](esr.md)
- [Defect-Spin Nanoscale MR](nano_mr.md)
- [Chemical / Site Exchange](exchange.md)
- [Internal / Susceptibility Gradients](internal_gradients.md)
- [Bipolar 13-Interval PGSTE](bipolar_pgste.md)

### Build sequences and systems

- [Phase Cycling Findings](phase_cycling.md)
- [Sequence IR and Pulseq Import](sequence_ir.md)
- [Cross-Component Composition](composition.md)
- [Image-Guided Magnetic Therapy](../image_guided_magnetic_therapy.md)
- [Analysis](analysis.md)

### Look up and validate

- [Parameters](parameters.md)
- [Core Numerical Functions](core.md)
- [API Reference](api_reference.md)
- [Performance](performance.md)
- [Validation](validation.md)
- [Known Gaps](known_gaps.md)
- [Deprecation Policy](deprecation_policy.md)
- [Development Environment](../development_environment.md)

## Current Supported Surface

The validated Python API currently covers:

- a unified declarative experiment facade (`spin_dynamics.experiment`) over the
  CPMG family, phase-encoded imaging, pulsed NQR (SLSE/SORC), and pulsed ESR
  (FID/Hahn echo): declarative specs, a `plan()` stage with compatibility rules
  and runtime/memory estimation, automatic coil-geometry-to-B1 field wiring for
  imaging, reduced-vs-full NQR engine dispatch, and NPZ save/load with JSON
  provenance and spec round-trip (see
  [Unified Experiment Workflow](experiment_workflow.md));
- ideal CPMG asymptotic magnetization and echo construction;
- public ideal, tuned, untuned, and matched CPMG runners returning a common
  `CPMGResult`;
- public finite ideal CPMG acquisition returning `CPMGTrainResult`;
- finite CPMG train rephasing checks, optional grid refinement, and chunked
  multicore isochromat propagation;
- ideal, tuned, untuned, and matched CPMG inversion-recovery finite trains over
  `tauvect`;
- Python-native finite-train Q/mistuning sweeps for tuned, untuned, and
  matched probes;
- first matched-probe diffusion CPMG workflow and compact diffusion Q sweep;
- rectangular PGSE and PGSE-prepared CPMG workflows with deterministic
  gradient-moment and explicit random-walker backends;
- fixture-validated ideal, tuned, and matched CPMG imaging, k-space
  reconstruction, and arbitrary B0/B1 field-map loading helpers;
- frequency-encoded (spin-warp and RARE) imaging, slice-selective excitation
  with gradient-shaped pulses, and true-3D slice-selective multi-slice imaging in
  spatially varying `(B0, B1)` fields (`run_multislice_imaging`), with a fast
  separable approximation, built on a dimension-agnostic 1D/2D/3D field-map layer
  (`spin_dynamics.fields`: `SpatialDomain`, `SpatialFieldMaps`) shared by the
  imaging and moving-isochromat workflows;
- analytic magnet field sources (`spin_dynamics.fields.magnetostatics`):
  closed-form B0 of permanent-magnet bars with a soft-iron return yoke (method of
  images), Biot-Savart coil B1, finite four-rod Halbach dipole field maps, and a
  single-sided NMR-MOUSE depth-resolved relaxation/diffusion simulation
  (`spin_dynamics.workflows.single_sided`) that diffuses walkers through the
  magnet's real static gradient, validated against the exact constant-gradient
  Carr-Purcell law;
- fixture-validated pulse-shape utilities for JMR rectangular pulse responses,
  phase quantization, and untuned segment adjustment;
- tuned and matched CPMG Q/mistuning sweep workflows;
- matched-probe z-magnetization Q sweep workflow;
- ideal, tuned, untuned, and matched time-varying-field CPMG final-echo and
  amplitude-sweep workflows;
- WURST pulse construction, matched-probe WURST transmit response, ideal and
  matched WURST inversion, and matched WURST-CPMG workflows;
- ideal FID acquisition and time-domain trace construction;
- 1D and separable 2D inverse Laplace analysis helpers for T1, T2, T1-T2, and
  D-T2 kernels with manual or SNR-selected Tikhonov regularization;
- opt-in moving-isochromat motion helpers for B0/B1 field-map sampling,
  deterministic advection, seeded diffusion, and receive summation;
- prepolarization helpers that convert polarizer field strength, residence
  time, and T1 into sequence-ready `m0`/`mth` arrays, plus BPP-style scalar
  relaxation helpers for temperature-dependent `T1` and `T2`;
- ideal-probe finite acquisition with relaxation through
  `calc_macq_ideal_probe_relax4`;
- SPA refocusing pulse catalog, normalized SNR/FOM metric bookkeeping,
  tuned/untuned/matched fixed-refocusing evaluators, lightweight discrete
  phase-search scaffold, ideal v0crit, excitation-aware v0crit, and
  time-varying-field refocusing evaluation/search, bounded refocusing phase
  optimizers, and tuned excitation-pulse evaluation/phase search for supplied
  refocusing axes, diagnostic tuned inverse-excitation search for target
  spectra, with optional SciPy continuous optimization backend;
- array-returning multi-start optimization driver scaffolds for repeated
  ideal/tuned/untuned/matched refocusing, tuned excitation, and phase-flipped
  tuned inverse-excitation searches;
- plotting-free optimization pipeline handoff from selected refocusing result
  to tuned excitation and inverse-excitation searches;
- MATLAB-style optimization result-cell conversion, `.npz` archive loading and
  export, script-aware `plot_opt_*_results.m` layout analysis, selected
  score/program/metadata inspection, tuned original/inverse result-pair
  comparison, and optional SciPy-backed `.mat` import/export for multi-start
  results;
- plotting examples for CPMG comparisons, finite trains, parameter sweeps,
  diffusion, PGSE D-T2 inverse Laplace, time-varying fields, imaging, motion,
  WURST, inverse Laplace, and compact optimization workflows;
- low-level rotation matrix and effective-axis helpers;
- first pulsed NQR helpers for spin-1 quadrupolar sites, selective
  transition pulses, single-crystal and powder orientations, SLSE echo trains,
  perturbation-plus-detection population-transfer experiments, weak-B0 spectra,
  polarization-enhanced NQR transport, and CIF-based proton-coupling estimates;
- first ESR helpers for single-electron spin-1/2 systems, scalar/anisotropic
  `g` tensors, single-crystal and powder orientation grids, and fixed-field or
  fixed-frequency spectra, CW derivative/lineshape display, static
  strain/disorder sampling, rectangular-pulse FID and Hahn-echo helpers, and
  Liouville-space pulsed T1/T2 relaxation, plus isotropic electron-nuclear
  hyperfine doublet simulations;
- defect-spin nano-MR foundations (`spin_dynamics.nano_mr`): explicit
  laboratory/defect/surface frames, arbitrary-spin ZFS-plus-Zeeman
  Hamiltonians, ODMR transition analysis, planar sensor geometry,
  point-dipole electron-nuclear tensors, diamond NV-minus and 4H-SiC PL6
  presets, phase-aware dynamical-decoupling control, ideal and finite-pulse
  filter functions, addressed-qubit propagation, and optical photon-count
  readout, plus statistical/thermal/fixed-polarization nuclear baths,
  analytic layers and half-spaces, voxel densities, multi-isotope spectra,
  Gaussian filter-overlap coherence, exact resolved clusters and 2-D
  correlation spectroscopy, and seeded diffusing/confined dipolar field
  records with correlations and spectra, plus raster/arbitrary/array scan
  geometries, coherent-field and statistical-variance dipolar operators,
  analytic depth profiling, nonnegative density reconstruction, sparse
  point-source localization, and local uncertainty estimates, plus correlated
  target/sensor field processes, time-resolved optical cycling, SPAD detector
  effects, shot records, covariance-weighted reconstruction, clocked Qdyne,
  synchronized I/Q readout, independent coherence budgets, sensor-memory
  correlation, coherent thermal chemical-shift/J spectra, and optional
  explicit DNP preparation, plus unified statistical-spectrum/Qdyne specs,
  nested TOML/result archives, and adaptive Qdyne design;
- the current `sim_spin_dynamics_arb10` kernel;
- the legacy-compatible `sim_spin_dynamics_arb7` path needed by ideal FID;
- original/reference tuned, untuned, and matched probe CPMG paths.

Full historical MATLAB `.mat` file parity, strong inverse excitation
cancellation parity, and broad `fmincon` parity validation are still
reference-only beyond the compact optimization fixtures.
