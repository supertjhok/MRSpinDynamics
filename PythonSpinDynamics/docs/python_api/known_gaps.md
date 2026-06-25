# Known Gaps

The main MATLAB-to-Python porting phase is now largely complete for the
canonical Version 2 workflows. The remaining gaps are mostly validation depth,
specialized variants, packaging polish, and performance backends rather than
missing primary workflows.

Ported and validated:

- ideal CPMG asymptotic magnetization and echo construction;
- ideal FID acquisition and time-domain trace construction;
- `sim_spin_dynamics_arb10` and the FID-compatible `arb7` path;
- tuned-probe CPMG original/reference asymptotic path;
- untuned-probe CPMG original/reference asymptotic path;
- matched-probe CPMG original/reference asymptotic path;
- ideal-probe finite acquisition with relaxation through
  `calc_macq_ideal_probe_relax4`;
- tuned, untuned, and matched finite-acquisition receiver wrappers;
- isochromat-grid rephasing analysis, warning/raise behavior, and optional
  finite-train grid refinement;
- chunked multicore isochromat propagation for the `arb10` finite-acquisition
  path;
- public finite ideal CPMG echo-train workflow;
- public finite tuned-probe CPMG echo-train workflow;
- public finite untuned- and matched-probe CPMG echo-train workflows;
- ideal, tuned, untuned, and matched CPMG inversion-recovery finite trains over
  `tauvect`;
- tuned and matched Q/mistuning sweep workflows;
- Python-native finite-train Q/mistuning sweeps for tuned, untuned, and
  matched probes;
- first matched-probe diffusion CPMG workflow and compact diffusion Q sweep;
- ideal PGSE and PGSE-prepared CPMG workflows with a deterministic
  gradient-moment backend, explicit random-walker backend, Stejskal-Tanner
  validation, and a PGSE D-T2 inverse-Laplace plotting example;
- fixture-validated ideal, tuned, and matched CPMG imaging, k-space
  reconstruction, arbitrary B0/B1 field-map loading helpers, and tuned
  receive-weighted imaging mode;
- ideal inversion-recovery T1-prepared phase-encoded CPMG imaging with
  selected-echo, echo-summed, fitted-rho, and fitted-T2 image formation;
- moving-isochromat sequence driver primitives, including explicit sequence
  intervals, RF/free-precession substeps, receive samples, and a rectangular
  CPMG runner for static-gradient diffusion/advection studies;
- 1D and separable 2D inverse Laplace transform helpers for T1, T2, T1-T2,
  D-T2, and T2-T2 synthetic analysis, with adjustable Tikhonov regularization
  and SNR-informed automatic strength selection plus SciPy-backed non-negative
  solves;
- Bloch-McConnell site/chemical exchange (`spin_dynamics.exchange`):
  multi-site kinetic generators with detailed-balance checks, transverse
  lineshape coalescence, longitudinal mixing propagators, and encode-mix-detect
  T2-T2 relaxation exchange (REXSY) data that inverts to an exchange map;
- fixture-validated pulse-shape utilities for JMR rectangular pulse responses,
  phase quantization, and untuned segment adjustment;
- WURST pulse construction, matched-probe frequency-swept transmit response,
  ideal WURST inversion, matched WURST inversion, and matched WURST-CPMG
  workflows;
- SPA refocusing pulse catalog and normalized SNR/FOM metric bookkeeping;
- tuned, untuned, and matched fixed-refocusing-pulse SPA/OCT evaluation wrappers;
- tuned, untuned, and matched SPA summary workflows over rectangular and
  catalog pulses;
- lightweight discrete phase-program search scaffold for optimizer experiments;
- ideal no-probe v0crit, excitation-aware v0crit, and time-varying-field
  refocusing evaluation, bounded optimizers, and multi-start drivers;
- tuned, untuned, and matched bounded refocusing phase optimizer wrappers
  around the fixed-amplitude SNR evaluators;
- tuned excitation-pulse evaluation and bounded phase optimizer wrappers for
  supplied refocusing axes;
- tuned inverse-excitation evaluation and bounded phase optimizer wrappers for
  target received spectra, with compact MATLAB optimizer-result and residual
  spectrum validation;
- array-returning multi-start optimization driver scaffolds for repeated
  refocusing, tuned excitation, and phase-flipped tuned inverse-excitation
  searches;
- plotting-free optimization pipeline handoff from selected refocusing result
  to tuned excitation and inverse-excitation searches;
- MATLAB-style optimization result-cell conversion plus `.npz` archive
  load/export, script-aware `plot_opt_*_results.m` layout analysis,
  selected score/program/metadata inspection, tuned original/inverse
  result-pair comparison, compact tuned/untuned/matched refocusing result
  fixtures, and optional SciPy-backed `.mat` import/export;
- optional SciPy continuous bounded optimizer backend for phase optimization,
  with NumPy pattern search as the minimal-install fallback;
- matched-probe z-magnetization Q sweep workflow;
- ideal, tuned, untuned, and matched time-varying-field CPMG final-echo and
  amplitude-sweep workflows;
- public CPMG workflow runners returning `CPMGResult`.
- initial single-electron ESR helpers with scalar/anisotropic `g` tensors,
  dense Zeeman diagonalization, orientation grids, frequency spectra, and field
  sweeps, Gaussian/Lorentzian derivative display, diagonal `g` strain and
  applied-field offset distributions, plus rectangular-pulse FID and Hahn-echo
  simulations with Liouville-space T1/T2 relaxation, and isotropic
  electron-nuclear hyperfine doublet simulations.

Remaining gaps:

- newer tuned-probe helper variants outside the original/reference and JMR
  rectangular-pulse paths;
- newer untuned-probe helper variants outside the original/reference and JMR
  rectangular-pulse paths;
- newer matched-probe helper variants outside the original/reference and JMR
  rectangular-pulse paths;
- probe-shaped T1-prepared imaging for tuned or matched inversion pulses;
- general phase cycling is only partially first-class. The new
  `spin_dynamics.phase_cycling` table owns the default CPMG two-step branch
  combination and records PGSTE selected-pathway metadata, but arbitrary
  cycle-table support is not yet wired through all workflows, and NQR/ESR
  pathway selection remains workflow-specific. See
  [Phase Cycling Findings](phase_cycling.md);
- broad diffusion sweeps, Q>2000 matched-diffusion validation, stimulated-echo
  or bipolar PGSE variants, restricted/anisotropic diffusion models, and
  probe-shaped PGSE pulses;
- full moving-isochromat imaging workflows with phase/frequency encoding,
  probe-shaped pulses, and direct reconstruction outputs;
- exact MATLAB WURST fixture parity beyond finite-output and physical sanity
  tests, because the MATLAB WURST scripts are exploratory and include
  placeholder or plotting-oriented branches;
- full historical MATLAB `.mat` result-file parity, broad `fmincon` parity
  validation, and strong inverse excitation cancellation parity;
- compiled or GPU acceleration backends;
- received-signal noise-model validation: the probe-noise variance carries a
  `/ dx**2` rescaling onto a user-supplied `sample_axis` (a no-op for the
  default unit grid) whose convention is not independently validated for
  physical sample spacings. The matched-probe amplifier noise-figure term uses
  the available-power `k*T*Rin*(F-1)` basis, consistent with the matched coil
  term (the factor-of-4 difference from the open-circuit `4*k*T*R` form is the
  matched 1/2-voltage / 1/4-power transfer); absolute SNR magnitudes still
  benefit from validation against a measured noise figure;
- multi-pulse ESR beyond the first rectangular-pulse FID/Hahn-echo helpers,
  anisotropic hyperfine, exchange/dipolar couplings, higher-spin zero-field
  splitting, temperature-dependent equilibrium magnetization, and ESR
  saturation or resonator models;
- stable public packaging and autogenerated API reference docs.

The next natural work is therefore a stabilization phase: broader PGSE and
constant-gradient diffusion validation, stronger inverse-excitation
cancellation workflows, exact historical MATLAB result-file parity where those
files are still authoritative, and packaging/API-reference polish.
