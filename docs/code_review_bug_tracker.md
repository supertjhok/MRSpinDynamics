# Code-review bug tracker

Findings from the whole-repository review of commit `e25d982` (2026-11-01).
The review covered `PythonSpinDynamics`, `QuadrupolarDFT`, `NQRDatabase`,
`integration`, CI workflows and a light pass over `MATLABSpinDynamics`.
At review time ruff and pytest were clean (2381 passed, 120 skipped), so
every item below gets past the current test suite.

Keep this file current: when an item is closed, set **Status** to `Fixed`,
add the commit hash, and name the regression test in **Resolution**. Use
`Won't fix` (with a reason) or `Not a bug` for items that are closed
without a code change.

**Paths.** Unless prefixed, paths are relative to
`PythonSpinDynamics/src/spin_dynamics/`. `QDFT/` means
`QuadrupolarDFT/src/quadrupolar_dft/`, and `NQRDB/` means `NQRDatabase/`.
Line numbers refer to `e25d982`.

**Evidence.** Repro: a reviewer reproduced it by running code. Verified: the
code was independently re-read and confirmed. Reading: inferred from the
code, not run.

**Severity.**
- High: wrong results or a crash in normal use.
- Medium: edge-case wrong results, a convention trap, or drift between twins.
- Low: quality or maintainability.

## Summary

| Severity | Open | In progress | Fixed | Closed without change |
|---|---:|---:|---:|---:|
| High | 0 | 0 | 13 | 0 |
| CI / gates | 7 | 0 | 0 | 0 |
| Medium | 44 | 0 | 0 | 0 |
| Low | 51 | 0 | 1 | 0 |

## High

| ID | Location | Problem | Evidence | Status | Resolution |
|---|---|---|---|---|---|
| H-01 | `probes/tuned.py:101-145, 234-280` | Tuned-probe transient divides the first homogeneous term by 2α twice (also in `dicd1`/`dicd2`), so pulse edges and ringdown are wrong (51% early-pulse current error, 4× in the first rotating-frame window). The error was introduced in MATLAB Version_3 `tuned_probe_lp.m` / `tuned_probe_lp_Orig.m` (V1/V2 are correct); Octave-parity fixtures lock it in. | Verified + Repro (`solve_ivp`) | Fixed | Removed the extra division in both Python functions and in both MATLAB V3 files. Octave is unavailable, so 12 CSV fixtures and the scores in 3 `.mat` fixtures were regenerated from the corrected Python (documented in `validation/fixtures/README.md`; committed MATLAB optimizer phases kept). Test: `test_tuned_probe_transient.py` (DOP853 ODE reference, error 0.53 → 1.7e-11). The π-periodicity and mandal2015 echo tests now normalize by the first echo (see M-44), and the compact-MATLAB inverse test uses 16 pattern passes. Rerun the Octave generators when Octave is available. Commit `1263cee`. |
| H-02 | `core/kernels.py:513-545` | `_matrix_elements_power` uses the principal branch of `eig`; for rotations ≥ π the sub-step is not a rotation (half of a π pulse maps Mz → i·Mz). Radiation damping with `apply_during_pulses=True` mishandles every refocusing pulse. | Verified + Repro | Fixed | The kernel takes per-pulse generators (`pulse_segments`, checked against `Rtot`), and the tuned/matched CPMG trains pass them, so in-pulse feedback follows the physical path. The matrix-power fallback is now a true rotation power: it warns, and raises for the ambiguous θ ≈ π case or non-rotations. Error vs an independent RK4 Bloch+RD reference: tuned 0.195 → 6.5e-4, matched 0.230 → 3.9e-4. Tests: `test_radiation_damping.py::RadiationDampingDuringPulsesTests`, `::test_cpmg_trains_during_pulses_match_independent_bloch_reference`, `::test_tuned_cpmg_phase_kick_pulses_carry_post_rotation`. Commit `c215ed5`. |
| H-03 | `nqr/simulation.py:417-438` | `simulate_slse` with default `relaxation=None` applies only the pulse propagator and drops free precession, and it samples at a different point from the relaxing branch. Off-resonance SLSE and `simulate_slse_offset_sweep` are wrong. | Verified + Repro | Fixed | Both branches now share the `_relaxing_slse_steps` cycle (pulse + detuned free precession) and sample at the end of the cycle; the unitary branch is the T1,T2→∞ limit. Test: `test_nqr.py::test_unitary_slse_includes_off_resonance_free_precession` (independent Bloch-rotation reference). Commit `23def73`. |
| H-04 | `optimal_control/hamiltonians.py:253-262`, `optimal_control/objectives.py:125` | NQR powder GRAPE discards `OrientationSample.weight`, and the ensemble objective takes an unweighted mean, so the optimizer solves the wrong problem (0.258 vs weighted 0.185). | Verified + Repro | Fixed | New `ensemble_weights` on `make_grape_objective` / `grape_optimize*` and `robust_ensemble_fidelity(weights=)`; new `nqr_powder_control_weights(orientations)`. The example trains on and reports powder-weighted means. The default (no weights) is unchanged. Tests: `test_grape_powder_weights.py` (run with JAX: weighted 0.185 vs NumPy reference). Commit `c629a3b`. |
| H-05 | `workflows/bssfp.py:160, 356-364` | The α/2 catalyzation prep has the same RF phase as the first α pulse (3α/2 total), so catalyzation amplifies the transient instead of suppressing it. In the imaging path, `pulse_index` never advances, contrary to its comment. | Verified + Repro | Fixed | The prep uses the phase of pulse index −1 (−dphi; π for the default alternation), and the imaging path advances `pulse_index` after the prep. Tests: `test_bssfp.py::test_catalyzed_transient_matches_analytic_on_resonance` (Scheffler approach + Freeman–Hill steady state), `::test_catalyzed_train_is_smooth_at_isocentre`; the old `test_catalyzation_reaches_steady_state_sooner` encoded the bug and is replaced by `::test_catalyzation_suppresses_transient_oscillation`. Note: −dphi is exact only for dphi = π; φ ≈ 3π/2 − dphi/2 is closer for other increments (not adopted). Commit `a2f5480`. |
| H-06 | `fields/nonlinear_magnetostatics.py:826-880` | The axisymmetric A_φ operator is missing the ∂ν/∂r term, so H_z is not continuous at radial μ jumps. Ferrite-core solenoid core field: 1.58 vs 4.25 (μr = 5). The existing test only asserts "> 1.5×". | Repro | Fixed | Conservative finite-volume form in ψ = rA_φ, assembled as D·M_ψ·D (SPD, shared by the CG and LU paths), so H_z is continuous across radial μ jumps; Dirichlet rows rescaled to the interior diagonal (residual 0.4% → 1e-13). Tests: `test_nonlinear_magnetostatics.py::test_long_cored_solenoid_matches_closed_form`, `::test_ferrite_core_matches_conservative_flux_function_reference`, and the tightened `::test_high_permeability_core_boosts_axial_field`. The `plot_logging_ferrite_b1_focusing.py` B1 gain changes ×5.27 → ×11.08. Commit `5df21b0`. |
| H-07 | `fields/magnetostatics.py:99-103` | The vertical-face yoke image reverses its endpoints, which cancels `-sigma`. Images of x-magnetized bars have the wrong sign (tangential B = 0.43 T on the iron). | Verified + Repro | Fixed | The image extents are kept ordered `(2Y−b, 2Y−a)` so the `−σ` flip takes effect; br_y-only output (including the MOUSE preset) is bit-identical. Tests: `test_magnetostatics.py::test_oblique_bar_on_yoke_matches_explicit_image_bar`, `::test_yoke_image_for_nonzero_plane_height`, `::test_mouse_preset_matches_explicit_image_bars`. Commit `5df21b0`. |
| H-08 | `fields/coil_properties.py:420` | Skin-annulus area π·δ(d−δ) has no DC limit, so R_ac goes negative or non-monotonic when δ > d (R_ac = −0.13 Ω at 2 kHz). | Verified + Repro | Fixed | δ is capped at the wire radius, and the proximity/current-sharing factors are blended in as (a/δ)⁴, so R_ac → R_dc as f → 0. Results with δ ≤ a are unchanged (QOIL references bit-identical). Tests: `test_coil_properties.py::test_ac_resistance_tends_to_dc_resistance`, `::test_ac_resistance_is_monotonic_and_continuous_in_frequency`. The low-frequency blend is an interpolation, not a derived model. Commit `5df21b0`. |
| H-09 | `optimization/pipeline.py:143-197` | The excitation pipeline computes `neff` with the ideal `calc_rot_axis_arba4` instead of the tuned-probe axis with free precession (`calc_rot_axis_tuned_probe_lp_orig2`), giving `neff` errors up to 1.99. `maxoffs` is ignored on the evaluation path. | Repro | Fixed | Both the MATLAB-cell and Python-result paths compute `neff` with `calc_rot_axis_tuned_probe_lp_orig2`, with `tfp` taken from the cell padding or the tuned SPA default; a stored ideal `neff` is no longer reused. `maxoffs` now defaults to `None`, and values other than the evaluator's 10 raise. Tests: `test_optimization_pipeline_axis.py` (4 tests, direct comparison to the tuned-probe axis); two Octave pipeline tests updated, because they encoded the ideal-axis pass-through. Commit `c985409`. |
| H-10 | `esr/hyperfine.py:441-443` | The automatic field axis ignores nuclear spin I (half-span 0.6·Σ\|A\|), which cuts off the outer ¹⁴N nitroxide lines. | Repro | Fixed | Half-span is now 1.2·Σ I_k·\|A_k\| + 5·broadening (unchanged for spin-1/2). Test: `test_esr.py::test_hyperfine_auto_field_axis_contains_all_nitroxide_14n_lines`. Commit `28e3214`. |
| H-11 | `esr/hyperfine.py:369-376` | `detection_mode="derivative"` differentiates with respect to microwave frequency, not field, so the sign is inverted and the units are per Hz relative to `simulate_field_sweep`. | Repro | Fixed | Each line is weighted by −dν/dB (the Hellmann–Feynman slope), so the result is dS/dB in 1/T with the same sign as `simulate_field_sweep`. Tests: `test_esr.py::test_hyperfine_derivative_matches_field_sweep_derivative_convention`, `::test_hyperfine_derivative_matches_numerical_field_gradient`. Commit `28e3214`. |
| H-12 | `NQRDB/scripts/build_cwru_database.py:353` | CWRU T1/T2 values with no unit are stored as seconds, but the source tables are in ms. NaNO₂ is 1000× too long (cross-checked against NRL). The glycine T2 = 12.5/17.1 "s" values are suspect and are hard-coded in `nqr/piezo_detection.py:224`. | Repro (second source) | Fixed | `parse_time_s` uses whole-word unit matching, and CWRU unitless cells are ms. The 6 NaNO₂ values in `lines.jsonl`, `line_records.jsonl` and `nqr.sqlite` were patched from their stored source text (`References/` not available to rebuild); the audit, docs and evidence registry are updated. Glycine is left in seconds (no independent unit evidence; NRL repeats the numbers) and still needs curation. Tests: `NQRDatabase/tests/test_relaxation_units.py` (CWRU vs NRL agree within 1%), `test_nqr_database_relaxation_audit.py`. Commit `23def73`. |
| H-13 | `QDFT/finite_displacement.py:392-396` | `parse_abinit_structure` silently ignores `angdeg`, so a hexagonal cell becomes orthorhombic. The `n*value` multiplier syntax crashes it, and `!` comments are not stripped. | Repro | Fixed | Adds `angdeg` (ABINIT construction, including the trigonal setting), `n*value` repeats, `!` comments, and D-exponent floats (also closes L-22); strained jobs drop `angdeg` when writing `rprim`. Tests: `QuadrupolarDFT/tests/test_finite_displacement.py` (hexagonal and trigonal against textbook vectors), `test_abinit.py`, `test_strain_coupling.py`. Commit `23def73`. |

## CI and quality gates

| ID | Sev | Location | Problem | Evidence | Status | Resolution |
|---|---|---|---|---|---|---|
| G-01 | High | `PythonSpinDynamics/pyproject.toml` `[tool.mypy]` | mypy checks only `deprecation.py` and `experiment/cli.py` with `follow_imports = "skip"`, yet the CI job is named "Types, coverage, and full validation". | Verified | Open | |
| G-02 | High | `PythonSpinDynamics/tests/test_perf_golden.py:37,60-63` | `tests/perf_fixtures/golden.npz` is gitignored (`PythonSpinDynamics/.gitignore:11`) and absent, so the Numba/JAX golden-parity test skips on every run. `docs/performance.md:108` claims the fixture is committed. | Verified | Open | |
| G-03 | High | `.github/workflows/*.yml` path filters | Tests read sibling projects' data (the `nqr.sqlite` export, QuadrupolarDFT CIFs), but `python-spin-dynamics.yml`, `integration.yml` and `quadrupolar-dft.yml` don't trigger on those paths, so cross-project breakage goes unseen. | Reading | Open | |
| G-04 | High | `.github/workflows/nqr-database.yml` | The source data lives under the gitignored `References/`, so the rebuild and freshness steps never run in CI. The diff also omits the tracked `exports/nqr.sqlite`. | Reading | Open | |
| G-05 | Medium | `.github/workflows/release.yml` | Several safety gaps: no tests before tagging and releasing; `${{ inputs.version }}` interpolated into shell; `--find-links` without `--no-index`; a third-party action pinned by tag (not SHA) under `contents: write`; `workflow_dispatch` can tag any branch. | Reading | Open | |
| G-06 | Medium | `integration/src/mr_integration/database.py:15-21`, `coverage.py:18-22` | Default data paths use `Path(__file__).parents[3]`, which only works in a checkout. Installed from the wheel, `measured_lines()` raises and `load_dft_summary()` silently returns `()`. | Reading | Open | |
| G-07 | Medium | `PythonSpinDynamics/pyproject.toml` `python_files` | The `*_tests.py` pattern makes pytest collect about 552 tests twice (smoke/fixture/example aggregators). | Repro (`--collect-only`) | Open | |

## Medium

| ID | Area | Location | Problem | Evidence | Status | Resolution |
|---|---|---|---|---|---|---|
| M-01 | core | `phase_cycling.py:88-92` | The cycles assume M+ detection, but the arb10 kernels and motion code acquire M−, so combining them cancels the signal (\|S\| ≈ 1e-4). The convention is undocumented. | Repro | Open | |
| M-02 | core | `exchange.py:411-415` | The default spectrum span uses `ptp(offsets)`, not `max\|offset\|`, so peaks alias (1000/1100 Hz → 92/192 Hz). | Repro | Open | |
| M-03 | core | `pulses.py:98-110` | `quantize_phase` doesn't wrap (2π → 3π/2, −π/2 → 0), which has drifted from `absolute_phase.quantize_phase_to_bins`. | Repro | Open | |
| M-04 | core | `core/rotations.py:266` | `calc_rot_axis_arba4` calls `arccos` without clipping, giving NaN angles near the identity and a non-unit axis for an exact identity. | Repro | Open | |
| M-05 | core | `radiation_damping.py:440` | The drive sign is opposite to the package pulse convention (Mz tips to +y instead of −y). | Repro | Open | |
| M-06 | core | `core/kernels.py:68-80` vs `radiation_damping.py:247-273` | `time_scale` means s/unit in one API and unit/s in the other. | Reading | Open | |
| M-07 | core | `noise.py:390` vs `:423` | `target_snr` differs by √2 between the white and probe noise models. | Repro | Open | |
| M-08 | core | `coupling/multinuclear.py:205`, `coupling/isotopes.py` | Zeeman terms use \|γ\| for every nucleus, so the precession sense is wrong for ¹⁵N/²⁹Si mixed with ¹H. | Reading | Open | |
| M-09 | core | `composition.py:264` | `_spatial_resample` output inherits the input dtype, so integer channels are truncated. | Repro | Open | |
| M-10 | workflows | `workflows/imaging_frequency.py:333` | Spin-echo branch (RARE / spin-warp / multislice) samples k=0 one dwell after the echo. The gradient-echo branch uses `center_sample`. | Repro | Open | |
| M-11 | workflows | `workflows/imaging_3d.py:190-191` | Same echo timing error as M-10, plus a slice-rephase offset (`rephase_fraction·slice_duration/2`). | Reading | Open | |
| M-12 | workflows | `workflows/imaging.py:1389-1390` | Phase-encode grid misses k=0 for even `px` (half-pixel shift), and SENSE `_folding_matrix` assumes an exact DFT. The fix exists in `imaging_frequency.py` only. | Repro (emulated) | Open | |
| M-13 | workflows | `workflows/qspace.py:505,507` | `data_kind="magnitude"/"intensity"` keeps only the real part of complex input (an all-zero image for `1j`). | Repro | Open | |
| M-14 | workflows | `workflows/cpmg_ir.py:226,376,624` | The CPMG-IR combine `mrx2 − mrx1` has the opposite sign to every CPMG workflow, so a standard IR fit sees an inverted curve. | Repro | Open | |
| M-15 | workflows | `workflows/cpmg.py:809,1162,1383,1611`, `batched_sweeps.py:186`, `imaging.py:~1428`, `diffusion.py:938` | Echo-time bookkeeping is inconsistent: `(n+0.5)·TE` vs `(n+1)·TE`, and the encoding block is omitted, which biases `fit_imaging_echo_decay`'s `rho_map`. | Reading | Open | |
| M-16 | workflows | `workflows/diffusion.py:933` vs `:1120` | Matched and tuned diffusion CPMG differ by a factor of 2 (`/2` in the PAP combine). | Reading | Open | |
| M-17 | esr | `esr/relaxation.py:228-235` | T1 relaxes populations toward zero instead of the thermal deviation, so inversion never recovers. | Repro | Open | |
| M-18 | esr | `esr/eseem.py:409-412`, `esr/deer.py:372-373` | Quantum ESEEM/DEER traces are normalized by the first requested sample, not V(0). | Repro (ESEEM) / Reading (DEER) | Open | |
| M-19 | dnp | `dnp/nitroxide_relaxation.py:105` vs `dnp/nitroxide.py:121-123` | The two ¹⁵N halves disagree on the sign of A (signed tensor vs forced positive), so per-line T2s are assigned to the wrong lines. | Repro (partial) | Open | |
| M-20 | nqr | `nqr/full_dynamics.py:303` | `_default_carrier_hz` picks the line with `max()` over equal spin-1 strengths, so the excited line is arbitrary. | Repro | Open | |
| M-21 | nqr | `nqr/hamiltonians.py:117-120` | Transition labels from `argmax\|dipole\|` break for degenerate lines (at η=0, `transition('y')` raises KeyError) and are unstable across orientations. | Repro | Open | |
| M-22 | qdft | `QDFT/quadrupolar.py:57-60` | `nqr_frequencies_hz` returns every level difference, including forbidden lines for spin ≥ 5/2; these feed `efg_temperature_sweep`. | Repro | Open | |
| M-23 | qdft | `QDFT/finite_displacement.py:263` | The EFG rebuilt from ABINIT's 6-decimal eigenpairs fails the 1e-8 trace check, so the fallback can never succeed. | Repro | Open | |
| M-24 | nqrdb | `NQRDB/scripts/build_cwru_database.py:466-475` | `extract_isotope` defaults to ¹⁴N, so a 36 MHz ³⁵Cl furosemide line is stored as ¹⁴N. | Repro | Open | |
| M-25 | nqr | `nqr/structure_coupling.py:194-198` (and `QDFT/strain_coupling.py` twin) | The proton de-duplication key includes the lattice image, so protons on special positions are counted twice (√2 in the RMS). | Repro | Open | |
| M-26 | nqrdb | `NQRDB/data/exports/nqr.sqlite` | The shipped DB lacks the `site_consistency_flags` and Landolt flag tables, although the JSONLs are tracked. Stored η disagrees with the stored lines (ampicillin trihydrate, dipicolinic acid). | Repro | Open | |
| M-27 | nqrdb | `NQRDB` data, `integration/.../database.py:90-94` | NaNO₂ is stored as two compounds with conflicting dν/dT, and exact-string lookup can't merge them. | Repro | Open | |
| M-28 | probes | `probes/untuned.py:69-77` | `phi_eff` isn't updated when a leading segment is skipped, so the next segment gets the wrong phase (off by π/2). | Repro | Open | |
| M-29 | interference | `interference/cancellers.py:429-499` | Windowed ridge cancellers fail on windows with no baseline samples: `LinAlgError` with the defaults, or silently zero coefficients with a ridge. | Repro | Open | |
| M-30 | experiment | `experiment/io.py:203`, `interference/recordings.py:533` | `np.savez` appends `.npz`, so `save_run(rec, "out")` then `load_run("out")` fails, and the CLI prints the wrong path. | Repro | Open | |
| M-31 | experiment | `experiment/provenance.py:232-259` | Unseeded receiver-array noise is classified as deterministic. | Repro | Open | |
| M-32 | detection | `detection/gradiometer.py:97-115` | `Gradiometer.order` ignores loop radii and disagrees with `uniform_coupling()`. | Repro | Open | |
| M-33 | experiment | `experiment/config.py:76-108` | The config registry has drifted from the serialization registry, so valid saved runs (e.g. `SampledB0`, `ReceiverNetwork`, complex covariance) can't be shown. | Repro | Open | |
| M-34 | fields | `fields/nonlinear_magnetostatics.py:544,790` | Planar and axisymmetric magnets get μ_rec·Br remanence while the 3-D solver uses Br (about 5% too strong for NdFeB). | Repro | Open | |
| M-35 | fields | `fields/magnetostatics.py:556`, `coils.planar_spiral` | `circular_loop(axis="y")` is left-handed, so `solenoid(axis="y")` points along −y and `maxwell_pair(axis="y")` has the wrong gradient sign. | Repro | Open | |
| M-36 | fields | `fields/openems.py:1353,1867` | Multi-frequency dumps are normalized by port quantities computed only at `settings.frequency_hz`. | Reading | Open | |
| M-37 | fields | `fasthenry_interop.py:101-104,181` | FastHenry results are labelled with the caller's `freqs`, not its log-spaced `.freq` sweep. | Reading | Open | |
| M-38 | optimal_control | `optimal_control/multi_axis.py:70,112` | The `_built` cache is an init field, so `dataclasses.replace` shares stale operators. | Repro | Open | |
| M-39 | optimization | `optimization/_bounded.py:194-201,272-280` | Bounds are silently ignored by BFGS/CG and `result.x` isn't checked, so GRAPE can exceed `amplitude_max_hz`. | Repro | Open | |
| M-40 | optimal_control | `optimal_control/objectives.py:235`, `_jax_propagation.py:152` | With a gradient channel, `eigh` propagation is forced, and its gradient is NaN for a zero segment Hamiltonian (amplitude 0 at the position-0 member). | Reading | Open | |
| M-41 | thermal | `thermal/conduction.py:143-157` | Grids starting at r=0 aren't rejected (divide by zero), and the sphere cell volume `r²dr` isn't exact, contradicting the "exact / conserves energy" docstring. | Repro | Open | |
| M-42 | optimal_control | `optimal_control/solvers.py:685-686`, `objectives.py:410` | `best_fidelity`/`initial_fidelity` include the smoothness penalty, and `initial_fidelity` is evaluated at the unclipped `x0`. | Reading | Open | |
| M-43 | optimal_control | `detection_objective.py` (`per_rf_power`), `diffusion.gradient_moments` | Non-uniform `dt` is mishandled: a non-scalar power term, and `cumsum(g)*dt` instead of `cumsum(g*dt)`. | Reading | Open | |
| M-44 | workflows | `workflows/cpmg.py` `_calc_tuned_pulse_shape` | Found while fixing H-01. Each tuned pulse shape is max-normalized, and the corrected transient has a gate-on overshoot that depends on the absolute RF phase (peak 11.7 vs 13.7 against a steady 7.95). So the absolute pulse amplitude changes by about 15% with RF frequency/phase, and the π-step vs baseline echo ratio is 1.21, not 1. Normalizing by the steady-state amplitude would likely be the right reference. | Repro | Open | |

## Low

| ID | Area | Location | Problem | Status | Resolution |
|---|---|---|---|---|---|
| L-01 | core | `relaxation.py:1379` | `matrix_exponential` uses eig+inv and is ill-conditioned near exceptional points (2e-9 error); `exchange._matrix_exponential` is a robust twin. | Open | |
| L-02 | core | `sequences/motion.py:550-553` | `except TypeError` retry masks TypeErrors raised inside user callbacks. | Open | |
| L-03 | core | `sequences/compiler.py:104` vs `:227` | ADC offset phase is referenced to the block start but RF phase to the event start. | Open | |
| L-04 | core | `core/kernels.py` (numpy/numba/JAX) | Invalid `pul=0` silently uses the last pulse (JAX clamps); not validated against `len(Rtot)`. | Open | |
| L-05 | core | `motion.py:631` | The circular reflector treats the centre as a wall for large overshoots. | Open | |
| L-06 | core | `core/_jax_kernels.py`, `_jax_motion.py` | Importing them sets `jax_enable_x64` process-wide. | Open | |
| L-07 | repo | `PythonSpinDynamics/build/lib/` | A stale untracked build copy can shadow `src`. | Open | |
| L-08 | workflows | `workflows/imaging_frequency.py:229,567` | `gamma` is accepted but ignored by `run_rare_imaging` / `run_radial_ute_imaging`. | Open | |
| L-09 | workflows | `workflows/pgse.py:196-200` | Dead residual-moment check (`pass`). | Open | |
| L-10 | workflows | `workflows/pgse.py` | The `run_pgse` backends take different kwargs and use different default echo-time conventions. | Open | |
| L-11 | workflows | `workflows/imaging_3d.py:165-179` | `t1_map`/`t2_map` aren't shape-validated against `rho`. | Open | |
| L-12 | workflows | `workflows/acquisition.py` | `calc_macq_ideal_probe_relax4` and `_calc_macq_relax4` are identical. | Open | |
| L-13 | workflows | `workflows/electropermanent_*.py` | `_strict_axis`/`_positive`/`_bilinear_sample` are copy-pasted across 5 modules. | Open | |
| L-14 | workflows | `workflows/slice_selective.py` | `time_bandwidth` is half the conventional TBW (documented, but misleading). | Open | |
| L-15 | esr | `esr/lineshapes.py:292` vs `:310` | `width` means σ for Gaussian but HWHM for Lorentzian. | Open | |
| L-16 | esr | `esr/hyperfine.py:350-356` | Intensities aren't normalized by nuclear multiplicity (2× vs `ESRSpinSystem` for one proton). | Open | |
| L-17 | nonresonant | `nonresonant/field_reversal.py:267` | Rotation sense is +γ\|B\|dt (opposite to dM/dt = γM×B); flips the quadrature sign. | Open | |
| L-18 | nonresonant | `nonresonant/field_reversal.py:360,373` | Supercycle `echo_times` uses only `cycle[0]`'s duration, and `t2_seconds <= 0` is treated as infinite. | Open | |
| L-19 | esr | `esr/deer.py:231` | `deer_dipolar_spectrum` subtracts `signal[-1]`, not the (1−λ) offset its docstring describes. | Open | |
| L-20 | constants | `esr/systems.py:167`, dnp, nano_mr, nonresonant | Physical constants come from mixed CODATA vintages and are duplicated per module. | Open | |
| L-21 | nqr | `nqr/structure_coupling.py:319`, `QDFT/strain_coupling.py:633` | The posix-`shlex` CIF tokenizer rejects legal CIF quoting (`'O'Brien'`). | Open | |
| L-22 | qdft | `QDFT/abinit.py` | `_FLOAT` accepts Fortran `D` exponents, but `float()` fails on them. | Fixed | `_fortran_float` helper; fixed together with H-13 in `23def73`. |
| L-23 | nqrdb | `NQRDB/scripts/build_cwru_database.py` `first_number`/`parse_time_s` | No scientific notation; `µs` vs `us`; "hours" matches "us". | Open | |
| L-24 | nqr | `nqr/isotopes.py` | 65Cu γ is about 0.2% high; the docstring's higher axial line positions are wrong for integer spin ≥ 2. | Open | |
| L-25 | nqr | `nqr/hamiltonians.py:98` | The 1e-6 × span degeneracy floor can drop weak-field Zeeman transitions for heavy nuclei. | Open | |
| L-26 | nqr | `nqr/structure_coupling.py` vs `QDFT/strain_coupling.py` | The CIF parser, symmetry `eval`, `_spin_matrices` and cell matrix are duplicated across the two packages. | Open | |
| L-27 | nqrdb | `NQRDB/scripts/validate_database.py` | Checks only counts and orphans; no unit-plausibility checks. | Open | |
| L-28 | experiment | `experiment/config.py:238-284` | The TOML writer doesn't escape control characters or quote keys. | Open | |
| L-29 | interference | `interference/cancellers.py:945` | `frequency_domain_canceller` `hop`/`segment_length` are under-validated (crash, or 45% uncancelled). | Open | |
| L-30 | interference | `interference/_signal.py:55-74` | `spectral_phase_shift` zeroes the Nyquist bin for even n. | Open | |
| L-31 | experiment | `experiment/serialization.py:59-65` vs `provenance.py:33` | NaN arrays are emitted as non-standard JSON while scalars are rejected, so provenance can raise after the run. | Open | |
| L-32 | probes | `probes/tuned.py:365`, `probes/untuned.py:157-162` | `pnoise` omits the damping resistor's thermal noise (inherited from MATLAB). | Open | |
| L-33 | probes | `probes/*` | `tuned_probe_lp`/`_orig` are 95% identical; `find_coil_current*` duplicated; `_field`/`_with_fields` helpers have drifted. | Open | |
| L-34 | detection | `SQUIDMagnetometer`, `IdealFaradayCoil` | Advertise an `xp` (JAX) parameter but use boolean-mask indexing that JAX can't trace. | Open | |
| L-35 | probes | wiring `_SOLVE_CACHE` | Unbounded cache. | Open | |
| L-36 | fields | `fields/openems.py:1536` | The energy regex misses openEMS's padded `(- 9.50dB)` format. | Open | |
| L-37 | fields | `fields/coils.py`, `coil_properties.py` | The two inductance models assume different turn pitch (L/(N−1) vs L/N). | Open | |
| L-38 | fields | `nonlinear_magnetostatics._amg_linsolve_3d` | CG is used on a matrix the code itself says is non-symmetric. | Open | |
| L-39 | fields | `HarmonicEMSolution.magnetic_flux_density` | An explicit `relative_permeability` is ignored when B is stored; B/H consistency is never checked. | Open | |
| L-40 | fields | `fasthenry_interop.py`, FasterCap glue | Temp dirs leak; COM server not stopped on timeout; inconsistent `IsRunning`/`Quit` usage. | Open | |
| L-41 | fields | Brauer ν(B²), `_rod_dipoles`, birdcage helpers | Logic duplicated 2–4× (the remanence drift in M-34 came from this). | Open | |
| L-42 | optimization | `optimization/drivers.py:427-430` | AttributeError if the first inverse multistart is non-finite. | Open | |
| L-43 | analysis | `analysis/ilt.py:145` | Permitted complex input triggers ComplexWarning. | Open | |
| L-44 | thermal | `thermal/network.py:579-593` | `steady_state` `for/else` raises even when the last iteration converged; damping accepts T ≤ 0. | Open | |
| L-45 | thermal | `thermal/coupling.py:258` | `march` drops factory-internal sources and drifts from `fixed_point`. | Open | |
| L-46 | optimization | `UntunedProbeResponse`, `MatchedProbeResponse`, `GradientDriverResponse` | No `oversample >= 1` validation (ZeroDivisionError). | Open | |
| L-47 | thermal | `sar_source_from_eddy` | Inconsistent 1/2 factor between density and `EddyResult.power` when `time_average=False`. | Open | |
| L-48 | design | `ExperimentPredictor._cache`; `parameters/constructors.py` | Unbounded cache; `params.texc` and `pp.texc` alias one mutable array. | Open | |
| L-49 | core | `core/echo.py`, MATLAB `Version_3/code/calc_echo/calc_time_domain_echo.m` | Zero-fill places ω=0 off by up to 2 bins and `tvect` isn't the FFT grid (small at the default `numpts`); Octave parity locks it in. | Open | |
| L-50 | integration | `integration/src/mr_integration/database.py` | `with _connect()` doesn't close sqlite connections; the URI isn't built with `as_uri()`. Spin support is only 1 and 3/2, and DFT C_Q/η are averaged over non-equivalent atoms. | Open | |
| L-51 | repo | `AGENTS.md`, `.gitattributes`, `tests/test_validation_gaps.py` | AGENTS.md points at a stale OneDrive copy and omits the validation-matrix check; `.gitattributes` names a missing path; 5 permanently skipped placeholder tests. | Open | |
| L-52 | repo | `PythonSpinDynamics/docs/user_manual.pdf`, MATLAB tree, NQRDB exports | A 12 MB PDF in 129 commits (516 MiB pack); a Dropbox conflict copy and 23 paths with spaces/parentheses in MATLAB; 126 duplicate blobs across V2/V3; a `generated_at` timestamp in a tracked export. | Open | |

## Change log

- 2026-11-01: Tracker created from the review of `e25d982`. H-01 to H-13 assigned for fixing.
- 2026-11-01: H-10, H-11 fixed (ESR hyperfine field axis and derivative convention).
- 2026-11-01: H-03 (SLSE free precession), H-12 (CWRU T1/T2 units), H-13 and L-22 (ABINIT structure parser) fixed in `23def73`.
- 2026-11-01: H-06, H-07, H-08 (fields) fixed in `5df21b0`; H-04 (powder GRAPE weights) fixed in `c629a3b`, with the manual updated in `3d55b11`.
- 2026-11-01: H-05 (bSSFP catalyzation phase) fixed in `a2f5480`.
- 2026-11-01: H-02 (radiation damping during pulses) fixed in `c215ed5`.
- 2026-11-01: H-01 (tuned-probe transient) fixed in `1263cee` and H-09 (pipeline refocusing axis) in `c985409`; M-44 added (found during H-01). All High items closed.
