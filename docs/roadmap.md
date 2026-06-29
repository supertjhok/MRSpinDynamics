# MRSpinDynamics — Repository Survey and Roadmap

_Last updated: 2026-06-28_

This is a workspace-level survey and forward plan. Subproject-specific
status lives in each subproject's own docs (e.g.
`PythonSpinDynamics/docs/python_api/known_gaps.md`); this document is about
the **whole repository** and especially the connections *between* the
subprojects.

## 1. Current shape

Four components of very unequal maturity:

| Subproject | Scale | Maturity | Role |
|---|---|---|---|
| `PythonSpinDynamics/` | ~65k LOC, 39 test files, CI matrix (3 Py × 2 OS) | Production-grade | NMR/NQR/ESR simulation — the crown jewel |
| `NQRDatabase/` | ~4.5k LOC, ~900 measured lines, 184 compounds | Solid, app-backed | Curated **measured** NQR frequencies + provenance |
| `QuadrupolarDFT/` | ~1k LOC | Early/thin | Ab initio EFG → C_Q, η, **predicted** NQR frequencies |
| `MATLABSpinDynamics/` | large | Frozen reference | Validation oracle for the Python port |

`PythonSpinDynamics` is near feature-complete against its MATLAB origin; its
`known_gaps.md` reads as a stabilization list, not a missing-feature list. The
recent shared relaxation work also moved an important item out of the "pure
gap" category: NMR and NQR now share Liouville-space phenomenological
relaxation plus an opt-in Redfield/dipolar model with separate motional
averaging regimes for isotropic liquids and rigid solids. The highest-value
work is therefore still **structural and cross-cutting**: connect subprojects,
validate new physics against measurements, and make the mature Python surface
easier to install, cite, and use.

## 2. The headline gap: the subprojects don't talk to each other

The repository contains the three pieces of a complete **predict → simulate →
validate** loop, but (as of this survey) **zero cross-project imports** connect
them:

- `QuadrupolarDFT` computes a *predicted* NQR frequency from first principles
  (EFG → C_Q, η → ν).
- `PythonSpinDynamics` can *simulate* the full pulsed NQR response given
  (C_Q, η, spin).
- `NQRDatabase` holds the *measured* ν for 184 compounds, with citations.

Closing this loop is the single highest-science-value, modest-code improvement
available. A ready-made first case already exists on all three sides: **NaNO₂
¹⁴N**.

- DFT (ICSD 82857 run): C_Q ≈ −5.034 MHz, η ≈ 0.112 → lines 0.282, 3.635,
  3.916 MHz.
- Database (literature): QCC = 5.497 MHz, η = 0.378 → lines 1.038, 3.604,
  4.642 MHz.
- Simulator: given (C_Q, η, spin), diagonalizes to either line set.

### Convention bridge (validated)

The simulator parameterizes a site by `quadrupole_frequency_hz` (ν_Q, the η = 0
transition); the DFT side reports C_Q = eQVzz/h. These are linked by

```
ν_Q = C_Q · d / (4 I (2I − 1)),   d = 3 (spin-1), 6 (spin-3/2)
     = (3/4) C_Q   for spin-1
     = (1/2) C_Q   for spin-3/2
```

Verified end-to-end: feeding the NaNO₂ ¹⁴N C_Q/η through the conversion into
`spin_dynamics.nqr.diagonalize_site` reproduces `quadrupolar_dft.
nqr_frequencies_hz` to < 1 Hz (two independent Hamiltonian implementations
agreeing is the actual proof). This is implemented in the new `integration/`
package — see section 6.

## 3. Other technical gaps (workspace level)

**Performance / scale**
- ~~No compiled or GPU backend.~~ **Done** — the JAX/Numba acceleration
  workstream (Phases 0–4) is merged to main: opt-in Numba (`nogil` threaded) and
  JAX (`lax.scan`, vmap-batched, x64) backends for the `arb10` isochromat kernel,
  a batched primitive for sweeps/multistarts, batched NQR/ESR diagonalization, and
  — the headline — a **reverse-mode autodiff optimizer** that replaced
  finite-difference gradients (12–33× fewer evaluations on the ported objective).
  Full status in `docs/performance.md`. Remaining backend polish: GPU pays off only
  for vmap-batched work (a single sequential run is dispatch-bound), and only the
  ideal v0crit objective is JAX-ported so far (tuned/untuned/matched still route
  through the NumPy probe machinery).
- The NQR single-spin operator layer already builds matrices for any
  integer/half-integer spin; the real spin ≥ 5/2 gap is the high-field
  second-order-quadrupolar / MAS regime, not the operators (see §7, F1).

**Packaging / distribution**
- `version = "0.0.0"`, "Development Status :: 3 - Alpha", not on PyPI, no
  published API-docs site (the `generate_api_reference.py` + MkDocs scaffolding
  is half-built). This is the limiting factor on adoption/citation; a JOSS
  paper is realistic given the validation depth. A detailed beta/PyPI/MkDocs
  release plan now lives in `docs/publishing_plan.md`.
- The Python user manual has been reorganized around model boundaries,
  relaxation levels of description, and workflow guides, but there is still no
  hosted documentation site that ties the manual, API reference, examples, and
  roadmap into one navigable publication-grade artifact.
- ~~`QuadrupolarDFT/runs/**` commits raw ABINIT binary outputs (`.nc`, `_DDB`,
  `.abo0001`…) into git.~~ **Done** — `QuadrupolarDFT/.gitignore` ignores `runs/`
  and explicit ABINIT binary patterns (`*.nc`, `*_DDB`, `*_WFK`, `*.abo[0-9]*`,
  `*.run/`, …); no binary outputs are tracked.

**Test / CI**
- `PythonSpinDynamics` has the broadest CI surface. `QuadrupolarDFT` now has a
  dedicated workflow for lint, unit tests, finite-temperature examples, and
  ABINIT-input staging. `NQRDatabase` now has a rebuild/validation workflow that
  runs the PDF-backed builder, validates SQLite and JSONL exports, and fails if
  generated artifacts are stale.
- No coverage measurement; benchmarks exist but aren't gated, so perf
  regressions are invisible. ESR (newest module) has the thinnest test surface.

**Physics depth** (next-frontier, from the science-impact roadmap)
- q-space / averaged-propagator pore-size (diffusion-diffraction) — first
  inverse-imaging increment is now in `PythonSpinDynamics`: the existing
  circular-pore example showed forward diffusive diffraction minima, while the
  new `spin_dynamics.workflows.qspace` layer adds centered q-space axes, pore
  form factors, direct complex inversion, intensity/autocorrelation inversion,
  and support-constrained phase retrieval. The new
  `plot_pgse_qspace_pore_imaging.py` example reconstructs a circular pore from
  ideal and finite-SNR q-space intensity data. Remaining work is connecting the
  inverse path directly to finite-pulse walker measurements and broader pore
  geometries.
- Microscopic relaxation first increment is done in `spin_dynamics.relaxation`:
  shared Liouville helpers, `PhenomenologicalRelaxationModel`, dipolar bath
  sources, rigid-solid and isotropic-liquid motional averaging, and
  `RedfieldDipolarRelaxationModel`. Two examples exercise the model:
  `plot_redfield_nano2_slse.py` for coherent powder/full-density-matrix
  ¹⁴N SLSE in NaNO₂, and `plot_redfield_water_cpmg.py` for spin-1/2 proton
  CPMG in bulk water. Remaining depth: stronger experimental benchmarks,
  orientation/powder convergence studies, non-secular and cross-correlated
  terms where justified, and a coherent many-spin dipolar-network model when
  stochastic Redfield is not enough.

**Finite-temperature DFT** (done — harmonic path). A static DFT EFG is a 0 K,
fixed-geometry quantity, but measured NQR lines are strongly temperature
dependent. `quadrupolar_dft.vibrational` adds harmonic vibrational averaging of
the EFG tensor — `<V_ij>(T) = V_eq + ½ Σ_k (∂²V/∂Q_k²)⟨Q_k²⟩(T)`, averaged in
the crystal frame then diagonalized so η(T) shifts too — plus the analytic
single-libration Bayer fit. Validated against the NaNO₂ ¹⁴N temperature series
in the database (recovers a ~210 cm⁻¹ libration, right dν/dT sign/magnitude;
the residual is the ferroelectric-transition softening). The **full three-stage ABINIT DFPT workflow** is now done
(`quadrupolar_dft.finite_displacement` + `abinit_phonon`, CLI
`examples/abinit/efg_temperature.py`): **phonon** (generate DFPT + anaddb inputs
from a converged static input) → **displace** (parse anaddb eigenvectors →
±displaced EFG inputs + manifest; `run_finite_displacement_wsl.sh` loops ABINIT)
→ **collect** (parse the real `.abo` EFGs → central-difference mode curvatures →
sweep → ν(T), dν/dT). The EFG collect path is validated against the real
`nano2_efg.abo` (recovers C_Q = −5.17 MHz, η = 0.043). And the **integration
hook** (`mr_integration.temperature.compare_temperature_coefficients`) matches
predicted `(frequency, dν/dT)` against the database's `dnu_dt_khz_per_c`.
Remaining: verify the anaddb eigenvector parser against a real phonon run (or add
a netCDF/phonopy reader; a `--modes` JSON escape hatch exists meanwhile), and
AIMD/PIMD averaging for anharmonic cases like NaNO₂ near Tc.

## 4. Ranked opportunities (impact ÷ effort)

1. **Close the DFT → sim → DB loop.** Highest science value, modest code.
   Started here as the `integration/` package; NaNO₂ is the seed case.
2. **q-space diffusion-diffraction** (roadmap #4) — first inverse-imaging
   increment done; next value is finite-pulse/walker-to-image validation and
   non-circular pore examples.
3. **Validate and broaden microscopic relaxation.** The shared Redfield/dipolar
   model exists; next value is tying it to measured liquid NMR/NQR relaxation
   data, convergence checks, and clearer limits of validity.
4. ~~**JAX/Numba isochromat backend** — unlocks speed *and* autodiff pulse
   optimization.~~ **Done** (Phases 0–4 merged; `docs/performance.md`). The
   autodiff machinery it unlocked is now the launchpad for **GRAPE / optimal
   control** — see §7, F8 — and for Bayesian inference (§7, F6).
5. **Publish.** Release the workspace as a single citable unit: one repo version,
   one GitHub Release, one Zenodo DOI. Process: `docs/release_process.md`
   (scaffolding — `CITATION.cff`, `.zenodo.json`, `CHANGELOG.md`, version-sync
   scripts, and the release workflow — is in place; remaining is the one-time
   Zenodo↔GitHub hookup and tagging `v0.1.0`). Independent PyPI publication of a
   subpackage is a deferred future option (`docs/publishing_plan.md`).
6. **Database enrichment from DFT** — a "predicted vs measured" column in the
   NQR explorer UI. Visually striking, directly useful.
7. **Repo hygiene** — ABINIT binaries are now gitignored (done); remaining:
   add coverage reporting and broaden CI beyond smoke/rebuild checks where the
   newer subprojects still need deeper fixtures.

## 5. The q-space diffusion-diffraction layer (started)

Opportunity #2 is now underway in `PythonSpinDynamics`.

First increment (done):

- `spin_dynamics.workflows.qspace` — ideal q-space analysis helpers:
  `qspace_axes_from_real_space`, `real_space_axes_from_qspace`,
  `pore_form_factor_from_density`, `reconstruct_qspace_image`, and
  `phase_retrieve_qspace_magnitude`.
- Complex form-factor data invert directly to pore density.
- Magnitude/intensity-only data invert directly to the pore
  Patterson/autocorrelation image, with the non-uniqueness made explicit.
- Support-constrained, non-negative HIO/error-reduction phase retrieval provides
  a first pore-shape estimate from magnitude-only q-space data.
- `examples/plot_pgse_qspace_pore_imaging.py` demonstrates the inverse path for
  a circular pore, including a finite-SNR q-space intensity case (`--snr`) in
  addition to the ideal response.
- Tests validate exact complex inversion, intensity-to-autocorrelation behavior,
  magnitude-only phase retrieval up to shift/reflection ambiguity, and q-axis
  validation.

Next increments:

- Feed finite-pulse walker/PGSE or PGSTE q-space grids into the inverse layer,
  so the reconstruction accounts for realistic pulse blurring rather than only
  ideal short-gradient-pulse form factors.
- Add non-circular pore examples (ellipse, slit, connected domains) and compare
  how much shape survives magnitude-only phase retrieval at finite SNR.
- Add sampling-window and missing-k-space studies, since real q-space pore
  imaging is often limited more by q-coverage and SNR than by the ideal inverse.

## 6. The `integration/` layer (in progress)

A new top-level package, `mr_integration`, that depends on both
`spin_dynamics` and `quadrupolar_dft` and reads the NQR SQLite export. It is the
concrete realization of opportunity #1.

Scope of the first increment:

- `conversions` — validated C_Q ↔ ν_Q mapping and a `quadrupolar_site_from_cq`
  / `quadrupolar_site_from_efg_record` builder that returns a
  `spin_dynamics.nqr.QuadrupolarSite`.
- `cross_validation` — run DFT-derived parameters through the simulator and
  assert self-consistency with `quadrupolar_dft.nqr_frequencies_hz`.
- `database` — query measured lines for a compound/isotope from `nqr.sqlite`.
- `pipeline` — end-to-end **predicted vs measured** comparison report.
- `examples/nano2_dft_vs_measured.py` — the seed demonstration.

Second increment (done): **database self-consistency validator**
(`database_validation`). Each curated site stores both `(qcc, eta)` and its
measured lines; the simulator must reproduce one from the other.
`validate_database()` scans every supported site (currently 61: ¹⁴N spin-1 plus
a few spin-3/2) and sorts by discrepancy. On the current export, 56/61 are
self-consistent and 5 are flagged as likely transcription/OCR errors — e.g. a
³⁵Cl Cladribine line implying a ~4 MHz-larger `C_Q` than stored, and several
¹⁴N sites whose stored `eta` disagrees with their lines (the spin-1
implied-parameter back-solve localizes the error to `eta`, since QCC matches to
< 1 kHz). Example: `integration/examples/database_consistency.py`.

Third increment (done): **explorer flag overlay**. `flag_export` writes a
`site_consistency_flags` table (and JSONL) into the database; the NQR explorer
(`explorer_server.py` + `explorer_static`) reads it and shows a per-site badge
(flagged / simulator-verified) with the diagnostic detail, plus a compound-level
"N sites flagged" summary chip. Regenerate after a build with
`integration/scripts/write_consistency_flags.py`. The overlay is derived, so the
database build never depends on the simulator, and the explorer degrades
gracefully if the table is absent.

Fourth increment (done): **Landolt review-queue flagging**. The Landolt import
splits frequencies and `(QCC, eta)` into independent lists per measurement set,
so the site-level validator can't see them together. `landolt_validation`
predicts the two strong lines (`nu_+`, `nu_-`) per `(QCC, eta)` pair and checks
they appear among the tabulated frequencies; `landolt_review_export` routes
mismatches into `landolt_review_queue` (a `quad_consistency_mismatch` issue
flag + raised priority + a `landolt_consistency_flags` detail table/JSONL), and
the review GUI shows a diagnostic banner. On the current export, 26 of 141
checked Landolt sets are flagged — e.g. a `QCC` OCR error (313 MHz for a line at
~2.7 MHz). Regenerate with
`integration/scripts/write_landolt_review_flags.py`.

Later increments: feed DFT η/C_Q distributions into the simulator's
EFG-broadening models; widen DFT coverage so the predict-vs-measured loop runs
over more than NaNO₂; extend Landolt checking beyond ¹⁴N once spin ≥ 5/2 lands.

## 7. Next scientific frontiers (creative roadmap)

Sections 1–6 trace a workspace that has closed its first predict→simulate→validate
loop and ported the MATLAB physics. The questions below are deliberately
forward-looking: each is a *new physics regime* rather than a stabilization task,
each builds on machinery that already exists, and each is chosen for science reach
per unit of new code. They are ranked by impact ÷ effort.

A recurring enabler shows up in several of these: the single-spin operator layer
(`spin_dynamics.nqr.operators`) already constructs angular-momentum matrices for
**any** integer or half-integer spin (`validate_spin`, `spin_dimension = 2I+1`),
and `coupling/hamiltonians.py` already builds Zeeman + isotropic/secular J +
RF Hamiltonians in a full coupled-spin Hilbert space. The frontiers below mostly
add *new Hamiltonian terms, regimes, and observables* on top of that substrate,
not new linear algebra.

### F1. High-field quadrupolar solid-state NMR: second-order lineshapes, MAS, MQMAS

**What.** A high-field counterpart to the existing zero-field NQR engine:
the second-order quadrupolar broadening of the central transition, magic-angle
spinning (MAS) averaging, and the multiple-quantum MAS (MQMAS) and
satellite-transition MAS (STMAS) correlation experiments that remove it.

**Why it matters.** This is the single largest untapped audience for the package.
Half-integer quadrupolar nuclei — ²⁷Al (5/2), ¹⁷O (5/2), ⁵¹V (7/2), ⁷¹Ga,
⁹³Nb (9/2), ²⁰⁹Bi (9/2) — dominate the solid-state NMR of glasses, zeolites,
minerals, ceramics, battery cathodes, and heterogeneous catalysts. MQMAS is the
workhorse that makes those spectra interpretable. Today the simulator stops at
spin-1 and spin-3/2 in the *zero-field* limit; nothing models the high-field
second-order regime.

**Build on.** The operator layer already generalizes to any I; `nqr.zeeman`
already mixes a Zeeman term into the quadrupolar Hamiltonian; `nqr.orientations`
and the powder-averaging in `full_dynamics` already exist. What is new: the
second-order average Hamiltonian (or exact diagonalization in a tilted frame),
a spinning-frame time-dependence for MAS, and the 2-D shear/processing for MQMAS.
This is also the unlock that lets the `integration/` ν_Q convention and the
Landolt validator finally run for the *majority* of catalogued nuclei rather than
just ¹⁴N — directly retiring the "extend Landolt checking beyond ¹⁴N once spin ≥
5/2 lands" note above and widening the DFT→sim→DB loop to ²⁷Al/¹⁷O/⁵¹V.

**Effort.** Large, but high-leverage and incremental (static second-order
lineshape → MAS → MQMAS), and it compounds with the DFT and database work.

### F2. Pulsed dipolar EPR: DEER/PELDOR and ESEEM/HYSCORE

**Status: DEER and the ESEEM/HYSCORE/ENDOR family done.** First increment —
`spin_dynamics.esr.dipolar` + `esr.deer` — adds the point-dipole coupling
(canonical 52.04 MHz nm³, derived from constants), single-pair and
powder-averaged DEER kernels, a P(r) forward model, the dipolar (Pake) spectrum,
Tikhonov-regularized P(r) recovery (reusing `analysis.regularization`), and an
independent two-electron density-matrix check matching the analytic kernel to
~1e-14. Second increment — `esr.eseem` + `esr.hyscore` + `esr.endor` — adds the
secular/pseudosecular `HyperfineCoupling` for an S=1/2, I=1/2 pair, analytic two-
and three-pulse ESEEM with density-matrix validation (electron coherence-pathway
selection, plus an explicit phase-cycled variant proven to match it to ~1e-15),
2D HYSCORE with cross-peaks at the nuclear frequencies, and Davies/Mims ENDOR
with the Mims blind spots. Third increment generalizes the same engine to
**quadrupolar nuclei (I=1, 3/2)**: `HyperfineCoupling` gained `nuclear_spin`,
`quadrupole_hz`, and `eta` (the quadrupole term reuses the NQR Hamiltonian),
`manifold_frequencies` returns the per-manifold nuclear frequencies by
diagonalization, and the ESEEM/HYSCORE/ENDOR sequences work for any of these
spins — validated against the ¹⁴N exact-cancellation NQR lines. Examples
`plot_esr_deer.py` and `plot_esr_eseem_hyscore.py`; the previously thin ESR test
surface gained ~100 tests across
`test_esr_deer/coverage/eseem/hyscore/endor/quadrupolar.py`. **Remaining:** spin
I>3/2, anisotropic hyperfine `A` tensors with powder averaging, tilted
quadrupole tensors, multiple coupled nuclei, and finite/shaped pump pulses.

**What.** Turn the nascent single-electron ESR module into a *distance-* and
*weak-coupling-* measuring tool: four-pulse DEER/PELDOR yielding a dipolar
evolution trace and an inter-spin distance distribution P(r) (done), plus two-
and three-pulse ESEEM, 2-D HYSCORE, and Davies/Mims ENDOR for resolving weak
hyperfine couplings (done for I=1/2, 1, and 3/2).

**Why it matters.** DEER is *the* structural-biology EPR experiment —
nanometer distance distributions between site-directed spin labels constrain
protein conformations and complexes; the same physics maps spin-label distances
in polymers and MOFs. ESEEM/HYSCORE map ligand nuclei around metal centers.
These are the highest-citation pulsed-EPR methods and are entirely absent.

**Build on.** `esr/systems.py` already has g-tensors, `esr/hyperfine.py` the
hyperfine coupling, `esr/pulsed.py` rectangular-pulse FID/Hahn echo with
Liouville T1/T2, and `esr/orientations.py` powder grids. The DEER increment
reused exactly this plus the regularized inverse-problem machinery in
`analysis/regularization.py`. For ESEEM/HYSCORE the new piece is nuclear
modulation under the hyperfine Hamiltonian during the pulse delays and the 2-D
correlation processing for HYSCORE.

**Effort.** Medium. DEER (done) was mostly two-spin dipolar evolution + an
inversion already supported by the analysis layer; ESEEM/HYSCORE is next.

### F3. NQR detection-science toolkit (explosives, narcotics, pharma screening)

**What.** An application layer for ¹⁴N NQR *detection*: SORC/SSFP and SLSE
steady-state detection trains, temperature-compensated frequency tracking, RF
interference (RFI) modeling and mitigation, and quantitative
probability-of-detection / ROC curves versus SNR, scan time, and temperature drift.

**Why it matters.** NQR is a deployed standoff technique for explosives (RDX,
TNT, PETN, ammonium nitrate) and contraband, and an emerging pharmaceutical
quality-control method. This is the one frontier that exercises *all four*
subprojects at once: the database already holds ¹⁴N compounds (melamine,
metformin, paracetamol) and the U.S. Navy/NRL NQR data tables; the simulator has
spin-3/2 SLSE and full-density-matrix ¹⁴N dynamics; the DFT side supplies the
temperature coefficients that detection hardware must track. It is a compelling,
translational story for a methods paper.

**Build on.** `nqr.full_dynamics.simulate_full_slse`, `nqr.sequences`, the
finite-temperature dν/dT from `quadrupolar_dft.vibrational`, and the measured
lines + temperature series already in `NQRDatabase`. What is new: steady-state
SORC/SSFP trains, a detection-statistics layer (matched filter, ROC/PoD), and an
RFI noise model layered onto the existing `noise.py`.

**Effort.** Medium, mostly orchestration of existing primitives plus a small
detection-statistics module — high visibility for modest new physics.

### F4. Zero- to ultralow-field (ZULF) NMR and J-spectroscopy

**What.** Evolution of J-coupled spin systems at zero and ultralow field, where
chemical shift vanishes and the spectrum is governed by scalar couplings —
including the heteronuclear J-spectra and the field-cycling
(prepolarize-high / detect-low) protocol that ZULF instruments use.

**Why it matters.** ZULF NMR is a fast-growing, magnet-free modality
(spectrometers built around atomic magnetometers) that gives sharp, absolute
J-resolved spectra and pairs naturally with hyperpolarization. It is low-cost to
realize experimentally and currently has thin open-source simulation support.

**Build on.** `coupling/hamiltonians.py` already provides
`isotropic_j_hamiltonian` and `zeeman_hamiltonian` over a coupled-spin Hilbert
space — ZULF is largely the *zero-Zeeman* evolution of exactly that Hamiltonian
with a sudden field drop. `prepolarization.py` already models high-field
prepolarization for the field-cycling step. What is new: the sudden-transition
(non-adiabatic) field switch, a magnetometer-style detection operator, and
zero-field selection rules / observables.

**Effort.** Small-to-medium — arguably the highest novelty per line of new code,
because the Hamiltonian substrate is already present.

### F5. Hyperpolarization and long-lived states: PHIP/SABRE and singlet order

**What.** Parahydrogen-induced polarization (PHIP/SABRE) source terms and
long-lived singlet-state (LLS) preparation, storage, and readout — including
singlet relaxation times far longer than T1.

**Why it matters.** Hyperpolarization gives the 10³–10⁵ signal gains behind
metabolic MRI and trace-analyte NMR; long-lived singlet order extends the clock
on which hyperpolarized information survives. Both are high-impact and chronically
under-served by open simulators.

**Build on.** The `coupling/slic.py` SLIC module already models the
spin-locking-induced crossing used to access singlet order, and the coupled-spin
Hamiltonians are in place. What is new: a parahydrogen singlet initial state and
addition operator, singlet/triplet basis projectors, and a singlet-specific
relaxation channel layered onto `relaxation.py`.

**Effort.** Medium; strong synergy with F4 (ZULF) since singlet order is often
prepared and read at low field.

### F6. Bayesian inverse inference: fitting spin parameters to measured spectra

**What.** Invert the existing forward simulators: given a measured FID, echo
train, lineshape, or DEER trace, infer posterior distributions over the physical
parameters (C_Q, η, T1, T2, D, P(r), exchange rates) with calibrated uncertainty.

**Why it matters.** It converts the package from a forward simulator into a
*data-analysis* tool — the thing experimentalists actually need at the bench — and
it turns the validated forward models into likelihoods. It also naturally extends
the `integration/` consistency validators from "flag/keep" to "fit and report a
posterior," and gives the database a principled way to reconcile predicted and
measured values.

**Build on.** The forward models across `nqr`, `sequences`, `workflows`,
`exchange`, and ESR; the regularized inversion already in `analysis/`; and the
optimization drivers in `optimization/`. The JAX backend is **already merged**,
so its reverse-mode autodiff can supply the gradients that gradient-based MCMC
(HMC/NUTS) and simulation-based inference want — no new engine required. What is
new: a thin likelihood/MCMC (or simulation-based inference) layer with priors and
posterior-predictive checks.

**Effort.** Medium; the autodiff substrate it needs already exists (roadmap §4
item 4 is done), so the work is the inference layer, not the gradients.

### F7. Mechanism-resolved relaxation and field-cycling relaxometry (NMRD)

**What.** Add the *quadrupolar* relaxation mechanism (T1Q for spin ≥ 1, the
dominant channel for many quadrupolar nuclei) alongside the existing dipolar
Redfield model, and a field-cycling relaxometry workflow that produces NMRD
profiles — T1 as a function of Larmor frequency.

**Why it matters.** NMRD profiles are the standard fingerprint for molecular
dynamics, MRI contrast-agent design, and porous-media surface relaxivity;
quadrupolar relaxation governs line widths and detectability throughout the NQR
work. This deepens the physics already started in `relaxation.py` rather than
opening a new front.

**Build on.** `relaxation.py` (Redfield/dipolar spectral densities, motional
averaging) and `prepolarization.py` (the high-field polarize step of a
field-cycling experiment). What is new: the quadrupolar coupling spectral density
and a field-swept T1 driver.

**Effort.** Small-to-medium; a focused extension of an existing module.

### F8. GRAPE and quantum optimal control

**What.** A general gradient-based optimal-control layer: optimize the *full*
piecewise-constant control waveform (amplitude and phase, or I/Q) of an arbitrary
pulse to drive a spin system to a target — either state-to-state transfer fidelity
or full propagator/gate fidelity — using reverse-mode autodiff through the
differentiable forward kernel. GRAPE first; a Krotov variant and a *robust*
ensemble objective (averaging fidelity over B1-inhomogeneity and offset
distributions) as follow-ons.

**Why it matters.** Optimal-control pulses are the modern route to broadband,
B1-inhomogeneity-robust, low-power, or sharply selective excitation, inversion,
and refocusing. That is directly useful for the package's existing strengths:
single-sided / NMR-MOUSE hardware with huge static gradients and B1 falloff,
selective control of quadrupolar transitions, and — paired with F1 —
central-transition-selective pulses for half-integer quadrupolar nuclei, and —
paired with F2 — shaped DEER pump pulses. It generalizes the package's *current*
autodiff optimizer, which today covers only the single ideal-v0crit
refocusing-phase objective, into a reusable optimal-control toolkit.

**Build on.** This is now a short step rather than a new engine, precisely because
the JAX/Numba workstream is merged: the differentiable `lax.scan` forward kernel
(`core/_jax_kernels.py`), the `jax.value_and_grad` + jit objective factory
(`optimization/_jax_objectives.py`), the analytic-gradient L-BFGS-B driver
(`optimization/_bounded.py::scipy_maximize_with_grad`), and the vmap **batched**
primitive (Phase 2b) for multistarts and ensemble/robust averaging all already
exist. What is new: a general control parameterization (a time grid of
amplitudes/phases, not just composite-pulse phases), state-transfer and
propagator/gate fidelity objectives, and the robust ensemble objective.

**Effort.** Medium — mostly generalizing existing autodiff machinery — with high
payoff because it composes with F1 (selective pulses), F2 (DEER pump pulses), and
the single-sided/MOUSE workflows.

### Cross-cutting note

The **JAX/Numba acceleration backend is done and merged** (roadmap §4 item 4,
`docs/performance.md`), so it is no longer a blocking prerequisite — it is an
*enabler already in hand*. Three frontiers cash in on it directly: F8 (GRAPE)
generalizes its autodiff optimizer, F6 (Bayesian inference) reuses its gradients
for HMC/SBI, and F1/F2 lean on its vmap-batched primitive for the embarrassingly
parallel MAS/MQMAS and DEER powder averages (where GPU batching pays off, even
though a single sequential run is dispatch-bound). The remaining backend polish —
JAX-porting the tuned/untuned/matched objectives and broadening GPU-batched
adoption — is incremental and best done in service of these science frontiers
rather than as standalone infrastructure.
