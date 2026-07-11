# Microscopic Relaxation Validation

## Purpose

Relaxation calculations are only useful when the model boundary is as clear as
the numerical result. A curve that decays smoothly is not, by itself, evidence
that the underlying molecular motion or spin interaction is correct. This page
records the experimental datasets selected for PythonSpinDynamics, what each
dataset can test, and which conclusions the package may and may not draw from
it.

The validation sequence deliberately progresses from a compact solid-state
benchmark to models with more coupled mechanisms:

1. weak-field, homonuclear cross-relaxation in sodium chlorate;
2. transition-resolved quadrupolar relaxation in bismuth molecular crystals;
3. a joint NMR/NQR temperature and field-dispersion study of RDX;
4. broad plausibility checks against the local NQRDatabase;
5. zero- to ultralow-field liquid relaxometry; and
6. biomolecular NMR relaxation after the required CSA, diffusion, internal
   motion, and exchange terms are present.

The first three stages are quantitative literature reproductions. The database
stage is a coverage and anomaly-detection exercise, not a microscopic parameter
fit. The final two stages are intentionally gated on additional physics.

## Local NQRDatabase inventory

The inventory below was computed from
`../NQRDatabase/data/normalized/lines.jsonl` and the corresponding
`sites.jsonl` file on 2026-07-11. A relaxation record is a normalized line for
which at least one of `t1_s`, `t2_s`, or `t2_star_s` is present.

| Property | Count |
|---|---:|
| Relaxation records | 89 |
| Records with T1 | 84 |
| Records with T2 | 50 |
| Records with T2* | 3 |
| Assigned to 14N | 80 |
| Assigned to 35Cl | 2 |
| Assigned to 39K | 1 |
| Unassigned isotope | 6 |
| Records with QCC | 57 |
| Records with eta | 56 |
| Records with explicit temperature | 2 |
| Records whose original T1/T2 text contains an uncertainty | 9 |

The database is immediately valuable for unit checks, range checks, isotope
grouping, and detecting suspicious normalization. It is usually not sufficient
for estimating a correlation time or activation energy. In particular, an
absent temperature must not be silently interpreted as room temperature, and
T2, T2*, an echo-envelope lifetime, and a multipulse lifetime must not be
treated as interchangeable.

The largest source blocks are 62 records from the NRL line-summary material and
six furosemide records. The database also includes compact multi-line examples
for glycine and sodium nitrite. These are useful cross-line checks because QCC
and eta are available, but most do not state temperature, pulse sequence, or
uncertainty.

## Selected datasets

### Stage 1: 35Cl sodium chlorate in a weak magnetic field

Chen et al. measured room-temperature SLSE transverse lifetimes for sodium
chlorate powder at six weak fields [1]. The values are already represented in
`examples/plot_chen2020_slse_relaxation.py`:

| B0 (G) | T2,SLSE (ms) |
|---:|---:|
| 0.0 | 1.4 |
| 8.0 | 12.0 |
| 16.8 | 17.0 |
| 25.0 | 21.3 |
| 33.0 | 24.9 |
| 41.0 | 28.0 |

The zero-field T2* is 373 microseconds. The weak field splits and powder
broadens the spin-3/2 NQR transition. This reduces the spectral overlap between
neighbouring 35Cl sites and suppresses flip-flop relaxation. The validation
therefore tests the connection

`quadrupolar Zeeman splitting -> powder linewidth -> spectral overlap -> R2`.

The microscopic part of the prediction is the field-dependent spectral-overlap
factor obtained from the spin Hamiltonian. A field-independent floor and the
zero-field flip-flop strength are nuisance parameters fitted linearly. Passing
this benchmark does not establish an absolute dipolar rate from the crystal
structure.

Acceptance criteria:

- all six measurements and their provenance live outside plotting code;
- the zero-field point is reproduced and the lifetime increases monotonically;
- the RMS rate residual is at most 10 s^-1;
- the five nonzero-field lifetimes are within 15% of measurement; and
- convergence of the powder calculation is tested independently.

### Stage 2: transition-resolved 209Bi NQR coherence relaxation

Goesweiner, Westlund, and Scharfetter tabulate transition-specific experimental
R2 values for several 209Bi aryl crystals at 77 K and 310 K [2]. This dataset is
particularly diagnostic because a purely diagonal secular Redfield matrix is
not generally valid at zero field. Degenerate energy gaps couple density-matrix
coherences, including single- and double-quantum terms.

This stage will add fluctuating-EFG operators and retain Redfield blocks between
degenerate transition frequencies. It will first reproduce the published
fluctuation-only model and then the fluctuation-plus-vibration model. The
published transition-rate table, not values read back from a rendered plot,
will be the fixture.

Acceptance criteria:

- every value records compound, isotope, temperature, transition, QCC, eta,
  units, source table, and reported uncertainty where available;
- the implementation demonstrates the failure of the diagonal-only
  approximation for the lowest transition;
- fitted transition-rate ratios reproduce the published qualitative ordering;
- mean relative deviation is no worse than the published model within a
  documented digitization/rounding allowance; and
- trace preservation, Hermiticity preservation, and zero-coupling limits pass.

### Stage 3: joint 14N NQR and 1H NMR relaxation in RDX

The RDX study reports 14N NQR T1, T2, T2e, and T2* over roughly 230--330 K,
together with 1H T1 dispersion from near zero field to 5.4 MHz [3]. Both sets
are interpreted through hindered NO2 rotation with an activation energy near
92 kJ/mol. It is therefore a stronger microscopic test than fitting unrelated
NMR and NQR curves separately.

Values available only in figures will be digitized into a CSV fixture. Each
curve will carry the figure or table identifier, axis units, digitization date,
and a digitization uncertainty. Raw figure-derived points will never be labeled
as author-supplied data.

Acceptance criteria:

- NMR and NQR predictions share one Arrhenius correlation-time law;
- the fitted activation energy is consistent with the reported value within
  combined fit and digitization uncertainty;
- held-out temperatures or transitions are predicted rather than refitted; and
- T1, T2, T2e, and T2* remain distinct observables in the data model.

### Stage 4: NQRDatabase plausibility envelope

After the targeted mechanisms are in place, all usable local relaxation records
will be processed by a read-only inventory and validation script. This stage
will report coverage and anomalies by isotope, source, temperature availability,
and observable. Records missing essential conditions will be excluded from
microscopic fitting but retained in the coverage report.

The first known audit target is sodium nitrite: values in different sources
appear to mix intrinsic, inhomogeneous, and multipulse lifetimes. The audit must
resolve definitions and units before those records become fit targets.

### Stage 5: zero- to ultralow-field liquids

The open Dryad dataset accompanying zero- to low-field relaxometry of chemical
and biological fluids contains raw data for 1H-13C, 1H-15N, and 1H-31P spin
pairs, plus blood and plasma [4]. It is well suited to testing spectral-density
behavior when high-field secular assumptions cease to be reliable. Because the
archive is about 2.93 GB and contains raw experiments, this is an optional,
downloaded validation dataset rather than a repository fixture.

### Stage 6: BMRB biomolecular relaxation

BMRB NMR-STAR entries can provide residue-resolved T1, T2, R1rho, heteronuclear
NOE, field, temperature, sample conditions, values, and uncertainties [5]. This
is excellent machine-readable data, but it is not a fair target for the current
single-correlation-time dipolar model. Quantitative protein validation is gated
on chemical-shift-anisotropy relaxation, dipolar-CSA cross-correlation,
anisotropic rotational diffusion, model-free internal motion, and exchange
contributions to R2.

## Common validation record

Every ingested measurement must preserve the following fields when they are
available:

- stable dataset and observation identifiers;
- citation, DOI or repository identifier, and license/access notes;
- compound, phase or polymorph, sample form, and preparation details;
- isotope, site, and transition;
- B0 or NMR frequency, NQR frequency, temperature, and pressure;
- observable type, pulse sequence, delay definition, value, units, and
  uncertainty;
- QCC, eta, geometry, viscosity, or other model inputs;
- whether the value was author-supplied, transcribed from a table, or digitized
  from a figure; and
- transformations applied during normalization.

Tests may use normalized values, but the original text or table value remains
part of the fixture. A model fit must state which parameters were predicted,
which were independently measured, and which were fitted.

## Evidence levels and reporting

An analytical limiting-case test remains level A evidence. Reproducing a
published theory curve is level D. Comparing against a measured relaxation
dataset is level E. A regression test that preserves a previously obtained fit
is level R and does not become independent experimental evidence merely because
the expected numbers originated in a test.

The structured validation claim lives in `validation/evidence.json`; generated
summaries are produced by `docs/generate_validation_matrix.py`. Each completed
stage updates the evidence record with its tested range, metric, tolerance,
reproducer, reference, and limitations.

## References

1. C. Chen et al., "Single-shot spatially-localized NQR using field-dependent
   relaxation rates," *Journal of Magnetic Resonance* 311, 106660 (2020),
   <https://doi.org/10.1016/j.jmr.2019.106660>.
2. C. Goesweiner, P.-O. Westlund, and H. Scharfetter, "Spin-spin relaxation of
   nuclear quadrupole resonance coherences and the important role of degenerate
   energy levels," *Molecular Physics* 118, e1743888 (2020),
   <https://doi.org/10.1080/00268976.2020.1743888>.
3. M. J. Hunt et al., "14N quadrupole resonance and 1H T1 dispersion in the
   explosive RDX," *Journal of Magnetic Resonance* 213, 98--106 (2011),
   <https://doi.org/10.1016/j.jmr.2011.09.011>.
4. S. Alcicek et al., "Data from: Zero- to ultralow-field relaxometry of
   chemical and biological fluids," Dryad (2023),
   <https://doi.org/10.5061/dryad.nk98sf7z7>.
5. Biological Magnetic Resonance Data Bank, NMR-STAR data library and REST
   access, <https://bmrb.io/>.

