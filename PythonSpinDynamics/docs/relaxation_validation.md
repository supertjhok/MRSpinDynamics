# Microscopic Relaxation Validation

Relaxation times are often used as fitted inputs, but a microscopic model tries
to explain them from molecular motion and spin interactions. That stronger
claim needs comparison with experiments that isolate the relevant mechanisms.
This page is for readers evaluating that evidence, not for users who simply
want to assign phenomenological T1 and T2 values to a workflow.

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

The validated model pieces are exposed through two high-level workflow paths.
`run_quadrupolar_relaxation` and its Arrhenius wrapper return transition-labeled
initial population and coherence decay rates from the secular fluctuating-EFG
model. `run_nmrd` and `run_field_cycling_nmrd` return condition-by-frequency
BPP grids while retaining fields, temperatures, correlation times, and spectral
density terms. These workflow tests establish correct assembly, axes, limits,
and metadata; they do not add independent experimental evidence beyond the
datasets described below.

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

The 24 experimental rates, published FA/FVA predictions, and FVA fit parameters
are now stored in `validation/experimental/goesweiner2020_bi209_r2.csv`. They
were transcribed from Table 1 and visually checked against the open-access PDF.

`ZeroFieldRedfieldEFGModel` implements the one-sided Redfield tensor of paper
Equations (A3a-b), including the two Kramers-degeneracy terms absent from its
standard NMR form. It averages rotated rank-2 EFG fluctuation operators over
SO(3), selects the paper's single/double-coherence basis, and uses the shifted
Lorentzian FVA spectral density. This is intentionally a lineshape-validation
model rather than a generally completely-positive propagation model.

For deuterated triphenylbismuth at 77 K, the calculated rate ratios are
`1, 0.307, 0.143, 0.118`; the published FVA ratios are
`1, 0.308, 0.148, 0.131`. The coupled lowest-transition rate is more than twice
its diagonal-only value. Across all six compound/temperature groups, the mean
relative difference from the published FVA rate shapes is 7.9% after fitting
one amplitude scale per group. That scale is required because the paper's
reported fluctuating `q_cc` and the package's Cartesian tensor use different
normalization conventions; `tau_c` and the vibrational frequency remain fixed
at their published values.

Acceptance criteria:

- every value records compound, isotope, temperature, transition, QCC, eta,
  units, source table, and reported uncertainty where available;
- the implementation demonstrates the failure of the diagonal-only
  approximation for the lowest transition;
- fitted transition-rate ratios reproduce the published qualitative ordering;
- mean relative deviation from the published FVA rate shapes is below 10%; and
- the diagonal-only calculation is explicitly shown to underestimate the
  lowest-transition decay by more than a factor of two.

These criteria pass. Direct agreement with experiment is uneven for
tris(4-fluorophenyl)bismuth at 77 K, for which the published FVA model itself
has deviations as large as 54%. This dataset therefore validates the
non-diagonal mechanism and published calculation more strongly than it
validates the simplified single-mode dynamics for every material.

### Stage 3: joint 14N NQR and 1H NMR relaxation in RDX

The RDX study reports 14N NQR T1, T2, T2e, and T2* over roughly 230--330 K,
together with 1H T1 dispersion from near zero field to 5.4 MHz [3]. Both sets
are interpreted through hindered NO2 rotation with an activation energy near
92 kJ/mol. It is therefore a stronger microscopic test than fitting unrelated
NMR and NQR curves separately.

The publisher-hosted large images for Figures 2 and 8 were digitized into two
fixtures. `smith2011_rdx_nqr_t1.csv` follows the activated branch of the
representative 5192 kHz ring-nitrogen line. A weighted log-linear fit gives
`Ea = 91.8 +/- 1.4 kJ/mol` from the stated vertical digitization uncertainty,
in agreement with the paper's approximately 92 kJ/mol interpretation. The
additional uncertainty in reading `1/T` from the horizontal axis is recorded
in the metadata but is not included in that formal fit error, so the agreement
should be read at a few kJ/mol rather than at decimal precision.

`smith2011_rdx_nmr_cross_relaxation.csv` records the three minima on the 322 K
proton-dispersion curve. Their digitized centers (115, 390, and 510 kHz) agree,
within the 10 kHz reading uncertainty, with the paper's assignments to the
nitro-nitrogen `nu0`, `nu-`, and `nu+` transitions (120, 390, and 510 kHz).
This is a direct cross-modal check: the proton experiment detects the nitrogen
transition frequencies through cross-relaxation.

The NMR and NQR samples were not identical: the NMR experiment used RDX in
Galden oil, whereas the NQR experiment used PE4 containing about 88% RDX.
Consequently, absolute amplitudes are not jointly fitted. A full simultaneous
fit of the NMR dispersion shape also needs the heteronuclear cross-relaxation
geometry and is still beyond this stage. Applying a straight Arrhenius line to
the NMR `T1` curve would be physically misleading because it contains several
resonances and background mechanisms.

Acceptance criteria:

- the NQR activated branch gives an activation energy within 4 kJ/mol of the
  reported value (passes: 91.8 versus approximately 92 kJ/mol);
- all three digitized proton `T1` minima agree with their assigned nitrogen
  transition frequencies within 10 kHz (passes);
- figure-derived points retain their source, sample, observable, and
  digitization uncertainty (passes); and
- a joint absolute-amplitude or held-out-temperature NMR/NQR fit is not claimed
  until the full heteronuclear dispersion model is implemented.

### Stage 4: NQRDatabase plausibility envelope

`validation/audit_nqr_relaxation.py` now processes every normalized record
read-only and writes `validation/nqr_relaxation_audit.json`. The audit confirms
89 relaxation-bearing lines: 84 with T1, 50 with T2, and three with T2*. All
normalized values are positive, and none of the 47 records containing both T1
and T2 has T2 greater than T1. These are useful normalization checks, not proof
of a microscopic model.

Only two records state a temperature and only nine preserve a reported
uncertainty. No dataset in the normalized database combines a temperature
series, uncertainty, pulse-sequence definition, and enough microscopic inputs
for a defensible fit. The audit therefore marks zero records as immediately
fit-eligible while retaining all 89 in its coverage report.

The sodium-nitrite warning is now concrete. The CWRU HTML and NRL summary contain
the same three frequencies, but every HTML-normalized T1 and T2 is exactly 1000
times the corresponding NRL value. The bare HTML cells omit units; the NRL
values are in seconds and are consistent with interpreting the HTML numbers as
milliseconds. Until the source normalization is curated in NQRDatabase, the
six HTML relaxation values are an unresolved unit conflict and must not be
treated as independent measurements or model targets.

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
3. J. A. S. Smith et al., "14N quadrupole resonance and 1H T1 dispersion in the
   explosive RDX," *Journal of Magnetic Resonance* 213, 98--106 (2011),
   <https://doi.org/10.1016/j.jmr.2011.09.011>.
4. S. Alcicek et al., "Data from: Zero- to ultralow-field relaxometry of
   chemical and biological fluids," Dryad (2023),
   <https://doi.org/10.5061/dryad.nk98sf7z7>.
5. Biological Magnetic Resonance Data Bank, NMR-STAR data library and REST
   access, <https://bmrb.io/>.

