# Phase 1: absolute 14N reference

**Absolute-normalization regression passed; engineering Phase 1 remains open.** This study-local reference closes a declared
spin-to-ADC budget and checks two independent calculation paths. It is an
analytic calibration fixture, not measured instrument calibration or a scanner.
The reusable package physics is unchanged.

The working finite-pulse and inline-transport model is documented in
[PULSED_MODEL.md](PULSED_MODEL.md). The ideal reference below is retained only
as an absolute-normalization regression, not evidence of scanner feasibility.

## Run

From `PythonSpinDynamics` in the project's development/plot environment:

```bash
python studies/nqr_mail_screening/phase1/reference.py --output .tmp/nqr_mail_screening_phase1 --plot
python -m unittest tests.test_nqr_mail_screening_phase1
```

The runner first checks frozen Phase 0 readiness. It produces a JSON budget,
canonical schema-validated result record, compressed waveform arrays (seconds,
A m^2, Wb, V and ADC codes), and an optional plot. Nonzero exit indicates failure
of a declared numerical gate. `reference_report.json` here is the checked-in run
summary; regenerate waveform outputs under `.tmp`. The report records the base
commit, actual source hash, dirty-worktree status and numerical-library version.

## Scope and chosen reference

The addressed site is **one aniline 14N per fentanyl-HCl molecule**, using the
3.3043 MHz line and 610 us T2-star from [Malone et al., Table 2](https://doi.org/10.1093/pnasnexus/pgaf190).
The second aniline line sets the zero-field Hamiltonian together with the first;
only one transition is excited. Both nitrogens in the molecule must not be counted
as equivalent contributing sites. The salt molar mass is 372.93 g/mol from
[KEGG D10811](https://www.kegg.jp/entry/D10811).

A 1 g, fully crystalline, point-equivalent sample is placed at the center of a
10-turn reference solenoid (30 mm radius, 60 mm length). Its mass is concentrated
at one sensitivity value to isolate absolute normalization; it is not a geometric
model of an actual 1 g sample or an envelope-sized aperture. 14N fraction 0.9964
is a nominal assumption. Temperature 293.15 K is an assigned room-temperature
reference because the imported source does not give an exact reference temperature.

The pulse is an ideal, selective, on-resonance pi/2 pulse (50 us at full coupling),
powder averaged with 32-point Gauss-Legendre integration. After 100 us blanking,
record 1 ms of a real FID at 16 MHz. The receiver is an ideal high-impedance,
flat-gain voltage chain with gain 10,000, assumed 2 ohm coil resistance, and
1 nV/sqrt(Hz) input voltage noise. The ideal anti-alias bandwidth is fs/2.
The ADC has 16 bits across a 2 V bipolar span. These hardware values are
**predicted reference assumptions**, with no fitted or measured calibration.
Current-noise loading, coil/shield resonances, sample loading, RFI and actual
T/R overload are outside this fixture. It intentionally uses the simplest
receiver to expose normalization errors before adding those effects.

## Operator and unit audit

The library uses dimensionless spin matrices and Hamiltonians in rad/s, while
energy eigenvalues are exposed in Hz. Write p_j proportional to exp(-h f_j/kT).
The exact Gibbs deviation rho-I/3 is evaluated with expm1 to retain its small
amplitude without cancellation. Library `boltzmann_populations` is independently
checked against this implementation over a wide temperature range.

`nqr.simulation.equilibrium_density` instead returns
D = -diag(f_j-mean(f))/max|f_j-mean(f)|. Its high-temperature SI multiplier is
h max|f_j-mean(f)|/(3 k T). It is NOT already thermally normalized. Both the
selective pulse and the SLSE adapter are checked with exact and scaled initial
states. The factor 1/3 is the spin-1 partition-function factor.

For one orientation, the receive operator is I_b = b dot I in the energy basis.
The physical RF moment is N h gamma_Hz Re[2 (I_b)_lu rho_ul exp(-i omega t)].
The factor two includes the conjugate coherence; h gamma_Hz equals hbar gamma_rad.
Using `transition_signal` without this reconstruction would lose a factor two.
A full propagated operator-trace regression checks the real-waveform convention.

For an isotropic, selectively excited spin-1 transition with unit matrix element,
the independent analytic angular factor is (sin(a)-a cos(a))/a^2, where a is the
full-coupling flip angle. This is the integral of u sin(a u)/2 from -1 to 1.
Combined with the independent Gibbs population difference and nuclei count, it
provides a second absolute moment calculation.

Reciprocity gives flux = (B/I) dot moment and EMF = -d(flux)/dt. The first path
uses the library's segment Biot-Savart solenoid field at unit current. The second
integrates the source dipole vector potential around circular winding paths,
and agrees with the on-axis circular-loop expression. The waveform derivative
includes both carrier rotation and FID decay. A separate finite-difference flux
derivative checks its sign, phase and scale. See
[Ilott and Jerschow's reciprocity analysis](https://arxiv.org/abs/1806.01390).

## Noise and SNR convention

The real coil-noise **one-sided** PSD is 4 k T R; amplifier voltage-noise PSD is
e_n^2. Output sample variance is G^2(4 k T R+e_n^2) fs/2, using the classical
[Johnson-noise convention](https://pmc.ncbi.nlm.nih.gov/articles/PMC7339765/).
For a known real waveform s, SNR = sqrt(sum(s^2)/variance). This is a real matched
filter, not complex-envelope magnitude divided by two-quadrature noise RMS.

The ADC contributes q^2/12 variance only in the adequately dithered, unclipped
regime. The runner rejects less than 10 LSB analog RMS dither and verifies the
approximation through actual quantization of seeded noisy samples. It does not
claim an undithered sub-LSB signal is directly resolved by noiseless quantization.

The independent spectral path uses FFT/fs and **two-sided** PSD = variance/fs
with the package field-noise matched filter and rectangular DFT-bin integration.
Parseval matches the real time-domain norm. Field-reference invariance uses an
equivalent carrier-calibrated input-field coordinate with H = G omega Ncoil area;
it is not a claim that a dipole creates a uniform physical field across the coil.

## Acceptance evidence

The run uses 2,048 independent noisy/quantized records (seed 104729), with 32-record
batches. Averages group independent records without claiming shorter physical scan
time. Every physical repetition would also need recovery/state-history accounting.

| Check | Acceptance |
| --- | --- |
| Independent moment, high-T scale, mass and inverse-T scaling, SLSE scale, powder convergence, finite-difference EMF | Relative error below 1e-5 |
| Solenoid/dipole coupling, coil refinement and independent ADC waveform | Relative error below 1e-4 |
| Time/frequency and field/voltage SNR equality | Relative error below 1e-10 |
| Monte Carlo matched-filter noise SD | Within 6% of analytic value |
| Four-average SNR gain | Within 12% of 2 |
| Noise-score mean | Within 5 standard errors of zero |
| ADC clipping | Zero samples |

These are predeclared numerical-consistency tolerances, not experimental accuracy
bounds. The temperature scaling check doubles temperature while holding spectrum
and noise fixed; it audits the equilibrium law, not extrapolated material physics.

## Reference-run results

For the stated 1 g point-equivalent reference only:

| Quantity | Prediction / verification |
| --- | --- |
| Addressed nuclei | 1.6090e21 |
| Thermal population difference | 1.8032e-7 |
| Powder moment peak | 2.3971e-13 A m^2 |
| Coil unit-current sensitivity | 140.2835 uT/A |
| Pickup EMF peak at pulse end | 0.6982 nV |
| Real matched-filter SNR, 1 ms record | 0.009993 analytic; 0.009851 Monte Carlo |
| Independent waveform discrepancy | 0.00225% |
| Matched-filter noise-SD discrepancy | 1.44% |
| Four-average SNR ratio | 2.056, versus predicted 2 |

The deliberately simple FID/receiver has low single-shot SNR. This is not a
sensitivity estimate for an optimized SLSE/SORC envelope scanner. It supplies a
unit-consistent baseline against which later receiver/protocol improvements can
be measured. See the JSON report for the complete budget and acceptance results.

## Pre-polarization branch

`PreparedState` accepts a material-specific three-level population vector at the
end of buildup/transfer, explicit buildup/ramp/settling times, energy, evidence
class and source reference. It checks positivity and normalization, applies
settling relaxation toward thermal equilibrium, and returns the prepared density
and extra physical cycle time. Feed that density through `powder_pulse` using
`initial_deviation`; the same moment conversion applies. The energy remains
available for a future thermal/cycle ledger.

The existing `simulate_adiabatic_polarization_transfer` model is the candidate
upstream model. Its line enhancement is not blindly applied to every population:
a declared material-specific state reconstruction is required. Ramp losses must
already be included in the supplied state; the reference's single-T1 settling
model is an explicit approximation. The tests use a synthetic population fixture
to verify relaxation and cost accounting. No fentanyl transfer efficiency,
pre-polarization gain or equal-time performance benefit is claimed yet.

## Calibration and subsequent work

Unknown hardware uncertainties remain unknown: no confidence interval here covers
coil calibration, receiver gain/noise uncertainty or the point-sample approximation.
The source line uncertainties are retained in Phase 0; exact source temperature,
FID line shape, pulse relaxation and sample-form applicability need calibration.
The runner checks numerical correctness conditional on its input values.

The canonical record accounts for 2 s assumed handling plus the RF segment and
checks only the modeled winding current. Other hardware margins are absent, not
implicitly passed. Its detection decision remains not evaluated.

Phase 2 should replace the point/sample coil with the declared envelope geometry
and spatial reciprocity maps. Phase 3 supplies realistic SLSE/SORC waveforms,
loaded resonator transfer, state history and RFI. The existing SLSE API is audited
for absolute scale here; its effective decay is not inferred from the published
CPMG value. This analytic single-line gate permits that next engineering work
without pretending that the mail-screening problem has already been solved.
