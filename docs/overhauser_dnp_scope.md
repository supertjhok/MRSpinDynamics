# Overhauser DNP: literature survey and implementation scope

Survey date: 2026-10-24. Repository inspected at `4f39f0a`.
This document records the initial design and subsequent implementation status.
The original repository assessment below describes the survey-time baseline.

## Implementation status (2026-10-24)

The CW core, absolute FFHS translational rates, effective saturation, nuclear
buildup/recovery, measured-rate inputs, field/power workflows and TEMPOL/water
concentration examples are implemented. The motion-based stage now also has
absolute fixed-vector rotational dipolar rates, exponential scalar-contact
fluctuation rates, and `MotionBasedCWModel` for independent rate composition.
Spin prefactors and transition-frequency assignments are validated against
independent two-spin transition calculations; FFHS also has its diffusion-integral
audit. See [the implementation guide](../PythonSpinDynamics/docs/overhauser_dnp.md).

The E-stage two-spin reference now provides finite-temperature Gibbs recovery,
explicit cross-transition jumps, coherent microwave driving, trace-preserving
Lindblad propagation, stationary solves, and pair-motion rate conversions.
It has a 1:1 electron/nucleus population mapping; bulk dilute-solution rates
cannot be inserted without additional reservoir number-density information.

The F-stage resolved-line saturation model now tracks six 14N (or four 15N)
populations, adjacent nitrogen relaxation and concentration-dependent Heisenberg
population exchange, with an illustrative TEMPOL/water example. It uses a
high-temperature factorized thermal state and supplied electron/exchange rates;
it does not calculate exchange coherence transfer or line coalescence.
No independent experimental enhancement dataset has been used to calibrate or
validate the complete model. A many-nucleus/dilute-solution dynamics mapping
and experimental saturation/ELDOR calibration remain future work.

## Recommendation

Implement a dedicated `spin_dynamics.dnp` package in PythonSpinDynamics, starting
with liquid-state continuous-wave (CW) Overhauser DNP for spin-1/2 nuclei with
positive gyromagnetic ratio. Provide a transparent factor/rate calculation and
nuclear buildup before attempting a general driven electron–nuclear solver.
Use water/proton ODNP near X band as the first experimental benchmark, with
explicitly supplied effective electron saturation. Keep scalar relaxation in the
rate-level interface so that carbon-13 is a natural extension.

The small enhancement formula is straightforward. The substantial work is in
spectral-density normalization, thermal recovery, saturation, and independent
validation. Distinguish a model that accepts measured relaxation parameters from
one that predicts absolute rates from concentration and molecular motion.

## Repository baseline at the initial survey

| Existing component | Reuse | Missing physics or constraint |
| --- | --- | --- |
| `PythonSpinDynamics/src/spin_dynamics/nano_mr/high_resolution.py`: `DNPModel`, `dnp_polarization` | Result shape and exponential pump/recovery behavior | Enhancement and buildup time are inputs; polarization is clipped. No Overhauser rates, coupling factor, leakage, or saturation calculation. |
| `PythonSpinDynamics/src/spin_dynamics/esr/hyperfine.py`: `NuclearSite`, `ElectronNuclearSystem`, embedded operators and Hamiltonians | Small electron–nuclear reference systems, including nitrogen hyperfine structure | Static isotropic hyperfine coupling is not itself a stochastic cross-relaxation mechanism. |
| `PythonSpinDynamics/src/spin_dynamics/esr/pulsed.py` | Rotating-frame and microwave nutation conventions | `equilibrium_density` is an independently normalized, trace-zero high-temperature state; it does not preserve the absolute electron/nuclear thermal polarization ratio. |
| `PythonSpinDynamics/src/spin_dynamics/esr/relaxation.py` | Column-major Liouville conventions | Phenomenological population damping equalizes populations; it does not restore a finite-temperature polarized electron reservoir. |
| `PythonSpinDynamics/src/spin_dynamics/relaxation.py`: `RedfieldDipolarRelaxationModel` | Spin matrices, secular decomposition, dissipator patterns | Single target-spin covariance model with bath degrees of freedom removed. It cannot track electron saturation and electron–nuclear polarization transfer. |
| Same module: `spectral_density_lorentzian`, `IsotropicLiquidMotionalAveraging` | Explicit rotational model and unit conventions | Uses `2*tau/(1+(omega*tau)**2)`; not the force-free hard-sphere translational spectrum needed for standard water ODNP. |
| `PythonSpinDynamics/src/spin_dynamics/workflows/relaxation.py` | Dataclass results and sweep conventions | NMRD/BPP self-relaxation alone does not supply the electron–nuclear cross-relaxation rate. |

Source searches found no dedicated Overhauser/ODNP implementation in the Python
or MATLAB source trees. MATLAB is described in the project roadmap as a frozen
reference; Python should be the implementation target. The generic nano-MR DNP
helper should remain backward compatible, with an adapter added after the new
model can supply physically calculated enhancement and time constants.

## Theory to implement

### Physical mechanism and steady state

Microwave saturation changes electron populations; stochastic modulation of
electron–nuclear hyperfine interactions transfers polarization through unequal
zero- and double-quantum relaxation pathways. A static hyperfine Hamiltonian plus
independent phenomenological T1/T2 damping is insufficient to model this process.
The original proposal concerned conduction electrons in metals [1], with early
experimental verification in metals and sodium/ammonia [2]. Liquid radical
solutions are the most tractable first target for this repository.

For a positive-gamma spin-1/2 nucleus, use the convention

\[
E = P_I^{\mathrm{ss}}/P_I^0
  = 1-\xi f s\,r,\qquad r=|\gamma_e|/\gamma_I.
\]

`E` includes the thermal baseline: microwaves off means `E=1`; negative values
mean inverted nuclear polarization. Some publications instead label `1-E` as the
enhancement-related observable. Make this distinction explicit in every result
and benchmark [7]. With the free-electron/proton ratio approximately 658.2,
ideal extreme-narrowing dipolar coupling gives `E ≈ -328.1`, while ideal scalar
coupling gives `E ≈ +659.2`. These are limiting checks, not practical predictions.

Define nuclear paramagnetic self-relaxation `R_rho`, cross-relaxation `R_sigma`,
and nonparamagnetic relaxation `R_10` in inverse seconds:

\[
R_1=R_{10}+R_\rho,\quad
\xi=R_\sigma/R_\rho,\quad
f=R_\rho/R_1=1-T_1/T_{10}.
\]

For radical concentration `C` in mol/L, an empirical relaxivity interface uses
`R_rho=k_rho*C`, `R_sigma=k_sigma*C`. Relaxivities then have units
`s^-1 M^-1`. Require matched field/temperature conditions when deriving leakage
from measured T1 values. If `R_rho=0`, define zero transfer directly rather than
dividing by zero to calculate xi.

### Coupling factors and motion

For positive electron/nuclear frequency magnitudes, define
`J_plus=J(omega_e+omega_I)`, `J_minus=J(omega_e-omega_I)`, and
`J_I=J(omega_I)`. The conventional dipolar rate combination is [5]

\[
R_\rho^d=A_d[6J_-+3J_I+J_+],\qquad
R_\sigma^d=A_d[6J_--J_+].
\]

This recovers `xi=1/2` when all three spectral densities are equal. A pure
isotropic scalar fluctuating interaction contributes `R_rho^s=q` and
`R_sigma^s=-q`, hence `xi=-1`. Add dipolar and scalar **rates** before taking
their ratio. Do not average coupling factors without rate weights. The amplitude
`A_d` must be derived for the chosen spectral-density and spin normalization;
it must not silently absorb concentration or units in the public API.

Implementation review correction: with positive frequency magnitudes and
`H_Z=+omega_e*S_z-omega_I*I_z`, the dipolar DQ spin transition has the
**difference** frequency and ZQ has the **sum** frequency. This corrects the
original scope's assignment, which mixed signed and magnitude conventions.
The implementation verifies the weights and frequencies directly from the
two-spin dipolar Hamiltonian, independently of the closed-form rate formula.

For water interacting with dissolved nitroxides, support the Hwang–Freed
force-free hard-sphere (FFHS) translational model [4]. Its diffusion time is
conventionally `tau_D=d^2/(D_water+D_radical)`, with closest approach `d` and
relative diffusion coefficient. Specify this definition because alternate
correlation-time conventions exist. A rotational Lorentzian remains useful for
bound vectors and analytic checks, but fitting water with it changes the inferred
motion. FFHS is still a model: restricted hydration and pair correlations may
require extensions.

Before coding absolute rates, rederive the Fourier convention, radial/concentration
prefactor, and the published FFHS expression from the original reference.
Search-extracted equations can misrender powers and signs. Validate the spectral
shape separately from its absolute scale; agreement in xi cannot validate an
amplitude that cancels from numerator and denominator.

### Electron saturation

Accept an explicit effective `s` in `[0,1]` first. For a homogeneous isolated
electron transition, a later Bloch approximation is

\[
s=\frac{\omega_1^2T_{1e}T_{2e}}
 {1+\Delta\omega^2T_{2e}^2+\omega_1^2T_{1e}T_{2e}}.
\]

Define `omega_1` as the rotating-frame electron nutation angular frequency;
converting a linearly polarized laboratory B1 introduces a factor-of-two issue.
An empirical alternative is `s(p)=s_max*p/(p+p_half)`, with both powers in the
same units and calibration attached to the experiment.

Nitroxide ODNP depends on saturation across the electron/nitrogen hyperfine
populations. Saturating one observed line does not establish total saturation.
Nitrogen relaxation and intermolecular Heisenberg exchange redistribute it;
ELDOR measurements provide an independent route to the effective factor [6].
Expose `s_max` or measured `s` rather than assume unity for TEMPOL/TEMPONE.

### Dynamics, temperature, and field changes

With electron saturation held constant and a single nuclear reservoir,

\[
P_I(t)=P_I^{\mathrm{ss}}+
       [P_I(0)-P_I^{\mathrm{ss}}]e^{-R_1t}.
\]

Microwaves off restores `P_I^0` with the sample's microwave-off T1, which still
includes radical-induced relaxation. It is not generally the radical-free T10.
Solomon's coupled equations motivate a later model when electron dynamics cannot
be eliminated; a coupled pair need not have a single exponential transient [3].

The quantitative water-ODNP workflow must allow power-dependent measured
`T1(p)` and radical-free `T10(p)`, together with temperature/saturation metadata.
Microwave heating can otherwise masquerade as changing transfer efficiency;
Franck et al. describe the corrections and relaxation-based analysis [8].
For forward modeling at an assigned temperature, compute thermal nuclear
polarization consistently rather than multiplying arbitrary normalized ESR and
NMR density matrices.

If polarization and detection fields differ, propagate absolute polarization
through transfer and normalize by thermal polarization at the detection field.
At common temperature and high temperature, negligible transfer loss gives
`E_detect ≈ E_polarize*B_polarize/B_detect`, not `E_polarize`. Include relaxation
during transfer [7].

The first release should state weak-coupling/Zeeman, high-temperature,
spin-1/2, homogeneous-reservoir assumptions. Do not extend it to cryogenic large
polarization by clipping its result: use a thermally consistent population or
master-equation model. Negative-gamma nuclei require a separate derivation and
tests for polarization signs and sum/difference transition assignments.

## Proposed architecture and deliverables

Proposed names are illustrative, not existing APIs:

```text
src/spin_dynamics/dnp/
    __init__.py
    overhauser.py       # factors, relaxation rates, steady state, buildup
    spectral.py         # FFHS translational and named rotational models
    saturation.py       # effective/power/Bloch saturation models
    dynamics.py         # later coupled population/master-equation backend
src/spin_dynamics/workflows/dnp.py
tests/test_overhauser_dnp.py
tests/test_dnp_spectral.py
examples/plot_overhauser_dnp.py
docs/overhauser_dnp.md
validation/experimental/<benchmark and provenance files>
```

Use frozen parameter/result dataclasses, NumPy broadcasting, explicit units in
field names, and optional SciPy for fitting or matrix propagation. Keep the
factor-level core NumPy-only, consistent with the package's base dependency.
Results should expose `E`, `P_thermal`, `P_steady`, xi, f, s, each rate, T1,
spectral model, and validity assumptions. A sweep should report field/power/time
axes explicitly. Avoid a single opaque enhancement number.

Offer two distinct input paths: measured factors/rates for defensible prediction
with experimentally supplied inputs, and a microscopic spectral model that
calculates rates. Reject conflicting inputs. Integrate into nano-MR and pulse
workflows only through documented conversions of absolute polarization or
deviation-density normalization.

## Work packages and planning estimates

Estimates below are engineering judgments for one developer familiar with this
codebase, with scientific review and usable literature data available. They are
working days, not a delivery commitment; experimental data access may add delay.

| Work package | Concrete completion criterion | Effort |
| --- | --- | --- |
| A. Conventions and benchmark selection | Derivation note fixes signs, units, spectral conventions, thermal normalization, and one accessible reference dataset | 1–2 days |
| B. Factor/rate core and buildup | Explicit xi/f/s or measured-rate inputs, steady state, pumping/recovery, result dataclasses, analytic tests | 2–3 days |
| C. Motion-based model | Rotational and FFHS spectra, coupling factors, separately verified absolute prefactors, scalar/dipolar rate composition | 3–5 days |
| D. CW workflows and evidence | Field/power sweeps, effective saturation, heating-aware measured inputs, one benchmark, example/docs/API exports, CI | 3–5 days |
| E. Explicit electron–nuclear dynamics | Thermal recovery, driven two-spin population/Liouville reference, consistent cross transitions and rate-to-reservoir mapping | Additional 5–10 days |
| F. Nitroxide saturation and advanced applications | Nitrogen hyperfine populations, exchange mixing, comparison to saturation/ELDOR data; optional fitting and transfer workflows | Additional 5–10 days |

A–D make a scientifically useful first release: approximately **9–15 working
days (2–3 weeks)**. Including E–F gives roughly **4–7 weeks**. A calculator that
only accepts xi/f/s is a smaller milestone, not the complete motion-based feature.
Zero-field pumping, quadrupolar targets, insulating-solid mechanisms, spatial
spin diffusion, transport, and microwave electromagnetic/thermal simulation
need separate scopes.

## Verification and acceptance

1. **Analytic physics:** no microwaves/no radical/no cross-relaxation returns
   `E=1`; test zero-rate degeneracies without NaNs. Recover xi=1/2 for dipolar
   extreme narrowing, xi=-1 for scalar-only rates, cancellation of scalar and
   dipolar transfer, leakage limits, detuning, and pump/recovery initial conditions.
2. **Independent rates:** validate FFHS low/high-frequency behavior and several
   finite-frequency reference values against the primary derivation or numerical
   diffusion calculation. Validate absolute R1/relaxivity separately from xi.
   A second implementation of the same formula alone is not experimental evidence.
3. **Dynamics:** later require trace preservation, Hermiticity, nonnegative
   populations, correct microwave-off thermal state, and steady-state agreement
   with the rate model in its valid weak-coupling limit. Check reservoir number
   densities when moving from a single pair to solution ensembles. Use robust
   linear solves/exponentials rather than relying on generic eigendecomposition
   near repeated or defective eigenvalues.
4. **Literature benchmark:** begin with a stated-temperature water/nitroxide
   enhancement-versus-power or field dataset with T1/T10 and effective saturation.
   Separate supplied calibration inputs from predicted observables. Record DOI,
   figure/table, digitization method, uncertainty, and any fitted parameters;
   do not claim independent validation by fitting all three factors to E.
5. **Integration:** smoke representative analytic cases and exercise the new
   example. Update API documentation and the validation evidence registry.
   The current smoke workflow additionally checks
   `python docs/generate_validation_matrix.py --check`, beyond the three steps
   mentioned in AGENTS.md. Run the actual workflow's lint, API-generation diff,
   evidence check, and smoke suite before committing/pushing implementation.
   Use WSL `Ubuntu-24.04` for SciPy/Matplotlib work as required by project memory.

## Literature map and boundaries

1. **Overhauser (1953), “Polarization of Nuclei in Metals,” Physical Review 92,
   411.** Original mechanism and polarized electron reservoir.
   [Publisher](https://journals.aps.org/pr/abstract/10.1103/PhysRev.92.411).
2. **Carver and Slichter (1956), “Experimental Verification of the Overhauser
   Nuclear Polarization Effect,” Physical Review 102, 975.** Early tests showing
   the practical roles of incomplete saturation and competing relaxation.
   [Publisher](https://journals.aps.org/pr/abstract/10.1103/PhysRev.102.975).
3. **Solomon (1955), “Relaxation Processes in a System of Two Spins,” Physical
   Review 99, 559.** Coupled longitudinal dynamics and transient effects.
   [Publisher](https://journals.aps.org/pr/abstract/10.1103/PhysRev.99.559).
4. **Hwang and Freed (1975), “Dynamic effects of pair correlation functions on
   spin relaxation by translational diffusion in liquids,” JCP 63, 4017–4025.**
   Translational spectral-density theory and closest-approach boundary condition.
   [Author-hosted paper](https://www.acert.cornell.edu/PDFs/JChemPhys63_4017_1975.pdf),
   [DOI](https://doi.org/10.1063/1.431841).
5. **Cheng et al. (2014), “Cholesterol enhances surface water diffusion of
   phospholipid bilayers,” JCP 141, 22D513.** Concrete ODNP spectral-density and
   dipolar coupling-factor equations, Eqs. 3–4.
   [Paper](https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.4897539/13908749/22d513_1_online.pdf).
6. **Türke and Bennati (2011), “Saturation factor of nitroxide radicals in liquid
   DNP by pulsed ELDOR experiments,” PCCP 13, 3630–3633.** Effective saturation
   measurements rather than single-line assumptions.
   [Author manuscript](https://pure.mpg.de/rest/items/item_885578_6/component/file_1478767/content),
   [DOI](https://doi.org/10.1039/C0CP02126A).
7. **Keller and Maly (2021), “Overhauser dynamic nuclear polarization
   (ODNP)-enhanced two-dimensional proton NMR spectroscopy at low magnetic
   fields,” Magnetic Resonance 2, 117–128.** Practical low-field NMR integration,
   enhancement conventions and field-transfer penalty.
   [Open paper](https://mr.copernicus.org/articles/2/117/2021/).
8. **Franck, Pavlova, Scott and Han (2013), “Quantitative cw Overhauser effect
   dynamic nuclear polarization for the analysis of local water dynamics,”
   Progress in NMR Spectroscopy 74, 33–56.** Quantitative relaxometry and heating
   corrections. This methodological review supplies the analysis framework;
   experimental validation should cite its specific data or an underlying study.
   [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0079656513000629),
   [Open manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC3798041/).
9. **Liu et al. (2017), “One-thousand-fold enhancement of high field liquid
   nuclear magnetic resonance signals at room temperature,” Nature Chemistry 9,
   676–680.** High-field carbon-13 motivates a scalar-capable interface rather
   than a proton/dipolar-only architecture.
   [Author repository](https://flore.unifi.it/handle/2158/1107248),
   [DOI](https://doi.org/10.1038/nchem.2723).
10. **Levien et al. (2024), “Overhauser enhanced liquid state nuclear magnetic
    resonance spectroscopy in one and two dimensions,” Nature Communications.**
    Demonstration at 9.4 T with high-resolution and multidimensional NMR;
    high-field liquid ODNP should not be dismissed categorically.
    [Open paper](https://www.nature.com/articles/s41467-024-50265-5).
11. **“Zero-Field Overhauser Dynamic Nuclear Polarization” (2025), ACS Omega 10,
    37611–37616.** Circularly polarized RF selectively drives mixed hyperfine
    transitions near zero field. Relevant to future ZULF work, requiring a
    different model rather than setting B0=0 in the initial enhancement API.
    [Publisher](https://pubs.acs.org/doi/10.1021/acsomega.5c03912).

This was a targeted theory and implementation survey, not an exhaustive
bibliometric review. Some publisher full texts were inaccessible; accessible
papers, author manuscripts, abstracts, and indexed equations were used. No
experimental dataset was digitized and no numerical benchmark was reproduced
during this scoping task. Verifying the original FFHS prefactors and securing a
benchmark with enough calibration metadata remain explicit first tasks.
