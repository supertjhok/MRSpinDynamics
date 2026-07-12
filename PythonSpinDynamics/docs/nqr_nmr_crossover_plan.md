# NQR-to-NMR Crossover Modeling Plan

## Purpose

This document records the analytical and implementation groundwork for modeling
the continuous transition from zero-field nuclear quadrupole resonance (NQR),
through Zeeman-perturbed NQR, to quadrupole-perturbed NMR in
PythonSpinDynamics. The intermediate regime, where the Zeeman and quadrupolar
interactions have comparable strengths, is the principal target.

## Governing dimensionless parameters

The static NQR-to-NMR crossover is governed by

\[
r = \frac{\nu_L}{\nu_Q}
  = \frac{|\gamma| B_0}{\nu_Q}
  = \frac{B_0}{B_Q},
\qquad
B_Q \equiv \frac{\nu_Q}{|\gamma|}.
\]

Here, \(\nu_L\) is the Larmor frequency and \(\nu_Q\) is the chosen
quadrupolar-frequency scale. In conventional magnetic-resonance notation,
\(B_1\) is the oscillating RF field. Consequently, \(B_0/B_1\) controls RF
selectivity and pulse behavior, but it does not determine whether the static
Hamiltonian is NQR-like or NMR-like. An effective quadrupolar field should be
called \(B_Q\), rather than \(B_1\), to avoid this ambiguity.

RF excitation introduces a second independent parameter,

\[
s = \frac{|\gamma| B_1}{\nu_Q}.
\]

Both \(r\) and \(s\) matter for pulsed simulations, but only \(r\) identifies
the static crossover regime.

## Static Hamiltonian

For an isolated quadrupolar nucleus in the electric-field-gradient (EFG)
principal-axis system, the relevant Hamiltonian may be written as

\[
\frac{\mathcal H_0}{h}
=
-\gamma\,\mathbf B_0\mathbin{\cdot}\mathbf I
+
\frac{C_Q}{4I(2I-1)}
\left[
3I_z^2-I(I+1)+\eta(I_x^2-I_y^2)
\right].
\]

The numerical prefactor depends on whether the public parameter is the
quadrupole coupling constant \(C_Q=eQV_{zz}/h\), a conventional \(\nu_Q\), or
the lowest zero-field transition frequency. PythonSpinDynamics currently
defines `QuadrupolarSite.quadrupole_frequency_hz` as the lowest axial
zero-field transition frequency. That convention must remain explicit because
factor-of-two, -three, and -six convention errors would give incorrect
crossover fields.

For a complete high-field quadrupolar-NMR description, the Zeeman term should
eventually include isotropic chemical shift and chemical-shielding anisotropy,

\[
\mathcal H_Z =
-\gamma\,\mathbf B_0\mathbin{\cdot}
(\mathbf 1-\boldsymbol\sigma)\mathbin{\cdot}\mathbf I.
\]

## Physical regimes

### Quadrupole-dominated: \(r \ll 1\)

This is zero-field or Zeeman-perturbed NQR. The EFG principal axes supply the
approximate quantization frame. Half-integer spins contain zero-field Kramers
doublets, so weak magnetic fields require degenerate perturbation theory.
Ordinary non-degenerate perturbation theory is not reliable within those
doublets.

### Crossover: \(r \sim 1\)

Neither interaction is perturbative. Eigenstates change character, avoided
crossings appear, nominally forbidden transitions can acquire intensity, and a
fixed magnetic quantum-number label loses meaning. Exact diagonalization of
the complete Hamiltonian is the natural treatment.

This regime is computationally inexpensive for a single quadrupolar nucleus;
the difficulty lies in correctly treating state identity, degeneracies,
selection rules, polarization, excitation, detection, and relaxation.

### Zeeman-dominated: \(r \gg 1\)

This is quadrupole-perturbed NMR. The quantization axis approaches the static
field direction. First-order quadrupolar satellite shifts and second-order
central-transition shifts must emerge from the exact solution. For
half-integer spins, the central transition has no first-order quadrupolar
shift, and its leading quadrupolar broadening scales approximately as
\(\nu_Q^2/\nu_L\).

## Analytical and numerical treatments

Closed-form treatments exist for selected spins and geometries:

- Classical zero-field and perturbative limits are developed in R. V. Pound,
  "Nuclear Electric Quadrupole Interactions in Crystals," *Physical Review*
  **79**, 685 (1950), DOI: <https://doi.org/10.1103/PhysRev.79.685>, and
  M. Bloom, E. L. Hahn, and B. Herzog, "Free Magnetic Induction in Nuclear
  Quadrupole Resonance," *Physical Review* **97**, 1699 (1955), DOI:
  <https://doi.org/10.1103/PhysRev.97.1699>.
- R. B. Creel and D. A. Drabold derived an exact analytic solution for the
  spin-3/2 combined Zeeman-quadrupole Hamiltonian, *Journal of Molecular
  Structure* **111**, 85-90 (1983), DOI:
  <https://doi.org/10.1016/0022-2860(83)85101-1>.
- A. D. Bain developed an exact angular-momentum/Liouville-space treatment
  covering arbitrary spin and the complete NQR-to-NMR range, *Molecular
  Physics* **101**, 3163-3175 (2003), DOI:
  <https://doi.org/10.1080/00268970310001626298>.
- Bain and Khasawneh explicitly mapped spin-3/2 transition energies across the
  complete field range, *Concepts in Magnetic Resonance Part A* **22A**,
  69-78 (2004), DOI: <https://doi.org/10.1002/cmr.a.20013>.
- Bain later presented a unified Liouville-space view emphasizing that signal
  polarization differs between the NQR and NMR limits, *Magnetic Resonance in
  Chemistry* **55**, 198-205 (2017), DOI:
  <https://doi.org/10.1002/mrc.4418>.
- Ansari and Sauer give compact exact spin-3/2 eigenstates, transition
  frequencies, magnetic-dipole matrix elements, and Rabi coefficients when the
  static field is aligned with an EFG principal axis, *Physical Review B*
  **110**, 214422 (2024), DOI:
  <https://doi.org/10.1103/PhysRevB.110.214422>.

The analytical solutions are most valuable as validation targets. Direct
Hermitian diagonalization is simpler and more robust for production use.

The closest practical precedent is QUEST:

- F. A. Perras, C. M. Widdifield, and D. L. Bryce, "QUEST--QUadrupolar Exact
  SofTware: A fast graphical program for the exact simulation of NMR and NQR
  spectra for quadrupolar nuclei," *Solid State Nuclear Magnetic Resonance*
  **45-46**, 36-44 (2012), DOI:
  <https://doi.org/10.1016/j.ssnmr.2012.05.002>.

QUEST diagonalizes the full Zeeman-quadrupole Hamiltonian, calculates
transition probabilities from the eigenvectors, simulates all transitions
simultaneously, and powder-averages spectra over the full \(\nu_L/\nu_Q\)
range.

An important experimental crossover benchmark is:

- E. A. Donley et al., "Nuclear quadrupole resonances in compact vapor cells:
  The crossover between the NMR and the nuclear quadrupole resonance
  interaction regimes," *Physical Review A* **79**, 013420 (2009), DOI:
  <https://doi.org/10.1103/PhysRevA.79.013420>; open manuscript:
  <https://arxiv.org/pdf/0810.3928>.

Donley et al. traced spin-3/2 resonance branches continuously through the
crossover. They confirmed the failure of perturbation theory when the two
interactions are comparable and observed strong field dependence in line
amplitudes and decay rates near crossings.

## Existing PythonSpinDynamics foundation

The repository already contains much of the required static machinery:

- exact construction of \(H_Q+H_Z\) for an arbitrary static-field vector in
  `spin_dynamics.nqr.hamiltonians`;
- single-site and batched powder-grid Hermitian diagonalization;
- magnetic-dipole transition vectors calculated from the eigenstates;
- joint static-field/RF-field orientation sampling for powders; and
- full \((2I+1)\)-level density-matrix propagation for pulsed NQR.

The present static-spectrum API is intentionally limited to weak fields. It
warns when \(|\gamma B_0|/\nu_Q\) exceeds a configurable threshold, selects
transitions only within a window around chosen zero-field lines, and assigns
an excitation weight proportional to
\(w|\hat{\mathbf B}_1\cdot\mathbf I_{ij}|^2\). These choices should remain
available for Zeeman-perturbed NQR, but they must not constrain the general
crossover API.

The full pulse solver currently uses a single-band winding-number rotating-wave
approximation. Its documented scope is a carrier addressing one transition
band in zero-field or weak-Zeeman NQR; it is not yet a general multiband
crossover solver.

## Required modeling pieces

### 1. Regime-independent static spectrum

Add a general exact spectrum API that:

- enumerates all eigenvalue differences without a weak-field warning;
- does not select lines by proximity to zero-field transitions;
- calculates and reports \(r=\nu_L/\nu_Q\);
- accepts single-crystal and powder orientations; and
- preserves the existing weak-field helper as a specialized convenience API.

### 2. Physical transition intensities

For equilibrium absorption, use

\[
I_{ij}\propto
(p_i-p_j)
\left|
\langle i|\hat{\mathbf B}_1\mathbin{\cdot}\mathbf I|j\rangle
\right|^2,
\qquad
p_i=\frac{\exp(-E_i/kT)}{Z}.
\]

A detected FID may additionally depend on the preparation and receive
operator. The API should distinguish an excitation-weighted stick spectrum
from a physically prepared and detected signal. It should support complex RF
polarization vectors so that linear, circular, and elliptical fields can be
represented.

### 3. Degeneracy-safe transition handling

At zero field, eigenvectors within a degenerate subspace are not unique.
Individual transition matrix elements can change under an arbitrary unitary
rotation inside that subspace even though the observable spectrum cannot.
Zero-field intensities should therefore be calculated with degenerate-subspace
projectors or summed over complete degenerate manifolds.

During field sweeps, eigenstates should be tracked using eigenvector or
subspace overlap rather than sorted energy index alone. Transition labels
should describe adiabatically tracked branches or observable character rather
than assume a globally valid magnetic quantum number.

### 4. Powder averaging

For every crystallite, rotate the fixed laboratory \(B_0\), transmit \(B_1\),
and receive axes into the EFG principal-axis frame. When \(\eta\ne0\), both
polar and azimuthal orientations matter. Non-coincident shielding and EFG
tensors will require full Euler-angle sampling.

Use a powder grid with demonstrable convergence, such as ZCW, Lebedev, or an
Alderman-Solum-Grant-style scheme. Preserve explicit geometry rather than
assuming that transmit and receive fields are always perpendicular to the
instantaneous eigenstate quantization axis.

### 5. General RF dynamics

Static spectra require only exact diagonalization. Pulsed crossover simulations
need one or more of:

- direct laboratory-frame propagation of
  \(H_0+H_1(t)\), as the reference implementation;
- a carefully constructed multilevel rotating-frame/RWA solver; or
- Floquet propagation for long or strong periodic irradiation.

Large \(s\), overlapping transition bands, or near-degenerate transitions can
invalidate a selective two-level pulse model even when the static eigensystem
is exact.

### 6. Equilibrium and relaxation

Equilibrium must be derived from the complete \(H_Q+H_Z\), rather than from an
assumed pure-Zeeman or pure-quadrupolar polarization. This is one reason the
observable intensity changes through the crossover.

Phenomenological broadening is sufficient for the first static-spectrum
milestone. A later physical relaxation model should transform the appropriate
fluctuating EFG and magnetic operators into the instantaneous eigenbasis. Near
degeneracies and avoided crossings, a fully secular transition-by-transition
model may fail, so a nonsecular Liouvillian/Redfield treatment is the robust
endpoint.

A general linear-response formulation is attractive:

\[
\mathcal L\rho=-i[H_0,\rho]+\mathcal R\rho,
\]

\[
\chi(\omega)\propto
\operatorname{Tr}\left[
D(i\omega-\mathcal L)^{-1}
\bigl(-i[V,\rho_{\mathrm{eq}}]\bigr)
\right],
\]

where \(V\) is the excitation operator and \(D\) is the detection operator.
This unifies transition frequencies, polarization, detection, relaxation, and
near-degenerate coherences.

### 7. Field-sweep history

A set of independent equilibrium spectra evaluated at successive fields is not
the same as physically carrying a prepared state through a changing field. If
the intended experiment ramps \(B_0\), the model must also specify ramp rate,
relaxation, and adiabatic or Landau-Zener passage through avoided crossings.
This should be a separate dynamic workflow built after the static crossover
spectrum is validated.

### 8. Later extensions

After the isolated-site model is established, useful extensions include:

- isotropic shift and chemical-shielding anisotropy;
- non-coincident shielding and EFG tensors;
- dipolar and scalar coupling to neighboring spins;
- distributions in \(C_Q\), \(\eta\), shielding, and local magnetic field;
- sample rotation or magic-angle spinning; and
- field-dependent microscopic relaxation.

## Validation hierarchy

Implementation should be validated in the following order:

1. Reproduce exact zero-field spin-1 and spin-3/2 NQR limits, including
   asymmetry dependence.
2. Reproduce closed-form spin-3/2 aligned-field eigenvalues, transition
   frequencies, matrix elements, and Rabi coefficients from Ansari and Sauer.
3. Recover classical weak-field Zeeman-perturbed NQR splittings.
4. Recover high-field first- and second-order quadrupolar-NMR limits, including
   the first-order-free central transition for half-integer spins.
5. Sweep approximately
   \(10^{-3}\le\nu_L/\nu_Q\le10^3\) and compare exact results against an
   independent reference implementation.
6. Reproduce selected QUEST powder and overtone examples.
7. Compare spin-3/2 field/angle transition trajectories with Donley et al.
8. Test invariance under unitary rotations within degenerate subspaces.
9. Test oscillator-strength sum rules, spectral symmetry under
   \(B_0\rightarrow-B_0\), trace/Hermiticity preservation, and powder-grid
   convergence.
10. Add and validate pulsed crossover simulations.
11. Add field-dependent and nonsecular relaxation only after the coherent
    response is stable.

## Recommended development milestones

### Milestone 1: exact static crossover spectrum

Build a single-site, all-transition exact spectrum with physical population
weights, arbitrary transmit/receive geometry, degeneracy-safe zero-field
handling, and converged powder averaging.

### Milestone 2: continuous branch visualization

Add overlap-based state and transition tracking across a logarithmic field
sweep. Plot energy levels, transition frequencies, intensities, eigenstate
character, and avoided crossings versus \(\nu_L/\nu_Q\).

### Milestone 3: laboratory-frame pulse reference

Implement direct time-dependent propagation for selected spin-1 and spin-3/2
systems. Use it to determine where the existing selective and single-band RWA
models remain accurate.

### Milestone 4: efficient multiband dynamics

Introduce a multilevel rotating-frame or Floquet route, benchmarked against the
laboratory-frame reference over field, RF amplitude, orientation, and
detuning.

### Milestone 5: crossover relaxation and field ramps

Add nonsecular relaxation near crossings and explicit field-ramp workflows,
with separate equilibrium-spectrum and history-dependent APIs.

## Main conclusion

PythonSpinDynamics already solves the essential static eigenproblem correctly.
The next step should not be a new perturbative Hamiltonian. It should be a
regime-independent spectroscopy layer around the existing exact eigensolver,
with degeneracy-safe intensities, full laboratory geometry, equilibrium
polarization, and controlled powder averaging. A direct laboratory-frame pulse
solver should then serve as the reference for extending dynamics through the
intermediate regime.

## Implementation status

Completed through the RF-validation increment:

- regime-independent static `H_Q + H_Z` spectra with Boltzmann weighting and
  complex transmit/receive geometry;
- degeneracy-safe Kramers-manifold intensities;
- overlap- and phase-tracked field sweeps with transition responses and
  magnetic-quantum-number expectation values;
- analytical zero-, intermediate-, and high-field tests, field-reversal
  symmetry, and circular-polarization selection tests;
- a repository-parameterized NaNO2 `14N` / NaClO3 `35Cl` crossover plot; and
- a change-aware test selector with a short constant-time smoke tier.
- optional powder-averaged field-frequency maps with explicit SO(3) grid
  convergence; and
- an exact piecewise-constant laboratory-frame RF propagation reference with
  no rotating-wave approximation.
- quantitative single-band RWA error maps over static-interaction and RF-field
  ratios, including nearest-line isolation diagnostics; and
- finite-sideband Floquet propagation for monochromatic linear RF, validated
  against direct laboratory-frame propagation in the crossover regime.
- exact trace-one Gibbs equilibrium from the complete field-dependent static
  Hamiltonian;
- a degeneracy-safe completely-positive Gibbs-reset/dephasing model for robust
  relaxation through crossings; and
- a finite-temperature Davies/Lindblad model with magnetic and EFG channels,
  detailed balance, grouped degenerate Bohr frequencies, and explicitly
  field-dependent Lorentzian spectral filtering.
- a completely-positive unified-GKLS extension that groups unresolved Bohr
  frequencies and retains their nonsecular coherence-transfer terms, with an
  explicit Gibbs-stationarity diagnostic; and
- an exact-Floquet-pulse SLSE center-point workflow with static-field
  dissipation; and
- a waveform-level extension that samples each echo, coherently averages
  crystallites before receiver processing, applies an explicit finite-bandwidth
  receiver, and estimates the train by matched projection onto the first echo.

### Powder-waveform audit

The echo-center observable is accepted only for the narrow zero-field line. It
is not an experimental decay estimator once the static field broadens the
powder spectrum. Nonzero-field calculations must instead converge the complete
acquired waveform and its receiver-derived amplitude.

The first waveform implementation established the following infrastructure:

- nested low-discrepancy SO(3) orientation sequences;
- a frequency-slice selector that reports the retained full-powder weight;
- exact-Floquet and phase-consistent RWA pulse backends;
- deterministic orientation-level parallel execution via `num_workers`; and
- prefix reuse for convergence estimates, avoiding duplicate propagation.

For production powder runs, a chunked process backend and reduce-only return
mode avoid transferring per-orientation density-matrix diagnostics. On the
small four-echo benchmark used during this audit, serial, four-thread, and
four-process execution took 8.22 s, 7.55 s, and 4.28 s, respectively. The full
30-echo 8192-candidate calculation fell from 334 s with retained threaded
results to 82 s with four reduce-only processes.

The first audit exposed severe aliasing when an 800 microsecond waveform was
sampled at 65 points but paired with a nominal 200 kHz receiver. The receiver
now rejects bandwidths above the acquisition sampling rate. The validated
default is a 100 microsecond/129-point acquisition; doubling to 257 points
changes the matched train by less than 3%. With a 200 kHz receiver, frequency-slice reporting,
and nested active ensembles of 1024--2048 orientations, the 30-echo matched
trains converge to 1.5% (perturbed NQR), 1.9% (intermediate), and 0.81%
(quadrupolar NMR). The zero-field matched train fits 328 ms, consistent with the
measured 332 +/- 23 ms lifetime. A resonance-manifold quadrature remains a
valuable efficiency improvement, and independent sensitivity checks versus
time step, receiver bandwidth, and selected spectral width remain required for
quantitative experimental prediction.

The next numerical milestone is the resonance-manifold efficiency improvement
and receiver/spectral-width sensitivity audit described above. Arbitrary shaped
RF is already available through the laboratory-frame reference; an efficient
piecewise-Floquet envelope solver is a possible later optimization rather than
a correctness prerequisite.

### Field-sweep histories

The history-dependent physics milestone now has a separate
`simulate_field_sweep_history` API. It carries a single density matrix through
an arbitrary piecewise-linear vector-field path, including decreasing fields,
reversals, and rotations. Each interval uses midpoint propagation of the full
`H_Q + H_Z` Liouvillian and can include any instantaneous field-dependent
relaxation generator. The returned trajectory includes the density matrix,
instantaneous-eigenbasis populations and coherence norm, spin expectation,
energy expectation, Gibbs-state deviation, and a positivity diagnostic at every
user-supplied field node.

Validation covers three independent limits: a constant-field history agrees
with the existing fixed-field propagator, a slow relaxing ramp approaches the
instantaneous Gibbs state more closely than a fast ramp, and coherent rotating
vector-field ramps converge as midpoint substeps are doubled. The NaNO2 example
uses an up-and-down sweep from `nu_L/nu_Q = 0.02` to 3 at 300 K, starting from a
prepared quadrupolar ground state. It now separates three timescales: a 0.2 us
coherent nonadiabatic passage, a 0.2 ms coherent Hamiltonian-adiabatic passage
that drags the prepared polarization, and a 100 ms adiabatic passage with
relaxation that instead follows the nearly unpolarized room-temperature Gibbs
state. Its default 16-substep integration was checked against 32 substeps: the
maximum 0.2 ms trajectory density difference was 0.37%, while the 100 ms case
was 1.1% (the final slow-state difference was below 0.01%). These parameters
describe a phenomenological demonstration, not fitted NaNO2 field-cycling
kinetics.
