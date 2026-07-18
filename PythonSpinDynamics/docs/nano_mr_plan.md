# Defect-Spin Nanoscale Spectroscopy and MRI Plan

> **Status (2026-07-18): Phases 1-8 implemented.**
> The code now includes
> coordinate frames, defect Hamiltonians and presets, point-dipole geometry,
> phase-aware Ramsey/Hahn/CPMG/XY control, ideal and finite-width filter
> functions, addressed-qubit propagation, and effective optical photon-count
> readout, plus statistical/thermal/fixed-polarization nuclear baths, analytic
> layers and half-spaces, voxel densities, multi-isotope spectra, and Gaussian
> filter-overlap coherence; exact resolved clusters, nuclear RF, CW and
> two-block 2-D correlation spectroscopy; and seeded diffusion, drift,
> confinement, dipolar field correlations, and spectra; plus scanning-sensor
> and sensor-array nano-MRI forward models, depth profiles, nonnegative density
> reconstruction, sparse point localization, and uncertainty estimates.
> Phase 6.5 now adds correlated target/sensor fields, optical-cycle and SPAD
> shot records, and covariance-weighted inversion. Phase 7 adds clocked Qdyne,
> synchronized I/Q readout, separately budgeted coherence limits,
> sensor-memory correlation, coherent thermal chemical-shift/J spectra, and
> optional explicit DNP. Phase 8 now integrates the statistical-spectrum and
> Qdyne paths with the package experiment facade and Bayesian design layer.
> Calibration fitting, further detector variants, and microscopic memory/DNP
> dynamics remain staged.

## Literature assessment

The field contains several experimentally and computationally distinct
regimes. A useful simulation package must keep them separate.

In 2013, individual shallow diamond NV centers detected statistically
polarized proton ensembles in nanometre-scale volumes under ambient
conditions. Staudacher et al. reported a `(5 nm)^3` sample volume containing
approximately `10^4` nuclei, while Mamin et al. independently demonstrated
nanoscale NMR with an NV spin sensor [1, 2].

The technique then expanded to multiple isotopes, surface-layer thickness
measurements, and two-dimensional proton MRI [3, 4]. Correlation spectroscopy
and ancillary nuclear memories extended spectral resolution beyond the NV
electron coherence time. These developments enabled detection of multiple
nuclear species in individual surface-attached proteins and chemical-shift
resolution in zeptolitre samples [5, 6].

Qdyne and synchronized-readout methods separated frequency resolution from
sensor coherence for coherent signals [7, 8]. Ensemble-NV experiments also
detected thermally polarized NMR with chemical shifts and scalar couplings [9].
These ensemble experiments must not be conflated with statistically polarized
single-NV nano-NMR: their signal statistics, sample volumes, and dominant
resolution limits differ.

Exact and multidimensional protocols have reconstructed nuclear clusters
inside diamond. A 27-spin carbon-13 cluster was mapped with sub-angstrom
precision, and later work demonstrated parallel localization of more than 20
spins near shallow NVs [10, 11]. These are important algorithmic milestones,
but routine three-dimensional reconstruction of an arbitrary external molecule
remains an open frontier.

Liquids present a separate challenge because molecules diffuse through the
nanoscale detection volume. The standard short-correlation-time picture
predicts broadening, but theory and experiment show long power-law correlation
tails for dipolar detection geometries [12, 13]. Both trajectory simulations
and analytic spectral-density models are therefore needed.

For silicon carbide, the most relevant near-term sensor is the spin-1
PL6/divacancy-related center rather than a generic "NV in SiC." Peer-reviewed
work has demonstrated stable approximately 2-nm-deep PL6 centers, sensitivity
to external nuclear-spin noise, near-infrared optical operation, and favorable
surface chemistry [14]. A December 2025 preprint reports proton and fluorine
nano-NMR with an estimated `(3 nm)^3` detection volume [15]. It should remain
labelled preprint-level validation until a peer-reviewed result is available.

## Repository assessment

PythonSpinDynamics already provides:

- arbitrary-spin matrices in `spin_dynamics.nqr`;
- dense electron-nuclear and scalar-coupled Hamiltonians;
- Liouville propagation and phenomenological relaxation;
- ESR, DEER, ESEEM, HYSCORE, and ENDOR reference models;
- particle diffusion, flow, and boundary conditions;
- spatial fields, imaging, inverse problems, Bayesian design, and optimal
  control;
- experiment specifications and reproducible result containers.

The remaining major pieces are physically grounded stochastic acquisition,
high-resolution clocked protocols, and unified experiment-facade integration.
The existing `ESRSpinSystem` is deliberately restricted to spin one-half, so
the nano-MR layer remains additive rather than a breaking ESR rewrite.

## Architecture

The implementation belongs in `spin_dynamics.nano_mr` and uses three fidelity
backends:

1. **Filter-function/statistical-bath backend.** Fast XY-family spectroscopy,
   spectral overlap, analytic layers and half-spaces, and parameter sweeps.
2. **Trajectory backend.** Diffusion, flow, confinement, field
   autocorrelations, Qdyne records, and non-Lorentzian lineshapes.
3. **Exact backend.** Small resolved electron-nuclear clusters with anisotropic
   hyperfine coupling, chemical shifts, scalar coupling, nuclear dipolar
   coupling, and piecewise microwave/RF control.

All backends should return compatible metadata and should distinguish
statistical, thermal, and explicitly polarized nuclear samples.

## Phased implementation

### Phase 1: sensor physics and geometry

- Add explicit laboratory, crystal, sensor, and surface frames.
- Add arbitrary-spin `DefectSpinSensor`.
- Add symmetric `g` and ZFS tensors.
- Add ZFS plus electron Zeeman Hamiltonians and ODMR transition analysis.
- Add planar surfaces, target nuclei, and point-dipole hyperfine tensors.
- Add diamond NV-minus and 4H-SiC PL6 presets.
- Validate Hermiticity, rotational covariance, axial transition frequencies,
  and inverse-cube coupling.

### Phase 2: control, filter functions, and optical readout

- Add laser, electron-microwave, nuclear-RF, delay, and readout channels.
- Add Ramsey, Hahn, CPMG, XY4/8/16, and arbitrary phase-cycled sequences.
- Compile ideal toggling functions and finite-width Hamiltonian steps.
- Add effective optical initialization, contrast, photon rate, dead time, and
  Poisson shot noise.
- Cross-validate weak-field exact propagation against analytic filter
  functions.

Implemented in `spin_dynamics.nano_mr.sequences`,
`spin_dynamics.nano_mr.filter_functions`, `spin_dynamics.nano_mr.compiler`,
and `spin_dynamics.nano_mr.readout`. Laser initialization and fluorescence
readout are effective non-unitary windows; electron-microwave and nuclear-RF
pulses are explicit coherent channels, and delays are compiled between them.

### Phase 3: statistical nano-NMR

- Add statistical, thermal, and fixed-polarization sample modes.
- Add uniform layers, half-spaces, voxel densities, and multiple isotopes.
- Add analytic field variance and spectral-density models.
- Reproduce proton spectra, multi-isotope discrimination, and sensor-depth
  scaling.

This is the minimum useful release.

Implemented in `spin_dynamics.nano_mr.baths` and
`spin_dynamics.nano_mr.statistical`. Uniform planar samples use an analytic
dipolar variance integral for arbitrary sensor, field, and surface
orientations. Voxel samples use midpoint volume integration. Each isotope has
an explicit polarization state and correlation time; spectra use a normalized
two-sided angular-frequency PSD and can be passed directly to the Phase-2
filter-overlap coherence calculation.

### Phase 4: exact resolved-spin spectroscopy

- Add nuclear Zeeman, chemical shift, scalar coupling, and nuclear dipolar
  terms.
- Add anisotropic sensor-target coupling and nuclear RF control.
- Add two-block correlation and small 2-D spectroscopy workflows.
- State and enforce dense-system size limits; add cluster/secular
  approximations later.

Implemented in `spin_dynamics.nano_mr.exact`,
`spin_dynamics.nano_mr.correlation`, and `spin_dynamics.nano_mr.esr_bridge`.
Resolved clusters combine the arbitrary-spin defect Hamiltonian with
laboratory-frame nuclear Zeeman/chemical-shift tensors, full anisotropic
point-dipole sensor-target coupling, isotropic or secular scalar coupling, and
full or secular nuclear dipolar coupling. Microwave CW transitions, ideal
selective sensor pulses, ideal nuclear-RF rotations, two Hahn sensing blocks,
and a windowed 2-D Fourier workflow are provided. Dense allocation defaults to
a 64-state ceiling; users must explicitly raise the per-cluster limit. A
strict bridge shares compatible spin-1/2, zero-ZFS systems with the pure ESR
module without treating higher-spin NV/SiC defects as ordinary radical ESR.

### Phase 5: diffusion and confinement

- Connect existing particle motion and boundaries to dipolar sensing kernels.
- Add seeded field records, autocorrelations, spectra, diffusion, drift, and
  confined liquids.
- Validate stationary, free-diffusion, long-time power-law, and flow limits.

Implemented in `spin_dynamics.nano_mr.motion`. The trajectory simulator reuses
the general seeded Brownian/advection integrator and reflecting, periodic, or
clipping box boundaries, while adding precessing nuclear magnetic moments and
the point-dipole field at an arbitrarily oriented sensor. Motion substeps are
independent of the recorded field interval. Field records feed FFT
autocorrelations and either full-record or Welch-averaged one-sided PSDs.
Displacement statistics validate \(6Dt\) free diffusion and imposed drift;
explicit bounds validate confined liquids; the Gaussian return propagator
provides the stationary and \(t^{-3/2}\) three-dimensional long-time
reference. The point-dipole cutoff and wrapped-coordinate caveat are explicit.

### Phase 6: nano-MRI

- Add scanning-sensor and sensor-array dipolar forward operators.
- Add raster and arbitrary scan trajectories, depth profiling, density
  reconstruction, sparse point-spin localization, and uncertainty estimates.
- Begin with nonnegative regularized inversion and nonlinear point-position
  fitting.

Implemented in `spin_dynamics.nano_mr.imaging`. One scan abstraction covers
serpentine rasters, arbitrary trajectories, and independently oriented sensor
arrays. Forward operators distinguish signed projected fields from
nonnegative transverse statistical-field variance and include physical voxel
density conversion. Analytic planar-slab operators provide depth profiles.
A projected accelerated-gradient solver performs nonnegative dimensionless
Tikhonov inversion with zero-, first-, or second-difference penalties and
linearized covariance estimates. Optional SciPy bounded least squares refines
sparse point positions and amplitudes with Jacobian-based local uncertainty.
Synthetic tests cross-check the Phase-3 voxel and uniform-layer backends,
inverse-cube scaling, raster ordering, density recovery, and point
localization.

### Phase 6.5: stochastic spin, optical, and detector noise

The current fast path makes three useful limiting assumptions: each isotope
has one exponential correlation time, optical readout has fixed contrast and
an aggregated Poisson count, and imaging residuals are independent Gaussian
samples described by one standard deviation. The realistic acquisition layer
must retain those APIs while exposing the complete generative chain

`nuclear trajectories -> field history -> sensor phase/population -> optical
state path -> photon arrivals -> detector record -> reconstruction`.

#### Target spins and sensor environment

For sensor or scan position \(i\), the microscopic target field is

\[
B_i(t)=\sum_j \mathbf n_i^\mathsf{T}
\mathbf D[\mathbf r_j(t)-\mathbf r_i]\boldsymbol\mu_j(t).
\]

The mesoscopic backend should additionally construct the spatial and temporal
cross-covariance

\[
C_{ij}(\tau)=\rho\int d^3r\,d^3r'\,
K_i(\mathbf r)G(\mathbf r',\tau\mid\mathbf r)K_j(\mathbf r')
\cos(\omega_L\tau)e^{-|\tau|/T_{2,n}}.
\]

Here \(G\) may describe diffusion, drift, confinement, adsorption, or
exchange. This makes neighboring scan pixels and successive repetitions
correlated and permits the experimentally observed long-time diffusion
power law [12, 13]. The nearest few spins may instead use the exact backend,
with the remote bath retained as a Gaussian field process.

Target nuclear fluctuations are part of the signal, not an interchangeable
detector-noise term. They remain distinct from carbon-13, substitutional
nitrogen, surface-electron, electric-field, and technical sensor noise.
Shallow-NV experiments support depth-dependent surface baths with finite
correlation rates and, in some cases, double-Lorentzian spectra [17-19].
Each environmental component therefore requires its own PSD or sampled field
process. For sensing shot \(k\),

\[
\phi_k=\gamma_e\int y_k(t)B_{\rm total}(t)\,dt+\delta\phi_k,\qquad
p_{0,k}=\frac{1}{2}\left[
1+V_k e^{-\chi_k}\cos(\phi_k+\phi_{\rm read})
\right],
\]

where \(V_k\), \(\chi_k\), and \(\delta\phi_k\) can include pulse infidelity,
relaxation, quasi-static detuning, microwave amplitude/phase noise, clock
jitter, and sensor-height or magnetic-field drift.

#### Optical cycling and detector transfer

Conventional room-temperature readout should use a configurable continuous-
time rate model containing the bright and dark ground/excited triplet
manifolds and a metastable singlet. Five-level rates have been fitted directly
to time-resolved NV fluorescence [20]. An optional charge-augmented model adds
NV-zero, photoionization, and recombination [21, 22]. If \(\mathbf p(t)\) is
the optical-state population,

\[
\dot{\mathbf p}(t)=\mathbf Q[I_{\rm laser}(t)]\mathbf p(t),\qquad
\lambda(t)=\eta\sum_s\Gamma_{{\rm rad},s}p_s(t)
+\lambda_{\rm background}(t).
\]

The same rates then determine initialization, time-dependent contrast,
singlet shelving, optical spin pumping, and charge blinking. Literature
values are illustrative presets rather than universal defect constants;
experimental work should fit rates to bright/dark time traces and charge
records.

Photon emission and detector transfer remain separate. The initial detector
backend should cover a SPAD/APD quantum efficiency, dark counts,
nonparalyzable or paralyzable dead time, timing jitter, and exponentially
distributed afterpulses. Dead time and afterpulsing distort the count
distribution and interarrival correlations [23]. Later detector variants may
add analog-APD multiplication noise or camera read noise, dark current,
gain/offset, and fixed-pattern response.

Even an ideal detector does not generally give one Poisson distribution after
spin averaging. For bright-state probability \(p\) and conditional photon
means \(\lambda_0,\lambda_1\),

\[
\operatorname E[N]=p\lambda_0+(1-p)\lambda_1,\qquad
\operatorname{Var}(N)=\operatorname E[N]
+p(1-p)(\lambda_0-\lambda_1)^2.
\]

The second term is state-projection/readout-mixture noise. Optical switching
and technical drift add overdispersion. Likewise, an independently sampled
Gaussian spin variance obeys a scaled chi-square distribution rather than an
additive Gaussian model; correlations reduce its effective sample count.

#### Implementation and inference

- Add reusable nuclear-field and sensor-environment stochastic processes with
  sampled records and positive-semidefinite cross-covariances.
- Add backward-compatible optical-cycle and charge-state rate models beside
  the existing effective readout.
- Add SPAD photon-transfer and raw shot/time-bin acquisition records, phase
  cycling, signal/reference normalization, and seeded reproducibility.
- Extend Phase-6 inversion with full-covariance generalized least squares and
  signal-dependent count likelihoods while retaining scalar `noise_std`.
- Propagate calibration uncertainty through Hessian, bootstrap, or Bayesian
  intervals instead of interpreting one residual RMS as the complete noise.
- Add examples comparing ideal Poisson/Gaussian data with correlated
  spin-plus-optical acquisition and report residual autocorrelation,
  overdispersion, and interval coverage.

Validation must recover the existing Poisson model when all switching and
detector effects are disabled; the nonparalyzable dead-time rate
\(r_{\rm obs}=r/(1+r\tau_d)\); the configured afterpulse waiting-time law;
five-level bright/dark mean traces; Phase-3 field variance and Phase-5
trajectory correlations; positive-semidefinite cross-pixel covariance; and
synthetic reconstruction with calibrated uncertainty coverage.

The initial implementation is in `spin_dynamics.nano_mr.noise`,
`spin_dynamics.nano_mr.readout`, and `spin_dynamics.nano_mr.imaging`.
It provides exponential and power-law field components, temporal/spatial
covariance and seeded Gaussian records, effective-sample-size and linear
covariance helpers, a five-level-plus-charge optical rate model, CTMC
photon-emission paths, SPAD efficiency/background/dead-time/afterpulse/jitter
transfer, raw per-shot arrival records, and generalized least-squares
nonnegative inversion. The effective Phase-2 Poisson API and scalar
Phase-6 `noise_std` path remain unchanged. The first example contrasts the
noise sources and detector statistics; the existing XY8 and scan examples
offer opt-in `--rate-readout` and `--correlated-noise` paths.

### Phase 7: high-resolution protocols

Phase 7 adds a clocked coherent-signal layer without treating statistical
single-NV nano-NMR and thermally polarized ensemble NMR as interchangeable.
The Phase-2 control sequence supplies the complex sensing response

\[
Y(\omega)=\int_0^{T_s}y(t)e^{i\omega t}\,dt ,
\]

so a coherent field \(B\cos(\omega t+\phi)\) produces the per-shot sensor phase
\(\Phi_n=\gamma_e B\operatorname{Re}[e^{i(\omega t_n+\phi)}Y(\omega)]\). Sensor coherence then acts as readout visibility, \(s_n=C_s\sin(\Phi_n+\phi_a)\), rather than rescaling \(\Phi_n\).
Qdyne samples that phase at clocked times \(t_n\), while synchronized readout
also records the orthogonal analysis quadrature.  The result objects must
retain nominal and perturbed timestamps, raw detuning, Nyquist-folded beat and alias order, the complex filter response, expected bright probabilities, and optional seeded shot records.

Clock error is not another sensor-decoherence constant.  A clock model therefore
has separately configurable static fractional frequency offset, interval
fractional-frequency instability, and trigger jitter.  Likewise a resolution
budget keeps

- the sensor coherence envelope acting within each sensing block;
- sample transverse coherence acting between clocked measurements;
- diffusion-induced correlation loss;
- ancillary-memory storage coherence and transfer/retrieval fidelity; and
- clock timing/phase error

as independent factors.  In the weak-signal limit their envelopes may multiply,
but their parameters and reported contributions must remain distinct.

Sensor-memory correlation returns

\[
S(t)=C_0 F_{\rm tr}F_{\rm ret}
 \exp[-t/T_{2,\rm sample}]
 \exp[-t/\tau_D]
 \exp[-t/T_{2,\rm mem}]
 \cos(2\pi f t+\phi),
\]

with the two sensor interrogation blocks contributing their own finite-\(T_2\)
contrast.  This is an effective correlation protocol rather than a microscopic
gate-error simulation.

For coherent thermal or deliberately polarized signals, a first-order
chemical-shift/J-resolved backend complements the Phase-4 dense exact backend.
Each site has an isotope, chemical shift, complex amplitude, sample
\(T_2\), and zero or more weak scalar couplings.  Repeated spin-one-half
couplings generate the usual product modulation
\(\prod_k\cos(\pi J_k t)\); Fourier transformation produces the resolved
multiplet about
\(\gamma_nB_0(1+\delta\,10^{-6})\).  Exact strongly coupled or anisotropic
clusters remain the responsibility of `nano_mr.exact`.

Optional DNP is an explicit preparation model, never an implicit signal boost:

\[
P(t)=P_{\rm th}+
  \left(P_{\rm ss}-P_{\rm th}\right)
  \left(1-e^{-t/T_{\rm build}}\right),\qquad
P_{\rm ss}=\operatorname{clip}(\epsilon P_{\rm th},-1,1).
\]

Turning pumping off gives \(P(t)=P(0)e^{-t/T_{1n}}\).  The model reports the
thermal, steady-state, and time-dependent polarizations so enhancement cannot
be confused with optical contrast.

Validation must recover the sequence toggling integral for one coherent tone,
the Nyquist alias relation, \(1/T_{\rm record}\) Fourier-bin scaling, the
zero-error clock limit, multiplicative and separately reported decoherence
envelopes, the memory-correlation analytic expression, doublet and
doublet-of-doublets line positions, and DNP build-up/decay limits.  Examples
should compare Qdyne with synchronized quadrature acquisition and show
chemical-shift/J resolution with and without sample, diffusion, and clock
broadening.  The effective Phase-2 readout, Phase-4 exact cluster, and Phase-5
trajectory APIs remain unchanged.

The initial implementation is in `spin_dynamics.nano_mr.high_resolution`.
It provides clock-perturbed timestamps, Qdyne and synchronized quadrature
records using the existing sequence toggling integral, optional reuse of the
effective optical readout, independent sensor/sample/diffusion/memory
envelopes, effective sensor-memory correlation, first-order coherent
chemical-shift/J FIDs and spectra, and bounded DNP build-up/decay. The new
Qdyne and coherent-spectrum examples expose the limiting parameters directly;
all earlier examples and defaults remain valid.

### Phase 8: package integration — implemented

- Add compact, validated facade specifications for defect presets, planar
  nuclear layers, effective optical readout, statistical spectra, and Qdyne.
  Keep exact clusters, trajectory ensembles, scan reconstruction, and
  coherent site lists in their existing expert APIs until a declarative schema
  can represent them without hiding physical assumptions.
- Register the statistical-spectrum and Qdyne engines as ideal-detector
  workflows. Planning must reject missing sensor/layer inputs, report ignored
  fields, and expose deterministic work/memory estimates.
- Round-trip nested nano-MR specifications through canonical JSON and friendly
  TOML, and round-trip native results (including optional photon counts)
  through provenance-bearing NPZ archives.
- Add a planner-validated Bayesian Qdyne adapter whose action selects the
  reference frequency and optional sensing duration, while latent signal
  frequency and field amplitude remain explicit.
- Add config-driven and Bayesian examples, package/manual documentation,
  generated API coverage, and a validation-evidence record that distinguishes
  analytic/regression checks from experimental validation.
- Gate representative statistical and Qdyne workloads with deterministic
  complexity/memory assertions plus a deliberately loose wall-time ceiling;
  the ceiling catches accidental algorithmic regressions and is not a
  cross-host performance claim.

The implemented facade routes are intentionally limited to analytic planar
statistical spectra and coherent-tone Qdyne. Both delegate to the native
`spin_dynamics.nano_mr` engines. The Qdyne route supports optional effective
photon counting, while the Bayesian adapter exposes the deterministic
normalized spin record and leaves detector noise in its likelihood. Native
results round-trip through provenance-bearing NPZ archives, including nested
optical counts. Friendly TOML now supports arrays of component tables.

Phase-8 evidence is exercised by `tests/test_nano_mr_experiment.py`, the two
new runnable examples, strict MkDocs and generated-reference checks, the
package validation registry, and the consolidated manual. The performance
gate uses explicit work/memory scaling and a five-second 4096-shot ceiling;
it detects accidental complexity regressions but is not a hardware benchmark.

## Validation matrix

Validation should include analytic spin-1 and spin-3/2 energies, published
defect constants, dipolar inverse-cube and half-space depth scaling, ideal and
finite-pulse filter functions, statistical-versus-thermal scaling,
multi-isotope peak positions, exact/filter/trajectory overlap, seeded
trajectory convergence, synthetic MRI reconstruction, selected 2013 diamond
curves, stochastic optical limiting cases, detector dead-time/afterpulse
statistics, correlated-noise reconstruction coverage, and PL6
proton/fluorine comparison explicitly tagged as preprint-based.

Version 1 will not claim a full optical excited-state model, ab initio defect
or chemical-shift prediction, dense propagation of tens of nuclei, turnkey
protein reconstruction, or automatic support for every SiC defect family.

## References

1. T. Staudacher et al., *Science* **339**, 561–563 (2013),
   [doi:10.1126/science.1231675](https://doi.org/10.1126/science.1231675).
2. H. J. Mamin et al., *Science* **339**, 557–560 (2013),
   [doi:10.1126/science.1231540](https://doi.org/10.1126/science.1231540).
3. S. J. DeVience et al., *Nature Nanotechnology* **10**, 129–134 (2015),
   [doi:10.1038/nnano.2014.313](https://doi.org/10.1038/nnano.2014.313).
4. D. Rugar et al., *Nature Nanotechnology* **10**, 120–124 (2015),
   [doi:10.1038/nnano.2014.288](https://doi.org/10.1038/nnano.2014.288).
5. I. Lovchinsky et al., *Science* **351**, 836–841 (2016),
   [doi:10.1126/science.aad8022](https://doi.org/10.1126/science.aad8022).
6. N. Aslam et al., *Science* **357**, 67–71 (2017),
   [doi:10.1126/science.aam8697](https://doi.org/10.1126/science.aam8697).
7. S. Schmitt et al., *Science* **356**, 832–837 (2017),
   [doi:10.1126/science.aam5532](https://doi.org/10.1126/science.aam5532).
8. J. M. Boss et al., *Science* **356**, 837–840 (2017),
   [doi:10.1126/science.aam7009](https://doi.org/10.1126/science.aam7009).
9. D. R. Glenn et al., *Nature* **555**, 351–354 (2018),
   [doi:10.1038/nature25781](https://doi.org/10.1038/nature25781).
10. M. H. Abobeih et al., *Nature* **576**, 411–415 (2019),
    [doi:10.1038/s41586-019-1834-7](https://doi.org/10.1038/s41586-019-1834-7).
11. K. S. Cujia et al., *Nature Communications* **13**, 1260 (2022),
    [doi:10.1038/s41467-022-28935-z](https://doi.org/10.1038/s41467-022-28935-z).
12. S. Oviedo-Casado et al., *Scientific Reports* **10**, 19691 (2020),
    [doi:10.1038/s41598-020-76745-4](https://doi.org/10.1038/s41598-020-76745-4).
13. N. Staudenmaier et al., *npj Quantum Information* **8**, 120 (2022),
    [doi:10.1038/s41534-022-00632-1](https://doi.org/10.1038/s41534-022-00632-1).
14. P. Li et al., *Nature Materials* **24**, 1913–1919 (2025),
    [doi:10.1038/s41563-025-02382-9](https://doi.org/10.1038/s41563-025-02382-9).
15. Y. Chen et al., “Single-molecule Scale Nuclear Magnetic Resonance
    Spectroscopy using a Robust Near-Infrared Spin Sensor” (2025),
    [arXiv:2512.10278](https://arxiv.org/abs/2512.10278).
16. C. L. Degen, F. Reinhard, and P. Cappellaro, *Reviews of Modern Physics*
    **89**, 035002 (2017),
    [doi:10.1103/RevModPhys.89.035002](https://doi.org/10.1103/RevModPhys.89.035002).
17. B. A. Myers et al., *Physical Review Letters* **113**, 027602 (2014),
    [doi:10.1103/PhysRevLett.113.027602](https://doi.org/10.1103/PhysRevLett.113.027602).
18. Y. Romach et al., *Physical Review Letters* **114**, 017601 (2015),
    [doi:10.1103/PhysRevLett.114.017601](https://doi.org/10.1103/PhysRevLett.114.017601).
19. S. Sangtawesin et al., *Physical Review X* **9**, 031052 (2019),
    [doi:10.1103/PhysRevX.9.031052](https://doi.org/10.1103/PhysRevX.9.031052).
20. A. Gupta, L. Hacquebard, and L. Childress, *Journal of the Optical
    Society of America B* **33**, B28-B34 (2016),
    [doi:10.1364/JOSAB.33.000B28](https://doi.org/10.1364/JOSAB.33.000B28).
21. N. Aslam et al., *New Journal of Physics* **15**, 013064 (2013),
    [doi:10.1088/1367-2630/15/1/013064](https://doi.org/10.1088/1367-2630/15/1/013064).
22. L. Hacquebard and L. Childress, *Physical Review A* **97**, 063408
    (2018),
    [doi:10.1103/PhysRevA.97.063408](https://doi.org/10.1103/PhysRevA.97.063408).
23. M. Ware et al., *Journal of Modern Optics* **54**, 361-372 (2007),
    [doi:10.1080/09500340600722997](https://doi.org/10.1080/09500340600722997).
