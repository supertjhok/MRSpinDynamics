# API Reference

Generated from public class and function docstrings by `docs/generate_api_reference.py`.

This reference is an inventory, not a substitute for the user manual. For numerical assumptions, equations, and workflow guidance, see `docs/user_manual.tex`.

## `spin_dynamics.analysis.ilt`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `Regularization` | Tikhonov regularization settings for inverse Laplace solves. |
| class | `ILTResult1D` | Result returned by one-dimensional inverse Laplace transforms. |
| class | `ILTResult2D` | Result returned by separable two-dimensional inverse Laplace transforms. |
| function | `t2_kernel(echo_times: np.ndarray, t2_values: np.ndarray) -> np.ndarray` | Return the CPMG decay kernel ``exp(-te / T2)``. |
| function | `t1_kernel(recovery_times: np.ndarray, t1_values: np.ndarray, *, mode: Literal['saturation', 'inversion'] = 'saturation') -> np.ndarray` | Return a T1 recovery or inversion-recovery kernel. |
| function | `diffusion_kernel(b_values: np.ndarray, diffusion_values: np.ndarray) -> np.ndarray` | Return the diffusion attenuation kernel ``exp(-b D)``. |
| function | `laplace_kernel(sample_axis: np.ndarray, distribution_axis: np.ndarray, *, kind: KernelName = 't2') -> np.ndarray` | Build a named one-dimensional Laplace kernel. |
| function | `invert_laplace_1d(signal: np.ndarray, sample_axis: np.ndarray, distribution_axis: np.ndarray, *, kernel: KernelName | np.ndarray = 't2', regularization: float | Regularization = Regularization(), regularization_order: int | None = None, nonnegative: bool = True) -> ILTResult1D` | Estimate a non-negative 1D distribution from Laplace-domain data. |
| function | `invert_laplace_2d(data: np.ndarray, sample_axis1: np.ndarray, sample_axis2: np.ndarray, distribution_axis1: np.ndarray, distribution_axis2: np.ndarray, *, kernel1: KernelName | np.ndarray, kernel2: KernelName | np.ndarray, regularization: float | tuple[float, float] | Regularization | tuple[Regularization, Regularization] = Regularization(), regularization_order: int | tuple[int, int] | None = None, nonnegative: bool = True) -> ILTResult2D` | Estimate a 2D distribution from separable Laplace-domain data. |
| function | `invert_t2(signal: np.ndarray, echo_times: np.ndarray, t2_axis: np.ndarray, **kwargs) -> ILTResult1D` | Convenience wrapper for a 1D T2 inverse Laplace transform. |
| function | `invert_t1(signal: np.ndarray, recovery_times: np.ndarray, t1_axis: np.ndarray, *, mode: Literal['saturation', 'inversion'] = 'saturation', **kwargs) -> ILTResult1D` | Convenience wrapper for a 1D T1 recovery or inversion-recovery ILT. |
| function | `invert_t1_t2(data: np.ndarray, recovery_times: np.ndarray, echo_times: np.ndarray, t1_axis: np.ndarray, t2_axis: np.ndarray, *, t1_mode: Literal['saturation', 'inversion'] = 'saturation', **kwargs) -> ILTResult2D` | Convenience wrapper for a separable T1-T2 inverse Laplace transform. |
| function | `invert_d_t2(data: np.ndarray, b_values: np.ndarray, echo_times: np.ndarray, diffusion_axis: np.ndarray, t2_axis: np.ndarray, **kwargs) -> ILTResult2D` | Convenience wrapper for a separable D-T2 inverse Laplace transform. |

## `spin_dynamics.analysis.regularization`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `RegularizationCandidate1D` | A trial regularization strength and its 1D inversion result. |
| class | `RegularizationCandidate2D` | A trial regularization strength and its 2D inversion result. |
| class | `RegularizationSelection1D` | Selected 1D regularization result plus the full candidate trace. |
| class | `RegularizationSelection2D` | Selected 2D regularization result plus the full candidate trace. |
| function | `default_regularization_strengths(minimum: float = 1e-08, maximum: float = 10.0, count: int = 37) -> np.ndarray` | Return a logarithmic regularization-strength grid. |
| function | `estimate_noise_rms_from_snr(data: np.ndarray, snr: float) -> float` | Estimate noise RMS from observed real data and an RMS SNR estimate. |
| function | `expected_residual_norm_from_snr(data: np.ndarray, snr: float, *, target_multiplier: float = 1.0) -> float` | Return the discrepancy-principle residual norm target for an SNR. |
| function | `select_regularization_1d(signal: np.ndarray, sample_axis: np.ndarray, distribution_axis: np.ndarray, *, snr: float, kernel: KernelName | np.ndarray = 't2', strengths: Sequence[float] | None = None, regularization_order: int = 2, nonnegative: bool = True, target_multiplier: float = 1.0) -> RegularizationSelection1D` | Select a 1D regularization strength from an SNR estimate. |
| function | `select_regularization_2d(data: np.ndarray, sample_axis1: np.ndarray, sample_axis2: np.ndarray, distribution_axis1: np.ndarray, distribution_axis2: np.ndarray, *, snr: float, kernel1: KernelName | np.ndarray, kernel2: KernelName | np.ndarray, strengths: Sequence[float] | None = None, axis_strength_ratio: tuple[float, float] = (1.0, 1.0), regularization_order: int | tuple[int, int] = 2, nonnegative: bool = True, target_multiplier: float = 1.0) -> RegularizationSelection2D` | Select a shared 2D regularization scale from an SNR estimate. |

## `spin_dynamics.coupling.evolution`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `propagator(hamiltonian: np.ndarray, duration: float) -> np.ndarray` | Return ``exp(-i H duration)`` for a Hermitian Hamiltonian. |
| function | `evolve_density(density: np.ndarray, hamiltonian: np.ndarray, duration: float) -> np.ndarray` | Evolve a density operator under a time-independent Hamiltonian. |
| function | `propagate_density(density: np.ndarray, steps: list[tuple[np.ndarray, float]] | tuple[tuple[np.ndarray, float], ...]) -> np.ndarray` | Evolve a density operator through a sequence of Hamiltonian steps. |
| function | `equilibrium_density(system: CoupledSpinSystem, axis: str = 'z') -> np.ndarray` | Return a high-temperature equilibrium density operator. |

## `spin_dynamics.coupling.hamiltonians`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `zeeman_hamiltonian(system: CoupledSpinSystem) -> np.ndarray` | Return the rotating-frame offset Hamiltonian in radians per second. |
| function | `secular_j_hamiltonian(system: CoupledSpinSystem) -> np.ndarray` | Return the weak-coupling secular scalar Hamiltonian. |
| function | `isotropic_j_hamiltonian(system: CoupledSpinSystem) -> np.ndarray` | Return the isotropic scalar Hamiltonian for strongly coupled spins. |
| function | `rf_hamiltonian(system: CoupledSpinSystem, nutation_hz: float | Iterable[float], *, phase: float = 0.0, indices: Iterable[int] | None = None) -> np.ndarray` | Return an RF Hamiltonian for selected spins in radians per second. |

## `spin_dynamics.coupling.isochromats`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CoupledIsochromatEnsemble` | Static field maps for a coupled-spin isochromat ensemble. |
| class | `CoupledIsochromatStep` | One time-independent step for a coupled isochromat ensemble. |
| class | `CoupledIsochromatSequenceResult` | Signal and final states from a coupled isochromat sequence. |
| function | `coupled_isochromat_ensemble(base_system: CoupledSpinSystem, b0_offsets_hz: Iterable[float] | np.ndarray, *, weights: float | Iterable[float] | np.ndarray = 1.0, b1_tx_scale: float | Iterable[float] | np.ndarray = 1.0, b1_rx_scale: float | Iterable[float] | np.ndarray | None = None, offset_scales: Iterable[float] | np.ndarray | None = None) -> CoupledIsochromatEnsemble` | Build a coupled-spin isochromat ensemble. |
| function | `free_precession_step(duration: float, *, b0_offsets_hz: float | Iterable[float] | np.ndarray | None = None) -> CoupledIsochromatStep` | Return a free-precession step with optional time-varying B0 offsets. |
| function | `rf_step(duration: float, nutation_hz: float | Sequence[float], *, phase: float = 0.0, b0_offsets_hz: float | Iterable[float] | np.ndarray | None = None, b1_tx_scale: float | Iterable[float] | np.ndarray | None = None, indices: Sequence[int] | None = None) -> CoupledIsochromatStep` | Return an RF or spin-lock step with optional local B0/B1 overrides. |
| function | `simulate_coupled_isochromat_sequence(ensemble: CoupledIsochromatEnsemble, steps: Sequence[CoupledIsochromatStep], *, initial_axis: str = 'x', detect_axis: str = 'x', j_mode: str = 'isotropic') -> CoupledIsochromatSequenceResult` | Propagate a coupled-spin sequence over an isochromat ensemble. |

## `spin_dynamics.coupling.j_editing`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `JEditingFitResult` | Known-J least-squares fit of a J-modulation curve. |
| function | `j_modulation_curve(encoding_times: Iterable[float] | np.ndarray, couplings_hz: Iterable[float] | np.ndarray, amplitudes: Iterable[float] | np.ndarray | None = None, *, cycles: int = 1, background: float = 0.0, powers: Iterable[int] | np.ndarray | None = None) -> np.ndarray` | Return a superposition of J-modulated cosine components. |
| function | `carbon_detected_j_modulation(encoding_times: Iterable[float] | np.ndarray, couplings_hz: Iterable[float] | np.ndarray, abundances: Iterable[float] | np.ndarray, proton_counts: Iterable[int] | np.ndarray, *, cycles: int = 1, scale: float = 1.0) -> np.ndarray` | Return the carbon-detected low-field J-editing model. |
| function | `proton_detected_j_modulation(encoding_times: Iterable[float] | np.ndarray, couplings_hz: Iterable[float] | np.ndarray, amplitudes: Iterable[float] | np.ndarray, *, cycles: int = 1, background: float = 0.0) -> np.ndarray` | Return the proton-detected J-editing model. |
| function | `tango_b_filter(couplings_hz: Iterable[float] | np.ndarray, *, delay_seconds: float | None = None, target_coupling_hz: float | None = None, order: int = 1) -> np.ndarray` | Return the ideal TANGO-B coupled-spin transverse filter amplitude. |
| function | `fit_known_j_spectrum(encoding_times: Iterable[float] | np.ndarray, signal: Iterable[float] | np.ndarray, couplings_hz: Iterable[float] | np.ndarray, *, cycles: int = 1, powers: Iterable[int] | np.ndarray | None = None, include_background: bool = True) -> JEditingFitResult` | Fit amplitudes for a known set of J-coupling frequencies. |

## `spin_dynamics.coupling.operators`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `spin_operator(nspin: int, index: int, axis: str) -> np.ndarray` | Return a single-spin operator embedded in the full Hilbert space. |
| function | `total_operator(nspin: int, axis: str, indices: Iterable[int] | None = None) -> np.ndarray` | Return the sum of selected spin operators along one axis. |
| function | `product_operator(nspin: int, terms: Iterable[tuple[int, str]]) -> np.ndarray` | Return a product operator such as ``I1z I2z``. |

## `spin_dynamics.coupling.slic`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SLICSpectrumResult` | Simulated SLIC response as a function of spin-lock nutation frequency. |
| function | `two_spin_slic_transfer_time(offset_difference_hz: float) -> float` | Return the ideal two-spin SLIC maximum-transfer time. |
| function | `simulate_slic_spectrum(system: CoupledSpinSystem, nutation_frequencies_hz: Iterable[float] | np.ndarray, *, spin_lock_time: float, initial_axis: str = 'x', detect_axis: str = 'x') -> SLICSpectrumResult` | Simulate remaining transverse magnetization after a spin-lock pulse. |

## `spin_dynamics.coupling.systems`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SpinSite` | One spin-1/2 site in a coupled spin system. |
| class | `CoupledSpinSystem` | Small dense spin-1/2 system with scalar couplings in hertz. |
| function | `coupled_spin_system(offsets_hz: Iterable[float], couplings_hz: Iterable[Iterable[float]], *, labels: Sequence[str] | None = None, isotopes: Sequence[str] | None = None) -> CoupledSpinSystem` | Build a validated spin-1/2 system from offsets and couplings. |

## `spin_dynamics.core.echo`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `calc_time_domain_echo(spect: np.ndarray, wvect: np.ndarray, *, zero_fill: int = 4) -> tuple[np.ndarray, np.ndarray]` | Convert an offset-domain spectrum into a time-domain echo. |
| function | `calc_time_domain_echo_arb(mrx: np.ndarray, wvect: np.ndarray, tacq: float, tdw: float) -> tuple[np.ndarray, np.ndarray]` | Calculate a time-domain echo by direct summation. |
| function | `calc_fid_time_domain(mrx: np.ndarray, wvect: np.ndarray, tacq: float, tdw: float) -> tuple[np.ndarray, np.ndarray]` | Calculate a time-domain FID by direct summation. |

## `spin_dynamics.core.isochromats`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `RephasingAnalysis` | Rephasing estimate for a uniformly spaced offset grid. |
| function | `offset_spacing(del_w: np.ndarray) -> float` | Return the uniform offset spacing for an isochromat grid. |
| function | `estimate_rephase_time(del_w: np.ndarray) -> float` | Estimate the normalized rephasing time for a uniform offset grid. |
| function | `recommended_numpts_for_rephasing(maxoffs: float, max_time: float, safety_factor: float = 1.25) -> int` | Return the minimum grid size that keeps rephasing beyond max time. |
| function | `analyze_rephasing(del_w: np.ndarray, max_time: float, safety_factor: float = 1.25) -> RephasingAnalysis` | Analyze whether a grid is fine enough for the requested simulation time. |
| function | `check_rephasing(del_w: np.ndarray, max_time: float, safety_factor: float = 1.25, action: str = 'warn') -> RephasingAnalysis` | Warn or raise when the isochromat grid may produce rephasing artifacts. |

## `spin_dynamics.core.kernels`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `Arb10Parameters` | Parameters for `sim_spin_dynamics_arb10`. |
| class | `Arb10DiffusionParameters` | Parameters for `sim_spin_dynamics_arb10_diffusion`. |
| class | `Arb7Parameters` | Parameters for `sim_spin_dynamics_arb7`. |
| function | `sim_spin_dynamics_arb10(params: Mapping[str, Any] | Arb10Parameters | Any) -> np.ndarray` | Simulate arbitrary-pulse spin dynamics with precomputed pulse matrices. |
| function | `sim_spin_dynamics_arb10_radiation_damping(params: Mapping[str, Any] | Arb10Parameters | Any, radiation_damping: RadiationDampingSpec) -> np.ndarray` | Simulate `arb10` with ensemble radiation damping during free intervals. |
| function | `sim_spin_dynamics_arb10_diffusion(params: Mapping[str, Any] | Arb10DiffusionParameters | Any) -> np.ndarray` | Simulate arbitrary-pulse dynamics with a diffusion free-precession term. |
| function | `sim_spin_dynamics_arb10_chunked(params: Mapping[str, Any] | Arb10Parameters | Any, num_workers: int | None = None, min_chunk_size: int = 256) -> np.ndarray` | Run `sim_spin_dynamics_arb10` on contiguous isochromat chunks. |
| function | `sim_spin_dynamics_arb10_diffusion_chunked(params: Mapping[str, Any] | Arb10DiffusionParameters | Any, num_workers: int | None = None, min_chunk_size: int = 256) -> np.ndarray` | Run `sim_spin_dynamics_arb10_diffusion` on isochromat chunks. |
| function | `sim_spin_dynamics_arb7(params: Mapping[str, Any] | Arb7Parameters | Any) -> np.ndarray` | Simulate arbitrary-pulse dynamics with acquisition-window convolution. |

## `spin_dynamics.core.numerics`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `trapezoid(y: Any, x: Any | None = None, axis: int = -1) -> np.ndarray` | Integrate with NumPy's trapezoid rule across NumPy versions. |

## `spin_dynamics.core.rotations`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MatrixElements` | Rotation matrix elements in MATLAB's `M0`, `M-`, `M+` coherence basis. |
| function | `rf_matrix_elements(del_w: np.ndarray, w1: float, tp: float, phi: float) -> MatrixElements` | Calculate RF-pulse matrix elements without relaxation. |
| function | `free_precession_matrix_elements(del_w: np.ndarray, tf: float) -> MatrixElements` | Calculate free-precession matrix elements without relaxation. |
| function | `sim_spin_dynamics_asymp_mag3(tp: np.ndarray, phi: np.ndarray, amp: np.ndarray, neff: np.ndarray, del_w: np.ndarray, t_acq: float) -> np.ndarray` | Calculate asymptotic magnetization for a small-pulse sequence. |
| function | `sim_spin_dynamics_exc(tp: np.ndarray, phi: np.ndarray, amp: np.ndarray, del_w: np.ndarray) -> np.ndarray` | Calculate the magnetization vector after an excitation pulse. |
| function | `calc_rot_axis_arba4(tp: np.ndarray, phi: np.ndarray, amp: np.ndarray, del_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]` | Calculate effective rotation axis and angle for arbitrary amplitudes. |
| function | `calc_rot_axis_arba3(tp: np.ndarray, phi: np.ndarray, amp: np.ndarray, del_w: np.ndarray) -> np.ndarray` | Calculate effective rotation axis for arbitrary-amplitude cycles. |
| function | `calc_v0crit(del_w: np.ndarray, n: np.ndarray, alpha: np.ndarray) -> np.ndarray` | Calculate the critical-velocity parameter for a refocusing cycle. |
| function | `calc_rotation_matrix(del_w: np.ndarray, w_1: np.ndarray | float, tp: np.ndarray, phi: np.ndarray, amp: np.ndarray) -> MatrixElements` | Calculate the equivalent rotation matrix of a composite pulse. |

## `spin_dynamics.motion`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MotionFieldMaps2D` | Two-dimensional field maps used by moving isochromats. |
| class | `ParticleEnsemble` | Moving isochromat ensemble. |
| function | `make_motion_field_maps_2d(x_axis: Iterable[float] | np.ndarray, z_axis: Iterable[float] | np.ndarray, *, b0_map: Iterable[float] | np.ndarray | None = None, b0_vector_map: Iterable[float] | np.ndarray | None = None, b1_tx_map: Iterable[float] | np.ndarray | None = None, b1_tx_vector_map: Iterable[float] | np.ndarray | None = None, b1_rx_map: Iterable[float] | np.ndarray | None = None, b1_rx_vector_map: Iterable[float] | np.ndarray | None = None) -> MotionFieldMaps2D` | Validate and assemble two-dimensional field maps. |
| function | `transverse_b1_magnitude(b0_vector_map: Iterable[float] | np.ndarray, b1_vector_map: Iterable[float] | np.ndarray) -> np.ndarray` | Return the local B1 magnitude perpendicular to the local B0 direction. |
| function | `initialize_ensemble_from_density(rho: Iterable[float] | np.ndarray, x_axis: Iterable[float] | np.ndarray, z_axis: Iterable[float] | np.ndarray, *, walkers_per_cell: int = 1, diffusion_coefficient: float | Iterable[float] | np.ndarray = 0.0, seed: int | None = None, jitter: bool = False) -> ParticleEnsemble` | Create a walker ensemble from a two-dimensional spin-density map. |
| function | `advect_diffuse_positions(positions: np.ndarray, dt: float, *, velocity: Velocity = None, diffusion_coefficient: float | Iterable[float] | np.ndarray = 0.0, rng: np.random.Generator | None = None, time: float = 0.0, bounds: tuple[tuple[float, float], tuple[float, float]] | None = None, boundary: BoundaryMode = 'reflect') -> np.ndarray` | Advance positions with deterministic advection and Brownian diffusion. |
| function | `move_ensemble(ensemble: ParticleEnsemble, dt: float, *, velocity: Velocity = None, rng: np.random.Generator | None = None, time: float = 0.0, bounds: tuple[tuple[float, float], tuple[float, float]] | None = None, boundary: BoundaryMode = 'reflect') -> ParticleEnsemble` | Return an ensemble with advected/diffused positions. |
| function | `apply_free_precession(ensemble: ParticleEnsemble, dt: float, off_resonance: Iterable[float] | np.ndarray, *, t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0) -> ParticleEnsemble` | Apply relaxation and off-resonance precession to each particle. |
| function | `apply_rf_rotation(ensemble: ParticleEnsemble, duration: float, phase: float, amplitude: float, off_resonance: Iterable[float] | np.ndarray, *, b1_tx: float | Iterable[float] | np.ndarray = 1.0) -> ParticleEnsemble` | Apply a rectangular RF rotation using local B1 transmit scaling. |
| function | `free_precession_with_motion_step(ensemble: ParticleEnsemble, fields: MotionFieldMaps2D, dt: float, *, velocity: Velocity = None, rng: np.random.Generator | None = None, time: float = 0.0, gradient: tuple[float, float] = (0.0, 0.0), t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0, boundary: BoundaryMode = 'reflect') -> ParticleEnsemble` | Move particles and apply a first-order free-precession update. |
| function | `receive_signal(ensemble: ParticleEnsemble, fields: MotionFieldMaps2D | None = None) -> complex` | Sum weighted received transverse magnetization over particles. |
| function | `apply_boundary(positions: np.ndarray, bounds: tuple[tuple[float, float], tuple[float, float]], mode: BoundaryMode) -> np.ndarray` | Apply simple boundary conditions to two-dimensional positions. |

## `spin_dynamics.noise`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `NoiseSpec` | Configuration for additive received-signal noise. |
| class | `NoiseMetadata` | Summary of the generated noise realization. |
| class | `MatchedFilterSNRResult` | Matched-filter SNR estimate from clean and noisy spectra. |
| function | `as_noise_spec(noise: NoiseSpec | Mapping[str, Any] | float | int | None) -> NoiseSpec | None` | Normalize public noise inputs to a validated `NoiseSpec`. |
| function | `estimate_matched_filter_snr(clean_signal: np.ndarray, noisy_signals: np.ndarray, *, pnoise: np.ndarray | None = None, frequencies: np.ndarray | None = None, offsets: np.ndarray | None = None, noise_scale: float = 1.0, matched_filter: np.ndarray | None = None) -> MatchedFilterSNRResult` | Estimate matched-filter SNR from repeated noisy spectra. |
| function | `add_received_noise(signal: np.ndarray, noise: NoiseSpec | Mapping[str, Any] | float | int | None, *, pnoise: np.ndarray | None = None, frequencies: np.ndarray | None = None, sample_axis: np.ndarray | None = None) -> tuple[np.ndarray, NoiseMetadata | None]` | Return `signal` with additive noise plus generation metadata. |
| function | `ideal_noise_density(signal: np.ndarray, noise: NoiseSpec | Mapping[str, Any] | float | int) -> tuple[np.ndarray, np.ndarray]` | Return a flat output-referred density matching a white-noise spec. |
| function | `tuned_probe_output_noise_density(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any) -> tuple[np.ndarray, np.ndarray]` | Return tuned-probe output-referred noise density and frequencies. |
| function | `untuned_probe_output_noise_density(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any) -> tuple[np.ndarray, np.ndarray]` | Return untuned-probe output-referred noise density and frequencies. |
| function | `matched_probe_output_noise_density(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, tf1: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]` | Return matched-probe output-referred noise density and frequencies. |
| function | `frequency_bin_width(frequencies: np.ndarray) -> float` | Estimate a representative frequency-bin width. |

## `spin_dynamics.nqr.hamiltonians`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `quadrupole_hamiltonian(site: QuadrupolarSite) -> np.ndarray` | Return the zero-field quadrupole Hamiltonian in radians per second. |
| function | `zeeman_hamiltonian(site: QuadrupolarSite, b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray` | Return the Zeeman Hamiltonian in radians per second. |
| function | `nqr_hamiltonian(site: QuadrupolarSite, b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float] | None = None) -> np.ndarray` | Return the quadrupole plus optional Zeeman Hamiltonian. |
| function | `diagonalize_site(site: QuadrupolarSite, b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float] | None = None, *, strength_tolerance: float = 1e-12) -> NQREigensystem` | Diagonalize a site Hamiltonian and return transition metadata. |

## `spin_dynamics.nqr.operators`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `validate_spin(spin: float) -> float` | Return a validated integer or half-integer spin quantum number. |
| function | `spin_dimension(spin: float) -> int` | Return the Hilbert-space dimension for one spin. |
| class | `SpinMatrices` | Dense single-spin angular momentum matrices. |
| function | `spin_matrices(spin: float) -> SpinMatrices` | Return dense angular-momentum matrices for one spin. |

## `spin_dynamics.nqr.orientations`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `spherical_direction(alpha: float, beta: float) -> np.ndarray` | Return a unit vector from azimuth `alpha` and polar angle `beta`. |
| class | `OrientationSample` | One local EFG orientation relative to lab RF and static fields. |
| function | `single_crystal_orientation(alpha: float, beta: float, *, b0_alpha: float | None = None, b0_beta: float | None = None) -> tuple[OrientationSample, ...]` | Return a one-sample orientation ensemble. |
| function | `powder_average_grid(n_theta: int = 16, n_phi: int = 32) -> tuple[OrientationSample, ...]` | Return a normalized spherical powder-average grid. |
| function | `normalize_orientations(orientations: tuple[OrientationSample, ...] | list[OrientationSample]) -> tuple[OrientationSample, ...]` | Return orientation samples with weights normalized to unity. |

## `spin_dynamics.nqr.pulses`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SelectivePulse` | A rectangular selective RF pulse applied to one NQR transition. |
| function | `transition_drive_scale(transition: NQRTransition, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float]) -> float` | Return the relative RF coupling for a transition and B1 orientation. |
| function | `selective_pulse_hamiltonian(dimension: int, transition: NQRTransition, *, nutation_hz: float, phase: float = 0.0, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0), detuning_hz: float = 0.0) -> np.ndarray` | Return an embedded two-level RF Hamiltonian in radians per second. |
| function | `apply_selective_pulse(density: np.ndarray, transition: NQRTransition, pulse: SelectivePulse, *, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray` | Apply a selective pulse to a density matrix in the energy basis. |

## `spin_dynamics.nqr.sequences`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SLSESequence` | Spin-lock spin-echo detection sequence. |
| function | `slse_sequence(transition_label: str, *, pulse_duration_seconds: float, nutation_hz: float, echo_spacing_seconds: float, num_echoes: int, phase: float = 0.0, rf_frequency_hz: float | None = None) -> SLSESequence` | Build a rectangular-pulse SLSE sequence. |

## `spin_dynamics.nqr.simulation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SLSEResult` | Simulated SLSE echo train. |
| class | `PopulationTransferResult` | Perturbation plus SLSE detection result. |
| function | `equilibrium_density(levels_hz: np.ndarray) -> np.ndarray` | Return a trace-zero high-temperature density matrix in the energy basis. |
| function | `transition_signal(density: np.ndarray, transition: NQRTransition, *, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float]) -> complex` | Return the complex single-coil signal for a transition coherence. |
| function | `simulate_slse(site: QuadrupolarSite, sequence: SLSESequence, *, orientations: str | tuple[OrientationSample, ...] | list[OrientationSample] = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf, initial_density: np.ndarray | None = None) -> SLSEResult` | Simulate a selective-pulse SLSE echo train. |
| function | `simulate_population_transfer(site: QuadrupolarSite, perturbation: SelectivePulse, detection_sequence: SLSESequence, *, orientations: str | tuple[OrientationSample, ...] | list[OrientationSample] = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf) -> PopulationTransferResult` | Simulate a perturbation pulse followed by SLSE detection. |

## `spin_dynamics.nqr.systems`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `QuadrupolarSite` | One quadrupolar nucleus in its EFG principal-axis system. |
| class | `NQRTransition` | One transition between quadrupolar energy eigenstates. |
| class | `NQREigensystem` | Energy levels, eigenvectors, and allowed transitions for one site. |

## `spin_dynamics.nqr.workflows`

No public classes or functions found.

## `spin_dynamics.parameters.constructors`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SystemParameters` | Simulation/system parameters corresponding to MATLAB `sp`. |
| class | `PulseParameters` | Pulse-sequence parameters corresponding to MATLAB `pp`. |
| class | `FIDSystemParameters` | Simulation/system parameters corresponding to ideal FID MATLAB `sp`. |
| class | `FIDPulseParameters` | Pulse-sequence parameters corresponding to ideal FID MATLAB `pp`. |
| class | `TunedOrigParameters` | Compact tuned-probe parameters corresponding to MATLAB `params`. |
| class | `TunedSystemParameters` | Tuned-probe system parameters corresponding to MATLAB `sp`. |
| class | `TunedPulseParameters` | Tuned-probe pulse parameters corresponding to MATLAB `pp`. |
| class | `UntunedOrigParameters` | Compact untuned-probe parameters corresponding to MATLAB `params`. |
| class | `UntunedSystemParameters` | Untuned-probe system parameters corresponding to MATLAB `sp`. |
| class | `UntunedPulseParameters` | Untuned-probe pulse parameters corresponding to MATLAB `pp`. |
| class | `MatchedSystemParameters` | Matched-probe system parameters corresponding to MATLAB `sp`. |
| class | `MatchedPulseParameters` | Matched-probe pulse parameters corresponding to MATLAB `pp`. |
| function | `set_params_ideal(numpts: int = 10000) -> tuple[SystemParameters, PulseParameters]` | Construct default ideal no-probe CPMG parameters. |
| function | `set_params_ideal_fid(numpts: int = 20000) -> tuple[FIDSystemParameters, FIDPulseParameters]` | Construct default ideal no-probe FID parameters. |
| function | `set_params_tuned_orig(numpts: int = 10000) -> tuple[TunedOrigParameters, TunedSystemParameters, TunedPulseParameters]` | Construct original/reference tuned-probe CPMG parameters. |
| function | `set_params_tuned_spa(numpts: int = 5000) -> tuple[TunedOrigParameters, TunedSystemParameters, TunedPulseParameters]` | Construct tuned-probe SPA pulse-evaluation parameters. |
| function | `set_params_tuned_jmr(numpts: int = 10000) -> tuple[TunedSystemParameters, TunedPulseParameters]` | Construct JMR-paper tuned-probe parameters. |
| function | `set_params_untuned_orig(numpts: int = 10000) -> tuple[UntunedOrigParameters, UntunedSystemParameters, UntunedPulseParameters]` | Construct original/reference untuned-probe CPMG parameters. |
| function | `set_params_untuned_spa(numpts: int = 5000) -> tuple[UntunedOrigParameters, UntunedSystemParameters, UntunedPulseParameters]` | Construct untuned-probe SPA pulse-evaluation parameters. |
| function | `set_params_untuned_jmr(numpts: int = 2000) -> tuple[UntunedSystemParameters, UntunedPulseParameters]` | Construct JMR-paper untuned-probe parameters. |
| function | `set_params_matched_orig(numpts: int = 10000) -> tuple[MatchedSystemParameters, MatchedPulseParameters]` | Construct original/reference matched-probe CPMG parameters. |
| function | `set_params_matched_spa(numpts: int = 5000) -> tuple[MatchedSystemParameters, MatchedPulseParameters]` | Construct matched-probe SPA pulse-evaluation parameters. |
| function | `set_params_matched_jmr(numpts: int = 2000) -> tuple[MatchedSystemParameters, MatchedPulseParameters]` | Construct JMR-paper matched-probe parameters. |

## `spin_dynamics.optimization.drivers`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MultiStartOptimizationResult` | Array-returning result for repeated random-start phase optimization. |
| function | `random_phase_starts(num_starts: int, num_segments: int, *, bounds: tuple[float, float] = (0.0, 2 * np.pi), seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray` | Generate reproducible random phase starts within bounded phase limits. |
| function | `run_tuned_refocusing_multistart(num_segments: int, *, num_starts: int = 24, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | list[list[float]] | None = None, bounds: tuple[float, float] = (0.0, 2 * np.pi), **optimizer_kwargs) -> MultiStartOptimizationResult` | Run repeated random-start tuned refocusing phase searches. |
| function | `run_ideal_v0crit_refocusing_multistart(num_segments: int, *, num_starts: int = 24, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | list[list[float]] | None = None, bounds: tuple[float, float] = (0.0, 2 * np.pi), **optimizer_kwargs) -> MultiStartOptimizationResult` | Run repeated random-start ideal v0crit refocusing phase searches. |
| function | `run_ideal_v0crit_excited_refocusing_multistart(num_segments: int, *, num_starts: int = 24, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | list[list[float]] | None = None, bounds: tuple[float, float] = (0.0, 2 * np.pi), **optimizer_kwargs) -> MultiStartOptimizationResult` | Run repeated ideal v0crit searches with a fixed excitation vector. |
| function | `run_ideal_time_varying_refocusing_multistart(num_segments: int, *, num_starts: int = 24, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | list[list[float]] | None = None, bounds: tuple[float, float] = (0.0, 2 * np.pi), **optimizer_kwargs) -> MultiStartOptimizationResult` | Run repeated random-start ideal time-varying refocusing searches. |
| function | `run_untuned_refocusing_multistart(num_segments: int, *, num_starts: int = 24, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | list[list[float]] | None = None, bounds: tuple[float, float] = (0.0, 2 * np.pi), **optimizer_kwargs) -> MultiStartOptimizationResult` | Run repeated random-start untuned refocusing phase searches. |
| function | `run_matched_refocusing_multistart(num_segments: int, *, num_starts: int = 4, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | list[list[float]] | None = None, bounds: tuple[float, float] = (0.0, 2 * np.pi), **optimizer_kwargs) -> MultiStartOptimizationResult` | Run repeated random-start matched refocusing phase searches. |
| function | `run_tuned_excitation_multistart(num_segments: int, neff: np.ndarray | list[list[float]], *, num_starts: int = 24, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | list[list[float]] | None = None, bounds: tuple[float, float] = (0.0, 2 * np.pi), **optimizer_kwargs) -> MultiStartOptimizationResult` | Run repeated random-start tuned excitation phase searches. |
| function | `run_tuned_inverse_excitation_multistart(num_segments: int, neff: np.ndarray | list[list[float]], target_mrx: np.ndarray | list[complex], target_snr: float, reference_phases: np.ndarray | list[float], *, num_starts: int = 24, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | list[list[float]] | None = None, random_fraction: float = 0.3, bounds: tuple[float, float] = (0.0, 2 * np.pi), **optimizer_kwargs) -> MultiStartOptimizationResult` | Run MATLAB-style repeated starts for tuned inverse excitation search. |

## `spin_dynamics.optimization.excitation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `TunedExcitationEvaluation` | Non-plotting tuned-probe excitation-pulse evaluation. |
| class | `TunedInverseExcitationEvaluation` | Evaluation of a tuned excitation pulse against an inversion target. |
| class | `ExcitationOptimizationResult` | Result of bounded fixed-amplitude excitation phase optimization. |
| function | `evaluate_tuned_excitation_pulse(phases: np.ndarray | list[float], neff: np.ndarray | list[list[float]], *, segment_fraction: float = 0.1, numpts: int = 101) -> TunedExcitationEvaluation` | Evaluate a fixed-amplitude tuned-probe excitation phase program. |
| function | `evaluate_tuned_inverse_excitation_pulse(phases: np.ndarray | list[float], neff: np.ndarray | list[list[float]], target_mrx: np.ndarray | list[complex], target_snr: float, *, segment_fraction: float = 0.1, numpts: int = 101) -> TunedInverseExcitationEvaluation` | Evaluate a tuned excitation pulse as an inverse phase-cycle partner. |
| function | `optimize_tuned_excitation_phases(initial_phases: np.ndarray | list[float], neff: np.ndarray | list[list[float]], *, segment_fraction: float = 0.1, numpts: int = 101, bounds: tuple[float, float] = (0.0, 2 * np.pi), initial_step: float = np.pi / 2, step_decay: float = 0.5, min_step: float = 0.001, max_passes: int = 8, optimizer: str = 'auto', scipy_method: str = 'L-BFGS-B', scipy_options: dict[str, object] | None = None) -> ExcitationOptimizationResult` | Optimize tuned-probe fixed-amplitude excitation phases. |
| function | `optimize_tuned_inverse_excitation_phases(initial_phases: np.ndarray | list[float], neff: np.ndarray | list[list[float]], target_mrx: np.ndarray | list[complex], target_snr: float, *, segment_fraction: float = 0.1, numpts: int = 101, bounds: tuple[float, float] = (0.0, 2 * np.pi), initial_step: float = np.pi / 2, step_decay: float = 0.5, min_step: float = 0.001, max_passes: int = 8, optimizer: str = 'auto', scipy_method: str = 'L-BFGS-B', scipy_options: dict[str, object] | None = None) -> ExcitationOptimizationResult` | Optimize a tuned excitation pulse to invert a target received spectrum. |

## `spin_dynamics.optimization.pipeline`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `TunedExcitationInversePipelineResult` | Selected-refocusing to excitation/inverse-excitation pipeline result. |
| function | `run_tuned_excitation_inverse_pipeline(refocusing: Any, *, pulse_number: int | None = None, excitation_segments: int = 3, excitation_starts: int = 4, inverse_starts: int = 4, seed: int | None = None, numpts: int = 21, maxoffs: float = 10.0, result_times_are_t180: bool = True, random_fraction: float = 0.3, excitation_kwargs: dict[str, Any] | None = None, inverse_kwargs: dict[str, Any] | None = None) -> TunedExcitationInversePipelineResult` | Run excitation and inverse-excitation searches from a refocusing result. |

## `spin_dynamics.optimization.refocusing`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `IdealV0CritRefocusingEvaluation` | Ideal-probe refocusing evaluation for the v0crit objective. |
| class | `IdealTimeVaryingRefocusingEvaluation` | Ideal-probe refocusing evaluation for time-varying B0 fields. |
| class | `RefocusingOptimizationResult` | Result of bounded fixed-amplitude refocusing phase optimization. |
| function | `ideal_time_varying_excitation_vector(*, numpts: int = 101, maxoffs: float = 4.0, pulse_times: np.ndarray | list[float] | None = None, pulse_phases: np.ndarray | list[float] | None = None, pulse_amplitudes: np.ndarray | list[float] | None = None) -> np.ndarray` | Return the ideal excitation vector used by v0crit-excitation searches. |
| function | `evaluate_ideal_v0crit_refocusing_pulse(phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, free_precession_t180: float = 1.5, numpts: int = 101, maxoffs: float | None = None, acquisition_time_normalized: float | None = None, excitation_vector: np.ndarray | list[list[complex]] | None = None, v0crit_weight: float = 100.0) -> IdealV0CritRefocusingEvaluation` | Evaluate the ideal-probe refocusing objective used by v0crit workflows. |
| function | `evaluate_ideal_v0crit_excited_refocusing_pulse(phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, free_precession_t180: float = 1.5, numpts: int = 101, maxoffs: float = 4.0, acquisition_time_normalized: float | None = None, excitation_vector: np.ndarray | list[list[complex]] | None = None, v0crit_weight: float = 100.0) -> IdealV0CritRefocusingEvaluation` | Evaluate ideal v0crit refocusing with a supplied excitation spectrum. |
| function | `evaluate_ideal_time_varying_refocusing_pulse(phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, echo_spacing_t180: float = 4.0, field_offsets: np.ndarray | list[float] | None = None, fluctuation_amplitude: float = 1.5, num_echoes: int = 16, numpts: int = 101, maxoffs: float = 10.0, t1_seconds: float = 100000000.0, t2_seconds: float = 100000000.0, score_scale: float = 10000.0, num_workers: int | None = 1) -> IdealTimeVaryingRefocusingEvaluation` | Evaluate ideal refocusing phases for time-varying-field robustness. |
| function | `optimize_tuned_refocusing_phases(initial_phases: np.ndarray | list[float], **kwargs) -> RefocusingOptimizationResult` | Optimize tuned-probe fixed-amplitude refocusing phases. |
| function | `optimize_untuned_refocusing_phases(initial_phases: np.ndarray | list[float], **kwargs) -> RefocusingOptimizationResult` | Optimize untuned-probe fixed-amplitude refocusing phases. |
| function | `optimize_matched_refocusing_phases(initial_phases: np.ndarray | list[float], **kwargs) -> RefocusingOptimizationResult` | Optimize matched-probe fixed-amplitude refocusing phases. |
| function | `optimize_ideal_v0crit_refocusing_phases(initial_phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, free_precession_t180: float = 1.5, numpts: int = 101, maxoffs: float | None = None, acquisition_time_normalized: float | None = None, excitation_vector: np.ndarray | list[list[complex]] | None = None, v0crit_weight: float = 100.0, bounds: tuple[float, float] = (0.0, 2 * np.pi), initial_step: float = np.pi / 2, step_decay: float = 0.5, min_step: float = 0.001, max_passes: int = 8, optimizer: str = 'auto', scipy_method: str = 'L-BFGS-B', scipy_options: dict[str, object] | None = None) -> RefocusingOptimizationResult` | Optimize ideal-probe phases for the v0crit refocusing objective. |
| function | `optimize_ideal_v0crit_excited_refocusing_phases(initial_phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, free_precession_t180: float = 1.5, numpts: int = 101, maxoffs: float = 4.0, acquisition_time_normalized: float | None = None, excitation_vector: np.ndarray | list[list[complex]] | None = None, v0crit_weight: float = 100.0, bounds: tuple[float, float] = (0.0, 2 * np.pi), initial_step: float = np.pi / 2, step_decay: float = 0.5, min_step: float = 0.001, max_passes: int = 8, optimizer: str = 'auto', scipy_method: str = 'L-BFGS-B', scipy_options: dict[str, object] | None = None) -> RefocusingOptimizationResult` | Optimize ideal v0crit refocusing phases after a fixed excitation pulse. |
| function | `optimize_ideal_time_varying_refocusing_phases(initial_phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, echo_spacing_t180: float = 4.0, field_offsets: np.ndarray | list[float] | None = None, fluctuation_amplitude: float = 1.5, num_echoes: int = 16, numpts: int = 101, maxoffs: float = 10.0, t1_seconds: float = 100000000.0, t2_seconds: float = 100000000.0, score_scale: float = 10000.0, num_workers: int | None = 1, bounds: tuple[float, float] = (0.0, 2 * np.pi), initial_step: float = np.pi / 2, step_decay: float = 0.5, min_step: float = 0.001, max_passes: int = 8, optimizer: str = 'auto', scipy_method: str = 'L-BFGS-B', scipy_options: dict[str, object] | None = None) -> RefocusingOptimizationResult` | Optimize ideal refocusing phases for time-varying-field robustness. |

## `spin_dynamics.optimization.results`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MatlabResultSummary` | Compact score summary extracted from MATLAB-style result cells. |
| class | `PulseProgram` | Piecewise-constant pulse program extracted from optimization results. |
| class | `SelectedOptimizationProgram` | Selected pulse program and score from MATLAB-style result cells. |
| class | `MatlabResultLayout` | Column layout used by a MATLAB `plot_opt_*_results` script. |
| class | `OptimizationResultAnalysis` | Script-aware, plotting-free analysis of MATLAB optimization result cells. |
| class | `TunedInverseResultPairAnalysis` | Comparison corresponding to `plot_opt_exc_results_tuned_inv.m`. |
| function | `multistart_to_matlab_results(multistart: Any, *, segment_fraction_t180: float | None = None, free_precession_t180: float = 0.0, excitation_segment_fraction_t180: float | None = None, refocusing_segment_fraction_t180: float = 0.1) -> np.ndarray` | Convert a multi-start optimization result to a MATLAB-style cell array. |
| function | `multistart_summary_arrays(multistart: Any) -> dict[str, np.ndarray]` | Return compact numeric arrays useful for non-MATLAB result inspection. |
| function | `save_multistart_results_npz(multistart: Any, path: str | Path, *, variable_name: str = 'results', **conversion_options) -> Path` | Save multi-start results as a NumPy archive with MATLAB-style cells. |
| function | `load_multistart_results_npz(path: str | Path, *, variable_name: str = 'results') -> dict[str, np.ndarray]` | Load a NumPy optimization archive written by `save_multistart_results_npz`. |
| function | `load_optimization_results(path: str | Path, *, variable_name: str = 'results') -> np.ndarray` | Load optimization result cells from a `.npz` or MATLAB `.mat` file. |
| function | `save_multistart_results_mat(multistart: Any, path: str | Path, *, variable_name: str = 'results', **conversion_options) -> Path` | Save multi-start results to a MATLAB `.mat` file when SciPy is present. |
| function | `load_matlab_results_mat(path: str | Path, *, variable_name: str = 'results') -> np.ndarray` | Load MATLAB optimization result cells from a `.mat` file when SciPy exists. |
| function | `matlab_result_layouts() -> dict[str, MatlabResultLayout]` | Return known MATLAB optimization result layouts keyed by canonical name. |
| function | `get_matlab_result_layout(layout: str | MatlabResultLayout | None = None, *, results: Any | None = None) -> MatlabResultLayout` | Resolve or infer a MATLAB optimization result-cell layout. |
| function | `summarize_matlab_results(results: Any, *, pulse_kind: str | None = None, pulse_number: int | None = None, maximize: bool = True) -> MatlabResultSummary` | Summarize scores from MATLAB-style optimization result cells. |
| function | `select_matlab_result_program(results: Any, *, pulse_kind: str | None = None, pulse_number: int | None = None) -> SelectedOptimizationProgram` | Extract the selected pulse program from MATLAB-style result cells. |
| function | `analyze_matlab_optimization_results(results: Any, *, layout: str | MatlabResultLayout | None = None, pulse_number: int | None = None) -> OptimizationResultAnalysis` | Analyze MATLAB optimization cells using a specific plot-script layout. |
| function | `analyze_optimization_result_file(path: str | Path, *, layout: str | MatlabResultLayout | None = None, pulse_number: int | None = None, variable_name: str = 'results') -> OptimizationResultAnalysis` | Load and analyze a `.mat` or `.npz` optimization result file. |
| function | `analyze_tuned_inverse_result_pair(original_results: Any, inverse_results: Any, *, pulse_number: int | None = None) -> TunedInverseResultPairAnalysis` | Analyze the original/inverse files used by `plot_opt_exc_results_tuned_inv`. |
| function | `analyze_tuned_inverse_result_files(original_path: str | Path, inverse_path: str | Path | None = None, *, pulse_number: int | None = None, variable_name: str = 'results') -> TunedInverseResultPairAnalysis` | Load and analyze original/inverse tuned-excitation result files. |

## `spin_dynamics.optimization.spa`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SPAPulse` | Fixed-amplitude SPA refocusing pulse phase program. |
| class | `SPAMetrics` | Normalized SPA/rectangular pulse performance metrics. |
| class | `SPASummary` | Array-returning summary of rectangular and SPA pulse performance. |
| class | `SPAOptimizationResult` | Result of a lightweight discrete SPA phase-program search. |
| class | `TunedRefocusingEvaluation` | Non-plotting tuned-probe arbitrary-refocusing-pulse evaluation. |
| class | `UntunedRefocusingEvaluation` | Non-plotting untuned-probe arbitrary-refocusing-pulse evaluation. |
| class | `MatchedRefocusingEvaluation` | Non-plotting matched-probe arbitrary-refocusing-pulse evaluation. |
| function | `spa_pulse_list(segment_fraction: float = 0.1) -> tuple[SPAPulse, ...]` | Return the fixed broadband SPA refocusing pulses from Mandal et al. |
| function | `rectangular_refocusing_lengths() -> np.ndarray` | Return the rectangular reference pulse lengths used by MATLAB SPA scripts. |
| function | `evaluate_tuned_refocusing_pulse(phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, numpts: int = 101, excitation_amplitude: float = 6.0) -> TunedRefocusingEvaluation` | Evaluate a fixed-amplitude tuned-probe refocusing phase program. |
| function | `evaluate_untuned_refocusing_pulse(phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, numpts: int = 101, excitation_amplitude: float = 6.0) -> UntunedRefocusingEvaluation` | Evaluate a fixed-amplitude untuned-probe refocusing phase program. |
| function | `evaluate_matched_refocusing_pulse(phases: np.ndarray | list[float], *, segment_fraction: float = 0.1, numpts: int = 101, excitation_amplitude: float = 6.0) -> MatchedRefocusingEvaluation` | Evaluate a fixed-amplitude matched-probe refocusing phase program. |
| function | `evaluate_spa_metrics(spa_snr: np.ndarray | list[float], rectangular_snr: np.ndarray | list[float], *, free_precession_t180: float = 3.0, segment_fraction: float = 0.1, pulse_lengths_t180: np.ndarray | list[float] | None = None) -> SPAMetrics` | Normalize SPA and rectangular performance metrics like MATLAB. |
| function | `summarize_spa_refocusing(probe: str, *, numpts: int = 101, segment_fraction: float = 0.1, pulse_indices: Iterable[int] | np.ndarray | None = None, excitation_amplitude: float = 6.0) -> SPASummary` | Run MATLAB-style SPA rectangular/catalog summary for a probe. |
| function | `summarize_tuned_spa_refocusing(**kwargs) -> SPASummary` | Summarize tuned-probe rectangular and SPA refocusing pulses. |
| function | `summarize_untuned_spa_refocusing(**kwargs) -> SPASummary` | Summarize untuned-probe rectangular and SPA refocusing pulses. |
| function | `summarize_matched_spa_refocusing(**kwargs) -> SPASummary` | Summarize matched-probe rectangular and SPA refocusing pulses. |
| function | `optimize_spa_phase_program(initial_phases: np.ndarray | list[float], score_fn: Callable[[np.ndarray], float], *, phase_states: np.ndarray | list[float] | None = None, max_passes: int = 1) -> SPAOptimizationResult` | Discrete coordinate-search scaffold for SPA/OCT phase optimization. |

## `spin_dynamics.pulses`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ProbePulseResponse` | Transmit pulse response and receiver transfer function arrays. |
| class | `WURSTPulse` | Piecewise-constant WURST pulse representation. |
| class | `UntunedPulseAdjustment` | Quantized phase and segment-length adjustment for an untuned pulse. |
| function | `quantize_phase(phi: np.ndarray | list[float], num_phases: int) -> np.ndarray` | Quantize phases to the nearest evenly spaced phase state. |
| function | `create_wurst_pulse(*, duration_seconds: float, sweep_width_rad_per_s: float, num_steps: int = 2000, order: int = 20, amplitude: float = 1.0, initial_phase: float = np.pi / 2, center_frequency_offset: float = 0.0) -> WURSTPulse` | Create a WURST amplitude and frequency-sweep pulse. |
| function | `adjust_untuned_segment_lengths(segment_lengths: np.ndarray | list[float], phases: np.ndarray | list[float], sp: Mapping[str, Any] | Any | None = None, pp: Mapping[str, Any] | Any | None = None, *, num_phases: int | None = None) -> UntunedPulseAdjustment` | Adjust untuned-probe segment lengths to reduce switching transients. |
| function | `tuned_rectangular_pulse_response(*, voltage_scale: float = 62.5, numpts: int = 10000) -> ProbePulseResponse` | Return the JMR tuned-probe rectangular-pulse response. |
| function | `untuned_rectangular_pulse_response(*, voltage_scale: float = 62.5, numpts: int = 2000) -> ProbePulseResponse` | Return the JMR untuned-probe rectangular-pulse response. |
| function | `matched_rectangular_pulse_response(*, numpts: int = 2000) -> ProbePulseResponse` | Return the JMR matched-probe rectangular-pulse response. |
| function | `matched_wurst_pulse_response(pulse: WURSTPulse, *, numpts: int = 2000, q_value: float | None = None, drive_phase: float | None = None) -> ProbePulseResponse` | Return matched-probe transmit response to a WURST RF block. |

## `spin_dynamics.radiation_damping`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `radiation_damping_time(gamma: float, fill_factor: float, equilibrium_magnetization: float, probe_q: float) -> float` | Return the radiation-damping time constant ``Trd`` in seconds. |
| function | `proton_thermal_magnetization_density(field_tesla: float, *, proton_concentration_mol_per_liter: float = 111.0, temperature_kelvin: float = 300.0) -> float` | Estimate spin-1/2 proton thermal magnetization density in A/m. |
| class | `RadiationDampingSample` | Convenience description of a sample's equilibrium magnetization. |
| function | `water_proton_sample(field_tesla: float, *, temperature_kelvin: float = 300.0, polarization_scale: float = 1.0) -> RadiationDampingSample` | Return a liquid-water proton sample preset for RD coupling. |
| function | `hyperpolarized_proton_sample(field_tesla: float, *, proton_concentration_mol_per_liter: float = 111.0, temperature_kelvin: float = 300.0, polarization_scale: float = 10000.0) -> RadiationDampingSample` | Return a proton sample preset with boosted non-equilibrium polarization. |
| function | `normalized_radiation_damping_weights(density: np.ndarray, sensitivity: np.ndarray | None = None) -> np.ndarray` | Return normalized RD ensemble weights from density and coil sensitivity. |
| class | `RadiationDampingProbe` | Probe coupling parameters for radiation-damping simulations. |
| class | `RadiationDampingResult` | Time-domain magnetization and probe feedback from an RD simulation. |
| class | `RadiationDampingSpec` | Settings for RD-aware arbitrary-sequence propagation. |
| function | `radiation_damping_probe_from_parameters(sp: Mapping[str, Any] | Any, *, fill_factor: float, equilibrium_magnetization: float | None = None, q: float | None = None, phase: float = 0.0, detuning: float = 0.0, name: str = 'probe') -> RadiationDampingProbe` | Build a radiation-damping probe from existing tuned/matched ``sp``. |
| function | `radiation_damping_probe_from_tuned(sp: Mapping[str, Any] | Any, *, fill_factor: float, equilibrium_magnetization: float | None = None, phase: float = 0.0, detuning: float = 0.0) -> RadiationDampingProbe` | Build an RD coupling object from a tuned-probe parameter set. |
| function | `radiation_damping_probe_from_matched(sp: Mapping[str, Any] | Any, *, fill_factor: float, equilibrium_magnetization: float | None = None, phase: float = 0.0, detuning: float = 0.0) -> RadiationDampingProbe` | Build an RD coupling object from a matched-probe parameter set. |
| function | `initial_state_from_flip_angle(flip_angle: float, *, pulse_phase: float = 0.0, equilibrium_magnetization: float = 1.0) -> tuple[complex, float]` | Return the post-pulse normalized state for an ideal hard pulse. |
| function | `analytic_radiation_damping_envelope(time: np.ndarray, flip_angle: float, trd: float, *, equilibrium_magnetization: float = 1.0, t2: float = np.inf) -> np.ndarray` | Analytic FID envelope for an on-resonance hard pulse with no T1 term. |
| function | `simulate_radiation_damping(time: np.ndarray, probe: RadiationDampingProbe, *, initial_mxy: complex, initial_mz: float, t1: float = np.inf, t2: float = np.inf, equilibrium_mz: float = 1.0, drive: complex | Callable[[float], complex] | None = None, model: str = 'instant', initial_feedback: complex | None = None, max_step: float | None = None) -> RadiationDampingResult` | Integrate the rotating-frame Bloch equations with RD back-action. |
| function | `simulate_radiation_damping_fid(time: np.ndarray, probe: RadiationDampingProbe, *, flip_angle: float = np.pi / 2, pulse_phase: float = 0.0, t1: float = np.inf, t2: float = np.inf, equilibrium_mz: float = 1.0, model: str = 'instant', max_step: float | None = None) -> RadiationDampingResult` | Simulate an FID after an ideal hard pulse in the RD model. |
| function | `simulate_nmr_maser(time: np.ndarray, probe: RadiationDampingProbe, *, seed_mxy: complex = -1e-06j, initial_mz: float = -1.0, pump_mz: float = -1.0, t1: float, t2: float, model: str = 'circuit', initial_feedback: complex | None = None, max_step: float | None = None) -> RadiationDampingResult` | Simulate an idealized pumped NMR maser in the RD feedback model. |

## `spin_dynamics.sequences.motion`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MotionSequenceStep` | One interval in a moving-isochromat pulse sequence. |
| class | `MotionSequenceResult` | Result from a moving-isochromat sequence simulation. |
| function | `run_motion_sequence(ensemble: ParticleEnsemble, fields: MotionFieldMaps2D, steps: Sequence[MotionSequenceStep], *, velocity: Velocity = None, rng: np.random.Generator | None = None, t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0, boundary: BoundaryMode = 'reflect', default_substeps: int = 1) -> MotionSequenceResult` | Run a sequence while moving particles through sampled field maps. |
| function | `make_motion_cpmg_sequence(num_echoes: int, echo_spacing: float, *, excitation_duration: float, refocusing_duration: float, excitation_phase: float = np.pi / 2, refocusing_phase: float = 0.0, gradient: tuple[float, float] = (0.0, 0.0), substeps_per_interval: int = 1) -> tuple[MotionSequenceStep, ...]` | Build a rectangular-pulse CPMG sequence for moving isochromats. |
| function | `run_motion_cpmg_sequence(ensemble: ParticleEnsemble, fields: MotionFieldMaps2D, *, num_echoes: int, echo_spacing: float, excitation_duration: float, refocusing_duration: float, gradient: tuple[float, float] = (0.0, 0.0), velocity: Velocity = None, rng: np.random.Generator | None = None, t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0, boundary: BoundaryMode = 'reflect', substeps_per_interval: int = 1) -> MotionSequenceResult` | Run a rectangular-pulse CPMG sequence with moving isochromats. |

## `spin_dynamics.workflows.acquisition`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `calc_macq_ideal_probe_relax4(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, num_workers: int | None = 1, rephase_max_time: float | None = None, rephase_safety_factor: float = 1.25, rephase_action: str = 'ignore', radiation_damping: RadiationDampingSpec | None = None) -> np.ndarray` | Calculate acquired spectra for an ideal-probe arbitrary sequence. |
| function | `calc_macq_tuned_probe_relax4(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, num_workers: int | None = 1, rephase_max_time: float | None = None, rephase_safety_factor: float = 1.25, rephase_action: str = 'ignore', radiation_damping: RadiationDampingSpec | None = None) -> tuple[np.ndarray, np.ndarray]` | Calculate finite acquisition for a tuned probe. |
| function | `calc_macq_untuned_probe_relax4(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, num_workers: int | None = 1, rephase_max_time: float | None = None, rephase_safety_factor: float = 1.25, rephase_action: str = 'ignore', radiation_damping: RadiationDampingSpec | None = None) -> tuple[np.ndarray, np.ndarray]` | Calculate finite acquisition for an untuned probe. |
| function | `calc_macq_matched_probe_relax4(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, num_workers: int | None = 1, rephase_max_time: float | None = None, rephase_safety_factor: float = 1.25, rephase_action: str = 'ignore', radiation_damping: RadiationDampingSpec | None = None) -> tuple[np.ndarray, np.ndarray]` | Calculate finite acquisition for a tuned-and-matched probe. |

## `spin_dynamics.workflows.cpmg`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CPMGResult` | Common result object for ideal and probe-aware CPMG workflows. |
| class | `CPMGTrainResult` | Finite ideal CPMG acquisition result. |
| function | `calc_masy_ideal(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any) -> np.ndarray` | Calculate ideal CPMG asymptotic magnetization. |
| function | `run_ideal_cpmg_train(numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, *, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGTrainResult` | Run a finite ideal CPMG echo train with relaxation. |
| function | `run_tuned_cpmg_train(numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, *, q_value: float | None = None, mistuning_offset: float | None = None, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', noise: NoiseSpec | Mapping[str, Any] | float | int | None = None, radiation_damping: RadiationDampingSpec | Mapping[str, Any] | None = None) -> CPMGTrainResult` | Run a finite tuned-probe CPMG echo train with relaxation. |
| function | `run_untuned_cpmg_train(numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, *, q_value: float | None = None, mistuning_offset: float | None = None, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGTrainResult` | Run a finite untuned-probe CPMG echo train with relaxation. |
| function | `run_matched_cpmg_train(numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, *, q_value: float | None = None, mistuning_offset: float | None = None, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', noise: NoiseSpec | Mapping[str, Any] | float | int | None = None, radiation_damping: RadiationDampingSpec | Mapping[str, Any] | None = None) -> CPMGTrainResult` | Run a finite matched-probe CPMG echo train with relaxation. |
| function | `run_ideal_cpmg(numpts: int = 101, maxoffs: float = 10.0, *, noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGResult` | Run the validated ideal no-probe CPMG workflow. |
| function | `run_tuned_cpmg(numpts: int = 101, maxoffs: float = 10.0, *, noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGResult` | Run the original/reference tuned-probe CPMG workflow. |
| function | `run_untuned_cpmg(numpts: int = 101, maxoffs: float = 10.0, *, noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGResult` | Run the original/reference untuned-probe CPMG workflow. |
| function | `run_matched_cpmg(numpts: int = 101, maxoffs: float = 10.0, *, noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGResult` | Run the original/reference matched-probe CPMG workflow. |

## `spin_dynamics.workflows.cpmg_ir`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CPMGIRTrainResult` | Finite CPMG-IR echo train over inversion delays. |
| class | `MatchedCPMGIRTrainResult` | Finite matched-probe CPMG-IR echo train over inversion delays. |
| function | `run_ideal_cpmg_ir_train(num_echoes: int = 10, echo_spacing_seconds: float = 0.0005, tauvect: Iterable[float] | np.ndarray | None = None, t1_seconds: float = 0.005, t2_seconds: float = 0.005, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1, tau_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGIRTrainResult` | Run a compact ideal-probe CPMG-IR finite echo train. |
| function | `run_tuned_cpmg_ir_train(num_echoes: int = 10, echo_spacing_seconds: float = 0.0005, tauvect: Iterable[float] | np.ndarray | None = None, t1_seconds: float = 0.005, t2_seconds: float = 0.005, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1, tau_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGIRTrainResult` | Run a compact tuned-probe CPMG-IR finite echo train. |
| function | `run_untuned_cpmg_ir_train(num_echoes: int = 10, echo_spacing_seconds: float = 0.0005, tauvect: Iterable[float] | np.ndarray | None = None, t1_seconds: float = 0.005, t2_seconds: float = 0.005, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1, tau_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGIRTrainResult` | Run a compact untuned-probe CPMG-IR finite echo train. |
| function | `run_matched_cpmg_ir_train(num_echoes: int = 10, echo_spacing_seconds: float = 0.0005, tauvect: Iterable[float] | np.ndarray | None = None, t1_seconds: float = 0.005, t2_seconds: float = 0.005, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1, tau_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> MatchedCPMGIRTrainResult` | Run a compact matched-probe CPMG-IR finite echo train. |

## `spin_dynamics.workflows.diffusion`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MatchedDiffusionCPMGResult` | Matched-probe diffusion-aware finite CPMG result. |
| class | `MatchedDiffusionQSweepResult` | Q sweep result for matched-probe diffusion-aware CPMG. |
| function | `check_matched_diffusion_q_stability(q_value: float, *, action: str = 'warn') -> bool` | Check the compact matched-diffusion Q validation boundary. |
| function | `calc_macq_matched_probe_relax_diffusion(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, apply_receiver: bool = True, num_workers: int | None = 1) -> tuple[np.ndarray, np.ndarray]` | Calculate diffusion-aware matched-probe finite acquisition. |
| function | `run_matched_diffusion_cpmg(num_echoes: int = 5, echo_spacing_seconds: float = 0.001, t1_seconds: float = 0.1, t2_seconds: float = 0.1, dz: float = 0.001, diffusion_time: float = 0.001, t90_seconds: float = 0.0001, q_value: float = 50.0, *, numpts: int = 101, apply_receiver: bool = False, num_workers: int | None = 1, q_stability_action: str = 'warn', auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> MatchedDiffusionCPMGResult` | Run a compact matched-probe diffusion-aware CPMG train. |
| function | `run_matched_diffusion_q_sweep(q_values: Iterable[float] | np.ndarray | None = None, *, num_echoes: int = 5, echo_spacing_seconds: float = 0.001, numpts: int = 101, num_workers: int | None = 1, sweep_workers: int | None = 1, q_stability_action: str = 'warn', auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> MatchedDiffusionQSweepResult` | Sweep matched-probe Q for the compact diffusion CPMG workflow. |

## `spin_dynamics.workflows.fid`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `RadiationDampingFIDResult` | Workflow result for an ideal hard-pulse FID with radiation damping. |
| function | `calc_macq_fid(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, params: Mapping[str, Any] | Any) -> tuple[np.ndarray, float]` | Calculate acquired ideal FID magnetization. |
| function | `sim_fid_ideal(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]` | Simulate the ideal no-probe FID workflow. |
| function | `run_radiation_damping_fid(*, probe: str = 'matched', fill_factor: float = 0.7, equilibrium_magnetization: float | None = None, field_tesla: float = 1.0, proton_concentration_mol_per_liter: float = 111.0, temperature_kelvin: float = 300.0, polarization_scale: float = 1.0, flip_angle: float = np.pi / 2, pulse_phase: float = 0.0, phase: float = 0.0, detuning: float = 0.0, duration_seconds: float | None = None, num_points: int = 401, t1_seconds: float = np.inf, t2_seconds: float = np.inf, model: str = 'instant') -> RadiationDampingFIDResult` | Run an ideal hard-pulse FID with probe-coupled radiation damping. |

## `spin_dynamics.workflows.imaging`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `make_imaging_field_maps(rho: Iterable[float] | np.ndarray, *, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, b0_map: Iterable[float] | np.ndarray | None = None, b0_vector_map: Iterable[float] | np.ndarray | None = None, b1_tx_map: Iterable[float] | np.ndarray | None = None, b1_tx_vector_map: Iterable[float] | np.ndarray | None = None, b1_rx_map: Iterable[float] | np.ndarray | None = None, b1_rx_vector_map: Iterable[float] | np.ndarray | None = None, del_wx: Iterable[float] | np.ndarray | None = None, del_wz: Iterable[float] | np.ndarray | None = None) -> ImagingFieldMaps` | Validate and assemble spatial maps for CPMG imaging. |
| function | `load_imaging_field_maps_npz(path: str | Path, *, rho_key: str = 'rho', t1_key: str = 't1_map', t2_key: str = 't2_map', b0_key: str = 'b0_map', b0_vector_key: str = 'b0_vector_map', b1_tx_key: str = 'b1_tx_map', b1_tx_vector_key: str = 'b1_tx_vector_map', b1_rx_key: str = 'b1_rx_map', b1_rx_vector_key: str = 'b1_rx_vector_map', del_wx_key: str = 'del_wx', del_wz_key: str = 'del_wz') -> ImagingFieldMaps` | Load imaging field maps from a NumPy `.npz` archive. |
| function | `reconstruct_image_from_kspace(kspace: np.ndarray, echo_index: int = 0) -> np.ndarray` | Reconstruct an image from one echo of CPMG imaging k-space. |
| function | `fit_imaging_echo_decay(result: IdealCPMGImagingResult | ProbeCPMGImagingResult, *, echo_times: Iterable[float] | np.ndarray | None = None, min_signal: float = 0.0, use_noisy: bool = False) -> ImagingEchoFitResult` | Fit each voxel magnitude to `rho * exp(-t / T2)`. |
| function | `form_imaging_image(result: IdealCPMGImagingResult | ProbeCPMGImagingResult, *, mode: str = 'single', echo_index: int = 0, min_signal: float = 0.0, use_noisy: bool = False) -> np.ndarray` | Return a display-ready image from an imaging echo stack. |
| function | `summarize_imaging_noise_trials(results: Iterable[IdealCPMGImagingResult | ProbeCPMGImagingResult], *, mode: str = 'single', echo_index: int = 0, signal_mask: Iterable[bool] | np.ndarray | None = None, background_mask: Iterable[bool] | np.ndarray | None = None, min_signal: float = 0.0) -> ImagingNoiseStatistics` | Summarize repeated noisy imaging trials in image space. |
| function | `run_ideal_phase_encoded_cpmg_imaging(rho: Iterable[float] | np.ndarray | ImagingFieldMaps, *, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, num_echoes: int = 2, echo_spacing_seconds: float = 0.0002, gradient_duration_seconds: float = 0.0005, fov: tuple[float, float] | Iterable[float] = (20.0, 20.0), ny: int = 9, maxoffs: float = 5.0, num_workers: int | None = 1, phase_workers: int | None = 1, density_normalization: Literal['legacy', 'preserve'] = 'legacy', noise: NoiseSpec | Mapping[str, object] | float | int | None = None) -> IdealCPMGImagingResult` | Run a compact ideal-probe phase-encoded CPMG imaging simulation. |
| function | `run_t1_encoded_phase_encoded_cpmg_imaging(rho: Iterable[float] | np.ndarray | ImagingFieldMaps, *, inversion_time_seconds: float, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, num_echoes: int = 2, echo_spacing_seconds: float = 0.0002, gradient_duration_seconds: float = 0.0005, fov: tuple[float, float] | Iterable[float] = (20.0, 20.0), ny: int = 9, maxoffs: float = 5.0, num_workers: int | None = 1, phase_workers: int | None = 1, density_normalization: Literal['legacy', 'preserve'] = 'legacy', noise: NoiseSpec | Mapping[str, object] | float | int | None = None) -> IdealCPMGImagingResult` | Run ideal phase-encoded CPMG imaging with inversion-recovery T1 prep. |
| function | `run_t1_encoded_cpmg_imaging(rho: Iterable[float] | np.ndarray | ImagingFieldMaps, *, inversion_time_seconds: float, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, num_echoes: int = 2, echo_spacing_seconds: float = 0.0002, gradient_duration_seconds: float = 0.0005, fov: tuple[float, float] | Iterable[float] = (20.0, 20.0), ny: int = 9, maxoffs: float = 5.0, num_workers: int | None = 1, phase_workers: int | None = 1, density_normalization: Literal['legacy', 'preserve'] = 'legacy', noise: NoiseSpec | Mapping[str, object] | float | int | None = None) -> IdealCPMGImagingResult` | Compatibility alias for `run_t1_encoded_phase_encoded_cpmg_imaging`. |
| function | `run_ideal_cpmg_imaging(rho: Iterable[float] | np.ndarray | ImagingFieldMaps, *, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, num_echoes: int = 2, echo_spacing_seconds: float = 0.0002, gradient_duration_seconds: float = 0.0005, fov: tuple[float, float] | Iterable[float] = (20.0, 20.0), ny: int = 9, maxoffs: float = 5.0, num_workers: int | None = 1, phase_workers: int | None = 1, density_normalization: Literal['legacy', 'preserve'] = 'legacy', noise: NoiseSpec | Mapping[str, object] | float | int | None = None) -> IdealCPMGImagingResult` | Compatibility alias for `run_ideal_phase_encoded_cpmg_imaging`. |
| function | `run_tuned_phase_encoded_cpmg_imaging(rho: Iterable[float] | np.ndarray | ImagingFieldMaps, *, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, num_echoes: int = 2, echo_spacing_seconds: float = 0.0002, gradient_duration_seconds: float = 0.0005, fov: tuple[float, float] | Iterable[float] = (20.0, 20.0), ny: int = 9, maxoffs: float = 5.0, num_workers: int | None = 1, phase_workers: int | None = 1, receive_mode: str = 'raw', density_normalization: Literal['legacy', 'preserve'] = 'legacy', noise: NoiseSpec | Mapping[str, object] | float | int | None = None) -> ProbeCPMGImagingResult` | Run a compact tuned-probe phase-encoded CPMG imaging simulation. |
| function | `run_tuned_cpmg_imaging(rho: Iterable[float] | np.ndarray | ImagingFieldMaps, *, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, num_echoes: int = 2, echo_spacing_seconds: float = 0.0002, gradient_duration_seconds: float = 0.0005, fov: tuple[float, float] | Iterable[float] = (20.0, 20.0), ny: int = 9, maxoffs: float = 5.0, num_workers: int | None = 1, phase_workers: int | None = 1, receive_mode: str = 'raw', density_normalization: Literal['legacy', 'preserve'] = 'legacy', noise: NoiseSpec | Mapping[str, object] | float | int | None = None) -> ProbeCPMGImagingResult` | Compatibility alias for `run_tuned_phase_encoded_cpmg_imaging`. |
| function | `run_matched_phase_encoded_cpmg_imaging(rho: Iterable[float] | np.ndarray | ImagingFieldMaps, *, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, num_echoes: int = 2, echo_spacing_seconds: float = 0.0002, gradient_duration_seconds: float = 0.0005, fov: tuple[float, float] | Iterable[float] = (20.0, 20.0), ny: int = 9, maxoffs: float = 5.0, num_workers: int | None = 1, phase_workers: int | None = 1, density_normalization: Literal['legacy', 'preserve'] = 'legacy', noise: NoiseSpec | Mapping[str, object] | float | int | None = None) -> ProbeCPMGImagingResult` | Run a compact matched-probe phase-encoded CPMG imaging simulation. |
| function | `run_matched_cpmg_imaging(rho: Iterable[float] | np.ndarray | ImagingFieldMaps, *, t1_map: Iterable[float] | np.ndarray | None = None, t2_map: Iterable[float] | np.ndarray | None = None, num_echoes: int = 2, echo_spacing_seconds: float = 0.0002, gradient_duration_seconds: float = 0.0005, fov: tuple[float, float] | Iterable[float] = (20.0, 20.0), ny: int = 9, maxoffs: float = 5.0, num_workers: int | None = 1, phase_workers: int | None = 1, density_normalization: Literal['legacy', 'preserve'] = 'legacy', noise: NoiseSpec | Mapping[str, object] | float | int | None = None) -> ProbeCPMGImagingResult` | Compatibility alias for `run_matched_phase_encoded_cpmg_imaging`. |

## `spin_dynamics.workflows.imaging_types`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `IdealCPMGImagingResult` | Ideal-probe CPMG imaging result. |
| class | `ProbeCPMGImagingResult` | Probe-aware CPMG imaging result. |
| class | `ImagingEchoFitResult` | Voxel-wise mono-exponential fit of reconstructed echo magnitudes. |
| class | `ImagingNoiseStatistics` | Repeated-trial image-domain noise summary. |
| class | `ImagingFieldMaps` | Spatial sample and field maps for CPMG imaging workflows. |

## `spin_dynamics.workflows.sweeps`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CPMGParameterSweepResult` | Result for probe-parameter CPMG sweeps. |
| class | `ZMagnetizationSweepResult` | Result for matched-probe z-magnetization sweeps. |
| class | `CPMGFiniteParameterSweepResult` | Result for finite-train probe-parameter sweeps. |
| function | `run_tuned_q_sweep(q_values: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1) -> CPMGParameterSweepResult` | Sweep tuned-probe coil Q for the original/reference CPMG path. |
| function | `run_matched_q_sweep(q_values: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1) -> CPMGParameterSweepResult` | Sweep matched-probe coil Q for the original/reference CPMG path. |
| function | `run_tuned_mistuning_sweep(offsets: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1) -> CPMGParameterSweepResult` | Sweep tuned-probe frequency error in units of `fin / Q`. |
| function | `run_matched_mistuning_sweep(offsets: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1) -> CPMGParameterSweepResult` | Sweep matched-probe frequency error in units of `fin / Q`. |
| function | `run_matched_z_magnetization_q_sweep(q_values: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1) -> ZMagnetizationSweepResult` | Sweep matched-probe coil Q and return excitation z magnetization. |
| function | `run_tuned_finite_q_sweep(q_values: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, num_workers: int | None = 1, sweep_workers: int | None = 1, auto_refine_grid: bool = True, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGFiniteParameterSweepResult` | Sweep tuned-probe Q for finite CPMG echo trains. |
| function | `run_untuned_finite_q_sweep(q_values: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, num_workers: int | None = 1, sweep_workers: int | None = 1, auto_refine_grid: bool = True, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGFiniteParameterSweepResult` | Sweep untuned-probe Q for finite CPMG echo trains. |
| function | `run_matched_finite_q_sweep(q_values: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, num_workers: int | None = 1, sweep_workers: int | None = 1, auto_refine_grid: bool = True, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGFiniteParameterSweepResult` | Sweep matched-probe Q for finite CPMG echo trains. |
| function | `run_tuned_finite_mistuning_sweep(offsets: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, num_workers: int | None = 1, sweep_workers: int | None = 1, auto_refine_grid: bool = True, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGFiniteParameterSweepResult` | Sweep tuned-probe mistuning for finite CPMG echo trains. |
| function | `run_untuned_finite_mistuning_sweep(offsets: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, num_workers: int | None = 1, sweep_workers: int | None = 1, auto_refine_grid: bool = True, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGFiniteParameterSweepResult` | Sweep untuned-probe mistuning for finite CPMG echo trains. |
| function | `run_matched_finite_mistuning_sweep(offsets: Iterable[float] | np.ndarray | None = None, *, numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, num_workers: int | None = 1, sweep_workers: int | None = 1, auto_refine_grid: bool = True, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGFiniteParameterSweepResult` | Sweep matched-probe mistuning for finite CPMG echo trains. |

## `spin_dynamics.workflows.time_varying`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `IdealTimeVaryingCPMGResult` | Final-echo result for ideal CPMG with time-varying B0 offsets. |
| class | `IdealTimeVaryingSweepResult` | Amplitude sweep result for ideal time-varying-field CPMG. |
| class | `ProbeTimeVaryingCPMGResult` | Final-echo result for probe-aware CPMG with time-varying B0 offsets. |
| class | `ProbeTimeVaryingSweepResult` | Amplitude sweep result for probe-aware time-varying-field CPMG. |
| function | `run_ideal_time_varying_cpmg_final(field_offsets: Iterable[float] | np.ndarray, *, numpts: int = 101, maxoffs: float = 10.0, pulse_name: str = 'rect180', t1_seconds: float = 100000000.0, t2_seconds: float = 100000000.0, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> IdealTimeVaryingCPMGResult` | Run the final echo of an ideal CPMG train with per-echo B0 offsets. |
| function | `run_tuned_time_varying_cpmg_final(field_offsets: Iterable[float] | np.ndarray, **kwargs) -> ProbeTimeVaryingCPMGResult` | Run the final echo of a tuned-probe CPMG train with per-echo B0 offsets. |
| function | `run_untuned_time_varying_cpmg_final(field_offsets: Iterable[float] | np.ndarray, **kwargs) -> ProbeTimeVaryingCPMGResult` | Run the final echo of an untuned-probe CPMG train with per-echo B0 offsets. |
| function | `run_matched_time_varying_cpmg_final(field_offsets: Iterable[float] | np.ndarray, **kwargs) -> ProbeTimeVaryingCPMGResult` | Run the final echo of a matched-probe CPMG train with per-echo B0 offsets. |
| function | `sinusoidal_field_waveform(num_echoes: int, cycles: float = 0.5) -> np.ndarray` | Return the default sinusoidal normalized B0 waveform used by v0crit. |
| function | `run_ideal_time_varying_amplitude_sweep(amplitudes: Iterable[float] | np.ndarray | None = None, *, waveform: Iterable[float] | np.ndarray | None = None, num_echoes: int = 16, numpts: int = 101, maxoffs: float = 10.0, pulse_name: str = 'rect180', num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> IdealTimeVaryingSweepResult` | Sweep normalized B0 fluctuation amplitude for ideal CPMG final echoes. |
| function | `run_tuned_time_varying_amplitude_sweep(amplitudes: Iterable[float] | np.ndarray | None = None, **kwargs) -> ProbeTimeVaryingSweepResult` | Sweep normalized B0 fluctuation amplitude for tuned-probe CPMG. |
| function | `run_untuned_time_varying_amplitude_sweep(amplitudes: Iterable[float] | np.ndarray | None = None, **kwargs) -> ProbeTimeVaryingSweepResult` | Sweep normalized B0 fluctuation amplitude for untuned-probe CPMG. |
| function | `run_matched_time_varying_amplitude_sweep(amplitudes: Iterable[float] | np.ndarray | None = None, **kwargs) -> ProbeTimeVaryingSweepResult` | Sweep normalized B0 fluctuation amplitude for matched-probe CPMG. |

## `spin_dynamics.workflows.wurst`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `WURSTInversionResult` | Isochromat magnetization after a WURST inversion pulse. |
| class | `MatchedWURSTCPMGResult` | Matched-probe WURST excitation followed by a finite CPMG train. |
| function | `run_ideal_wurst_inversion(*, numpts: int = 101, maxoffs: float = 10.0, t90_seconds: float = 2.5e-05, duration_seconds: float | None = None, sweep_width_normalized: float = 20.0, num_steps: int = 256, order: int = 20, amplitude: float = 1.0, initial_phase: float = np.pi / 2) -> WURSTInversionResult` | Run an ideal-probe WURST inversion pulse over a uniform offset grid. |
| function | `run_matched_wurst_inversion(*, numpts: int = 101, maxoffs: float = 10.0, q_value: float | None = None, t1_seconds: float = 100000000.0, t2_seconds: float = 100000000.0, duration_seconds: float | None = None, sweep_width_normalized: float = 20.0, num_steps: int = 128, order: int = 20, amplitude: float = 1.0, initial_phase: float = np.pi / 2) -> WURSTInversionResult` | Run a matched-probe WURST inversion pulse over a uniform offset grid. |
| function | `run_matched_wurst_cpmg(*, num_echoes: int = 4, numpts: int = 101, maxoffs: float = 10.0, q_value: float | None = None, t1_seconds: float = 100000000.0, t2_seconds: float = 100000000.0, duration_seconds: float | None = None, sweep_width_normalized: float = 20.0, num_steps: int = 128, order: int = 20, amplitude: float = 1.0, initial_phase: float = np.pi / 2, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> MatchedWURSTCPMGResult` | Run matched-probe WURST excitation followed by rectangular CPMG echoes. |
