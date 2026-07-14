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
| function | `invert_t2_t2(data: np.ndarray, encode_times: np.ndarray, detect_times: np.ndarray, t2_axis_encode: np.ndarray, t2_axis_detect: np.ndarray | None = None, **kwargs) -> ILTResult2D` | Convenience wrapper for a T2-T2 (relaxation exchange) inverse transform. |

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

## `spin_dynamics.absolute_phase`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `PulseShape` | Piecewise-constant rotating-frame pulse shape. |
| class | `SinusoidalTransientPerturbation` | Simple absolute-phase-dependent pulse perturbation. |
| class | `LongitudinalPhaseKick` | Absolute-phase-dependent z phase shift after an RF pulse. |
| class | `PulseShapeLibrary` | Absolute-phase-indexed library of rotating-frame pulse shapes. |
| class | `InterpolatedPulseShapeModel` | Absolute-phase model backed by a pulse-shape library. |
| class | `AbsolutePhaseSpec` | Laboratory-frame RF phase configuration. |
| class | `AbsolutePhaseMetadata` | Absolute phase values used by a sequence simulation. |
| class | `FiniteCPMGPhaseSchedule` | Absolute phase schedule for a finite CPMG-like echo train. |
| class | `FiniteCPMGPulsePlan` | Pulse-matrix indices for finite CPMG phase cycling. |
| function | `wrap_phase(phase_rad: float | np.ndarray) -> float | np.ndarray` | Wrap phase into the interval [0, 2*pi). |
| function | `phase_bin_indices(phase_rad: float | np.ndarray, phase_bins: int) -> int | np.ndarray` | Return nearest uniform absolute-phase bin indices. |
| function | `quantize_phase_to_bins(phase_rad: float | np.ndarray, phase_bins: int | None) -> float | np.ndarray` | Return phase values snapped to a uniform absolute-phase grid. |
| function | `sinusoidal_transient_from_mapping(value: Mapping[str, Any]) -> SinusoidalTransientPerturbation` | Build a sinusoidal perturbation model from a mapping. |
| function | `longitudinal_phase_kick_from_mapping(value: Mapping[str, Any]) -> LongitudinalPhaseKick` | Build a longitudinal phase-kick model from a mapping. |
| function | `pulse_shape_library_from_mapping(value: Mapping[str, Any]) -> PulseShapeLibrary` | Build a pulse-shape library from plain arrays. |
| function | `interpolated_pulse_model_from_mapping(value: Mapping[str, Any]) -> InterpolatedPulseShapeModel` | Build an interpolated pulse-shape model from a mapping. |
| function | `build_nonresonant_circuit_pulse_library(*, absolute_phase_rad: Sequence[float] | np.ndarray, rf_frequency_hz: float, pulse_duration_seconds: float, time_scale_rad_per_s: float, tau_seconds: float, rotating_phase_rad: float = 0.0, post_delay_seconds: float = 0.0, pulse_kind: str = 'refocusing', segment_duration_seconds: float | None = None, samples_per_segment: int = 16, max_step_seconds: float | None = None) -> PulseShapeLibrary` | Build a first-order non-resonant transmit pulse-shape library. |
| function | `build_tuned_resonator_pulse_library(*, absolute_phase_rad: Sequence[float] | np.ndarray, rf_frequency_hz: float, pulse_duration_seconds: float, time_scale_rad_per_s: float, resonant_frequency_hz: float, quality_factor: float, rotating_phase_rad: float = 0.0, post_delay_seconds: float = 0.0, pulse_kind: str = 'refocusing', segment_duration_seconds: float | None = None, samples_per_segment: int = 16, max_step_seconds: float | None = None) -> PulseShapeLibrary` | Build a second-order tuned-resonator transmit pulse-shape library. |
| function | `nonresonant_dc_phase_perturbation(*, nutation_rate_rad_s: float, tau_seconds: float, longitudinal_fraction: float, phase_offset_rad: float = 0.0, applies_to: str = 'refocusing') -> LongitudinalPhaseKick` | Return the simple non-resonant DC transient phase-shift model. |
| function | `as_absolute_phase_spec(value: AbsolutePhaseSpec | Mapping[str, Any] | None) -> AbsolutePhaseSpec | None` | Normalize a user absolute-phase specification. |
| function | `apply_absolute_phase_model(shape: PulseShape, spec: AbsolutePhaseSpec | None, absolute_phase_rad: float, pulse_kind: str) -> PulseShape` | Apply the optional absolute-phase transient model to a pulse shape. |
| function | `cpmg_refocus_start_times(*, excitation_start_seconds: float, excitation_duration_seconds: float, correction_delay_seconds: float, pre_refocus_delay_seconds: float, echo_spacing_seconds: float, num_echoes: int) -> np.ndarray` | Return refocusing-pulse start times for a CPMG-like train. |
| function | `build_finite_cpmg_phase_schedule(*, spec: AbsolutePhaseSpec, excitation_start_seconds: float, excitation_duration_seconds: float, correction_delay_seconds: float, pre_refocus_delay_seconds: float, echo_spacing_seconds: float, num_echoes: int, excitation_phases_rad: Sequence[float] | np.ndarray = (np.pi / 2, 3 * np.pi / 2), refocus_rotating_phase_rad: float = 0.0) -> FiniteCPMGPhaseSchedule` | Build absolute RF phase schedule for a finite CPMG-like train. |
| function | `build_finite_cpmg_pulse_plan(num_echoes: int, *, per_echo_refocusing: bool) -> FiniteCPMGPulsePlan` | Return pulse-matrix index vectors for finite CPMG phase cycling. |
| function | `build_cpmg_absolute_phase_metadata(*, spec: AbsolutePhaseSpec, excitation_start_seconds: float, excitation_phases_rad: np.ndarray, refocus_start_seconds: np.ndarray, refocus_rotating_phase_rad: float, echo_spacing_seconds: float, pulse_matrix_count: int, schedule: FiniteCPMGPhaseSchedule | None = None, pulse_plan: FiniteCPMGPulsePlan | None = None, refocus_phase_bin: np.ndarray | None = None, refocus_matrix_phase_rad: np.ndarray | None = None, unique_refocus_phase_rad: np.ndarray | None = None, refocus_pulse_library: PulseShapeLibrary | None = None) -> AbsolutePhaseMetadata` | Build metadata for a CPMG-like absolute-phase schedule. |

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

## `spin_dynamics.coupling.isotopes`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `nuclear_isotope(isotope: str) -> NuclearIsotope` | Return the registry entry (spin and ``|gamma|/2pi``) for ``isotope``. |
| function | `larmor_frequency_hz(isotope: str, b0_tesla: float, *, gamma_hz_per_t: float | None = None) -> float` | Return the Larmor frequency ``gamma * B0`` in hertz for ``isotope``. |

## `spin_dynamics.coupling.j_editing`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `JEditingFitResult` | Known-J least-squares fit of a J-modulation curve. |
| function | `j_modulation_curve(encoding_times: Iterable[float] | np.ndarray, couplings_hz: Iterable[float] | np.ndarray, amplitudes: Iterable[float] | np.ndarray | None = None, *, cycles: int = 1, background: float = 0.0, powers: Iterable[int] | np.ndarray | None = None) -> np.ndarray` | Return a superposition of J-modulated cosine components. |
| function | `carbon_detected_j_modulation(encoding_times: Iterable[float] | np.ndarray, couplings_hz: Iterable[float] | np.ndarray, abundances: Iterable[float] | np.ndarray, proton_counts: Iterable[int] | np.ndarray, *, cycles: int = 1, scale: float = 1.0) -> np.ndarray` | Return the carbon-detected low-field J-editing model. |
| function | `proton_detected_j_modulation(encoding_times: Iterable[float] | np.ndarray, couplings_hz: Iterable[float] | np.ndarray, amplitudes: Iterable[float] | np.ndarray, *, cycles: int = 1, background: float = 0.0) -> np.ndarray` | Return the proton-detected J-editing model. |
| function | `tango_b_filter(couplings_hz: Iterable[float] | np.ndarray, *, delay_seconds: float | None = None, target_coupling_hz: float | None = None, order: int = 1) -> np.ndarray` | Return the ideal TANGO-B coupled-spin transverse filter amplitude. |
| function | `fit_known_j_spectrum(encoding_times: Iterable[float] | np.ndarray, signal: Iterable[float] | np.ndarray, couplings_hz: Iterable[float] | np.ndarray, *, cycles: int = 1, powers: Iterable[int] | np.ndarray | None = None, include_background: bool = True) -> JEditingFitResult` | Fit amplitudes for a known set of J-coupling frequencies. |

## `spin_dynamics.coupling.mixed_operators`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `hilbert_dimension(spins: Sequence[float] | Iterable[float]) -> int` | Return the tensor-product Hilbert dimension ``prod(2 I_k + 1)``. |
| function | `embedded_operator(spins: Sequence[float] | Iterable[float], index: int, axis: str) -> np.ndarray` | Return a single-spin operator embedded in the full mixed Hilbert space. |
| function | `product_operator(spins: Sequence[float] | Iterable[float], terms: Iterable[tuple[int, str]]) -> np.ndarray` | Return a product operator such as ``I1z I2z`` for mixed spins. |
| function | `total_operator(spins: Sequence[float] | Iterable[float], axis: str, indices: Iterable[int] | None = None) -> np.ndarray` | Return the sum of selected single-spin operators along one axis. |
| function | `dot_product_operator(spins: Sequence[float] | Iterable[float], index_a: int, index_b: int) -> np.ndarray` | Return the scalar ``I_a . I_b`` operator for two distinct spins. |

## `spin_dynamics.coupling.multinuclear`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MultinuclearSite` | One nucleus in a heteronuclear J-coupled system. |
| class | `MultinuclearSpinSystem` | Heteronuclear scalar-coupled spin system at a fixed field ``b0_tesla``. |
| function | `multinuclear_system(isotopes: Sequence[str], couplings_hz: Iterable[Iterable[float]], b0_tesla: float, *, labels: Sequence[str] | None = None, spins: Sequence[float] | None = None, gammas_hz_per_t: Sequence[float] | None = None) -> MultinuclearSpinSystem` | Build a validated heteronuclear system from isotope names and couplings. |
| function | `multinuclear_hamiltonian(system: MultinuclearSpinSystem, *, coupling: str = 'isotropic', include_zeeman: bool = True) -> np.ndarray` | Return the static lab-frame Hamiltonian in radians per second. |
| function | `multinuclear_rf_hamiltonian(system: MultinuclearSpinSystem, nutation_hz: float, *, phase: float = 0.0, indices: Iterable[int] | None = None) -> np.ndarray` | Return an RF Hamiltonian (rad/s) driving selected spins at one amplitude. |
| function | `multinuclear_equilibrium_density(system: MultinuclearSpinSystem, *, axis: str = 'z', gamma_weighted: bool = True) -> np.ndarray` | Return the high-temperature equilibrium density deviation. |
| function | `multinuclear_quadrupolar_rates(system: MultinuclearSpinSystem, *, correlation_time_seconds: float, quadrupole_coupling_hz: float | dict[str, float], asymmetry: float | dict[str, float] = 0.0, spin_half_rate_hz: float = 0.0) -> tuple[np.ndarray, np.ndarray]` | Return per-site ``(r1, r2)`` arrays for a shared rotational ``tau_c``. |
| function | `per_spin_relaxation_superoperator(spins: Sequence[float] | Iterable[float], r1_per_second: float | Sequence[float] | np.ndarray, r2_per_second: float | Sequence[float] | np.ndarray) -> np.ndarray` | Return a per-spin phenomenological ``R1``/``R2`` relaxation superoperator. |

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
| function | `two_spin_slic_matching_nutation_hz(coupling_hz: float) -> float` | Return the ideal two-spin SLIC matching nutation frequency. |
| function | `two_spin_slic_transfer_time(offset_difference_hz: float) -> float` | Return the ideal maximum-transfer time at the SLIC crossing. |
| function | `simulate_slic_spectrum(system: CoupledSpinSystem, nutation_frequencies_hz: Iterable[float] | np.ndarray, *, spin_lock_time: float, initial_axis: str = 'x', detect_axis: str = 'x') -> SLICSpectrumResult` | Simulate remaining transverse magnetization after a spin-lock pulse. |

## `spin_dynamics.coupling.systems`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SpinSite` | One spin-1/2 site in a coupled spin system. |
| class | `CoupledSpinSystem` | Small dense spin-1/2 system with scalar couplings in hertz. |
| function | `coupled_spin_system(offsets_hz: Iterable[float], couplings_hz: Iterable[Iterable[float]], *, labels: Sequence[str] | None = None, isotopes: Sequence[str] | None = None) -> CoupledSpinSystem` | Build a validated spin-1/2 system from offsets and couplings. |

## `spin_dynamics.coupling.zulf`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ZulfSpectrum` | Time-domain FID and its J-coupled spectrum. |
| function | `simulate_zulf_fid(system: MultinuclearSpinSystem, *, r1_per_second: float | Sequence[float] | np.ndarray, r2_per_second: float | Sequence[float] | np.ndarray, dwell_seconds: float, n_points: int, excite_indices: Iterable[int] | None = None, detect_indices: Iterable[int] | None = None, flip_rad: float = np.pi / 2.0, phase_rad: float = 0.0, gamma_weighted: bool = True, coupling: str = 'isotropic') -> tuple[np.ndarray, np.ndarray]` | Return ``(times, fid)`` for a ZULF ``pi/2``-acquire experiment. |
| function | `simulate_zulf_spectrum(system: MultinuclearSpinSystem, *, r1_per_second: float | Sequence[float] | np.ndarray, r2_per_second: float | Sequence[float] | np.ndarray, dwell_seconds: float, n_points: int, excite_indices: Iterable[int] | None = None, detect_indices: Iterable[int] | None = None, flip_rad: float = np.pi / 2.0, phase_rad: float = 0.0, gamma_weighted: bool = True, apodization_hz: float = 0.0, coupling: str = 'isotropic') -> ZulfSpectrum` | Return the J-coupled spectrum of a ZULF ``pi/2``-acquire experiment. |

## `spin_dynamics.composition`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `convert_units(values: Any, source_unit: str, target_unit: str) -> np.ndarray` | Convert values between units used at package component boundaries. |
| class | `SpatialGrid` | Named rectilinear spatial axes in metres. |
| class | `TimeAxis` | A strictly increasing absolute time axis in seconds. |
| class | `FieldSolutionAdapter` | Named field channels sampled on a shared spatial grid. |
| class | `ThermalStateAdapter` | Temperature channels in kelvin on an explicit time base. |
| class | `FlowFieldAdapter` | Eulerian velocity in m/s, optionally varying over an absolute time axis. |
| class | `HardwareResponseAdapter` | Typed bridge from a uniformly sampled channel to an LTI response. |
| class | `SequenceTimelineAdapter` | Sequence channels on one absolute, interval-centered time axis. |

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
| function | `set_arb10_backend(name: str) -> None` | Select the default backend for ``sim_spin_dynamics_arb10``. |
| function | `get_arb10_backend() -> str` | Return the current default ``sim_spin_dynamics_arb10`` backend. |
| class | `Arb10Parameters` | Parameters for `sim_spin_dynamics_arb10`. |
| class | `Arb10DiffusionParameters` | Parameters for `sim_spin_dynamics_arb10_diffusion`. |
| class | `Arb7Parameters` | Parameters for `sim_spin_dynamics_arb7`. |
| function | `sim_spin_dynamics_arb10(params: Mapping[str, Any] | Arb10Parameters | Any, *, backend: str | None = None) -> np.ndarray` | Simulate arbitrary-pulse spin dynamics with precomputed pulse matrices. |
| function | `sim_spin_dynamics_arb10_batched(params_list: Sequence[Mapping[str, Any] | Arb10Parameters | Any]) -> np.ndarray` | Run many same-structured arb10 simulations in one vmapped JAX call. |
| function | `sim_spin_dynamics_arb10_radiation_damping(params: Mapping[str, Any] | Arb10Parameters | Any, radiation_damping: RadiationDampingSpec) -> np.ndarray` | Simulate `arb10` with ensemble radiation damping during free intervals. |
| function | `sim_spin_dynamics_arb10_diffusion(params: Mapping[str, Any] | Arb10DiffusionParameters | Any) -> np.ndarray` | Simulate arbitrary-pulse dynamics with a diffusion free-precession term. |
| function | `sim_spin_dynamics_arb10_chunked(params: Mapping[str, Any] | Arb10Parameters | Any, num_workers: int | None = None, min_chunk_size: int = 8192) -> np.ndarray` | Run `sim_spin_dynamics_arb10` on contiguous isochromat chunks. |
| function | `sim_spin_dynamics_arb10_diffusion_chunked(params: Mapping[str, Any] | Arb10DiffusionParameters | Any, num_workers: int | None = None, min_chunk_size: int = 8192) -> np.ndarray` | Run `sim_spin_dynamics_arb10_diffusion` on isochromat chunks. |
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

## `spin_dynamics.detection.base`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `Detector` | A frequency-domain detector referred to magnetic field at the sensor. |
| class | `DetectedFieldSNR` | Result of a field-referred matched-filter SNR estimate. |
| function | `field_referred_from_output(output_psd, transfer)` | Refer an output-referred (volts^2/Hz) noise PSD to field via ``H``. |
| function | `detected_field_snr(field_spectrum, freqs, detector, *, df = None) -> DetectedFieldSNR` | Matched-filter detected SNR of a field-at-sensor spectrum. |
| function | `snr_from_field_noise_psd(field_spectrum, freqs, field_noise_psd, *, df = None) -> DetectedFieldSNR` | Matched-filter SNR of ``S(f)`` against an explicit field-noise PSD ``N^2(f)``. |

## `spin_dynamics.detection.gradiometer`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `Gradiometer` | A coaxial pickup: loops at ``positions_m`` with turns/sign ``weights``. |

## `spin_dynamics.detection.inductive`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `InductiveCoilDetector` | Field-referred adapter over an inductive probe's output-noise density. |
| class | `IdealFaradayCoil` | Idealized inductive coil with ``1/f`` field-referred noise. |

## `spin_dynamics.detection.opm`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `OPMMagnetometer` | Atomic magnetometer with a Lorentzian atomic-response bandwidth. |

## `spin_dynamics.detection.spatial`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `AmbientFieldSource` | A stochastic ambient field source with a field-noise PSD. |
| function | `pickup_signal_spectrum(pickup, positions, moment_spectra, *, reference = None) -> np.ndarray` | Pickup-weighted field-at-sensor spectrum ``S(f) = sum_k g(r_k) m_k(f)``. |
| function | `spatial_field_noise_psd(detector, freqs, ambient_sources, *, pickup = None, reference = None) -> np.ndarray` | Sensor floor augmented by ambient sources coupling through the pickup. |
| function | `detected_field_snr_spatial(detector, positions, moment_spectra, freqs, *, ambient_sources = None, pickup = None, reference = None, df = None) -> DetectedFieldSNR` | Detected SNR of a distributed sample through ``detector`` + its pickup. |

## `spin_dynamics.detection.squid`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SQUIDMagnetometer` | Untuned SQUID magnetometer: flat field-noise floor with a ``1/f`` knee. |

## `spin_dynamics.deprecation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SpinDynamicsDeprecationWarning` | Visible warning for a PythonSpinDynamics API scheduled for removal. |
| class | `DeprecationInfo` | Machine-readable lifecycle information attached to deprecated callables. |
| function | `warn_deprecated(name: str, *, since: str, removal: str, alternative: str, stacklevel: int = 2) -> None` | Emit the standard visible warning for a deprecated public API. |
| function | `deprecated(*, since: str, removal: str, alternative: str) -> Callable[[Callable[P, R]], Callable[P, R]]` | Mark a callable deprecated while preserving its signature and metadata. |

## `spin_dynamics.esr.deer`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `DeerKernel` | Powder-averaged DEER kernel mapping a distance distribution to a trace. |
| class | `DeerDistanceResult` | Recovered distance distribution from a DEER form factor. |
| function | `deer_pair_trace(times_seconds, distance_nm: float, theta_rad: float, *, lambda_depth: float = 1.0, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> np.ndarray` | Return the DEER form factor for a single spin pair at fixed orientation. |
| function | `deer_powder_kernel(times_seconds, distances_nm, *, lambda_depth: float = 1.0, n_theta: int = 2001, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> DeerKernel` | Return the powder-averaged DEER kernel ``K[t, r]``. |
| function | `gaussian_distance_distribution(distances_nm, mean_nm: float, sigma_nm: float) -> np.ndarray` | Return a normalized Gaussian distance distribution on a distance grid. |
| function | `simulate_deer(times_seconds, distances_nm, distribution, *, lambda_depth: float = 1.0, n_theta: int = 2001, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> np.ndarray` | Return the DEER form factor for a distance distribution. |
| function | `deer_dipolar_spectrum(times_seconds, form_factor, *, zero_fill: int = 4) -> tuple[np.ndarray, np.ndarray]` | Return the dipolar (Pake) spectrum of a DEER form factor. |
| function | `extract_distance_distribution(times_seconds, form_factor, distances_nm, *, lambda_depth: float = 1.0, snr: float = 100.0, n_theta: int = 2001, regularization_order: int = 2, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> DeerDistanceResult` | Recover a distance distribution ``P(r)`` from a DEER form factor. |
| function | `deer_pair_trace_quantum(times_seconds, distance_nm: float, theta_rad: float, *, pump_flip_rad: float = np.pi, tau1_seconds: float = 2e-07, tau2_seconds: float = 2e-06, observer_offset_hz: float = 5000000.0, pump_offset_hz: float = 0.0, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> np.ndarray` | Simulate the four-pulse DEER form factor from the spin Hamiltonian. |

## `spin_dynamics.esr.dipolar`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `dipolar_constant_hz_nm3(g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> float` | Return the perpendicular dipolar constant ``nu_perp * r^3`` in Hz nm^3. |
| function | `dipolar_frequency_hz(distance_nm: float | np.ndarray, *, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> float | np.ndarray` | Return the perpendicular dipolar frequency ``nu_perp(r)`` in Hz. |
| function | `distance_from_dipolar_frequency_nm(frequency_hz: float | np.ndarray, *, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> float | np.ndarray` | Return the distance (nm) for a perpendicular dipolar frequency in Hz. |
| function | `dipolar_angular_frequency_hz(distance_nm: float | np.ndarray, theta_rad: float | np.ndarray, *, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> float | np.ndarray` | Return the orientation-dependent dipolar frequency in Hz. |
| function | `secular_dipolar_hamiltonian(distance_nm: float, theta_rad: float, *, g_a: float = FREE_ELECTRON_G, g_b: float = FREE_ELECTRON_G) -> np.ndarray` | Return the secular two-electron dipolar Hamiltonian in radians per second. |

## `spin_dynamics.esr.distributions`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ESRDistributionSample` | One weighted ESR static-disorder sample. |
| class | `ESRFieldDistributionResult` | Field-swept ESR spectrum averaged over static disorder samples. |
| class | `ESRFrequencyDistributionResult` | Frequency-swept ESR spectrum averaged over static disorder samples. |
| function | `normalize_distribution(samples: list[ESRDistributionSample] | tuple[ESRDistributionSample, ...]) -> tuple[ESRDistributionSample, ...]` | Return static-disorder samples with weights normalized to unity. |
| function | `static_disorder_grid(system: ESRSpinSystem, *, g_std: float | np.ndarray | list[float] | tuple[float, ...] = 0.0, field_std_tesla: float = 0.0, g_points: int = 3, field_points: int = 5, n_sigma: float = 2.0) -> tuple[ESRDistributionSample, ...]` | Return weighted samples for diagonal ``g`` strain and field offsets. |
| function | `simulate_field_sweep_distribution(samples: list[ESRDistributionSample] | tuple[ESRDistributionSample, ...], microwave_frequency_hz: float, *, orientations: OrientationInput = 'single', broadening_tesla: float = 0.0001, points: int = 1024, span_tesla: float | None = None, fields_tesla: np.ndarray | list[float] | tuple[float, ...] | None = None, lineshape: str = 'gaussian', detection_mode: str = 'absorption') -> ESRFieldDistributionResult` | Return a field-swept ESR spectrum averaged over static disorder. |
| function | `simulate_frequency_spectrum_distribution(samples: list[ESRDistributionSample] | tuple[ESRDistributionSample, ...], b0_tesla: float, *, orientations: OrientationInput = 'single', broadening_hz: float = 1000000.0, points: int = 1024, span_hz: float | None = None, frequencies_hz: np.ndarray | list[float] | tuple[float, ...] | None = None, lineshape: str = 'gaussian', detection_mode: str = 'absorption') -> ESRFrequencyDistributionResult` | Return a frequency-swept ESR spectrum averaged over static disorder. |

## `spin_dynamics.esr.endor`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `EndorSpectrum` | An ENDOR spectrum sampled on a radiofrequency axis. |
| function | `endor_frequencies(coupling: HyperfineCoupling) -> np.ndarray` | Return the ENDOR line positions in Hz (all manifold transitions, sorted). |
| function | `mims_blind_spot_factor(frequency_hz: float, tau_seconds: float) -> float` | Return the Mims ENDOR response factor ``sin^2(pi nu tau)`` for one line. |
| function | `davies_endor_spectrum(frequencies_hz, coupling: HyperfineCoupling, *, linewidth_hz: float = 100000.0) -> EndorSpectrum` | Return a Davies ENDOR spectrum (all lines with equal weight, no blind spots). |
| function | `mims_endor_spectrum(frequencies_hz, coupling: HyperfineCoupling, *, tau_seconds: float, linewidth_hz: float = 100000.0) -> EndorSpectrum` | Return a Mims ENDOR spectrum with ``tau``-dependent blind-spot weighting. |

## `spin_dynamics.esr.eseem`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `HyperfineCoupling` | Secular model of one S=1/2 electron coupled to one nucleus. |
| function | `filter_electron_coherence(density: np.ndarray, order: int, *, nuclear_spin: float = 0.5) -> np.ndarray` | Keep only the requested electron coherence order of a density matrix. |
| function | `electron_nuclear_hamiltonian(coupling: HyperfineCoupling, *, electron_offset_hz: float = 0.0) -> np.ndarray` | Return the secular electron-nuclear Hamiltonian in radians per second. |
| function | `manifold_frequencies(coupling: HyperfineCoupling) -> tuple[np.ndarray, np.ndarray]` | Return nuclear transition frequencies in the two electron manifolds. |
| function | `nuclear_frequencies(coupling: HyperfineCoupling) -> tuple[float, float]` | Return the two spin-1/2 nuclear frequencies ``(nu_alpha, nu_beta)`` in Hz. |
| function | `modulation_depth(coupling: HyperfineCoupling) -> float` | Return the spin-1/2 ESEEM modulation-depth parameter ``k`` in ``[0, 1]``. |
| function | `two_pulse_eseem(times_seconds, coupling: HyperfineCoupling) -> np.ndarray` | Return the analytic spin-1/2 two-pulse ESEEM trace ``V(tau)``. |
| function | `three_pulse_eseem(times_seconds, coupling: HyperfineCoupling, *, tau_seconds: float) -> np.ndarray` | Return the analytic spin-1/2 three-pulse (stimulated-echo) trace ``V(T)``. |
| function | `eseem_spectrum(times_seconds, signal, *, zero_fill: int = 4) -> tuple[np.ndarray, np.ndarray]` | Return the ESEEM frequency spectrum ``(frequencies_hz, magnitude)``. |
| function | `two_pulse_eseem_quantum(times_seconds, coupling: HyperfineCoupling, *, electron_offset_hz: float = 0.0) -> np.ndarray` | Density-matrix two-pulse ESEEM, normalized to ``V(0) = 1``. |
| function | `three_pulse_eseem_quantum(times_seconds, coupling: HyperfineCoupling, *, tau_seconds: float) -> np.ndarray` | Density-matrix three-pulse ESEEM, normalized to the unmodulated echo. |
| function | `three_pulse_eseem_phase_cycled(times_seconds, coupling: HyperfineCoupling, *, tau_seconds: float, n_phase: int = 4) -> np.ndarray` | Three-pulse ESEEM selected by an explicit phase cycle. |

## `spin_dynamics.esr.hamiltonians`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `effective_g_vector(system: ESRSpinSystem, b0_direction_g: np.ndarray | Sequence[float]) -> np.ndarray` | Return ``g^T n`` for a unit static-field direction in the ``g`` frame. |
| function | `effective_g_value(system: ESRSpinSystem, b0_direction_g: np.ndarray | Sequence[float]) -> float` | Return the ESR effective ``g`` value for a static-field direction. |
| function | `resonance_frequency_hz(system: ESRSpinSystem, b0_vector_tesla_g: float | np.ndarray | Sequence[float]) -> float` | Return the spin-1/2 ESR transition frequency in hertz. |
| function | `resonance_field_tesla(system: ESRSpinSystem, microwave_frequency_hz: float, b0_direction_g: np.ndarray | Sequence[float] = (0.0, 0.0, 1.0)) -> float` | Return the resonant static-field magnitude for one ESR orientation. |
| function | `zeeman_hamiltonian(system: ESRSpinSystem, b0_vector_tesla_g: np.ndarray | Sequence[float]) -> np.ndarray` | Return the electron Zeeman Hamiltonian in radians per second. |
| function | `diagonalize_system(system: ESRSpinSystem, b0_vector_tesla_g: np.ndarray | Sequence[float], *, strength_tolerance: float = 1e-12, frequency_tolerance_hz: float = 1e-09) -> ESREigensystem` | Diagonalize the ESR Zeeman Hamiltonian and return transition metadata. |

## `spin_dynamics.esr.hyperfine`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `NuclearSite` | One nucleus coupled to the ESR electron spin. |
| class | `ElectronNuclearSystem` | One electron spin coupled isotropically to one or more nuclei. |
| class | `HyperfineTransition` | One ESR-active transition in an electron-nuclear spin system. |
| class | `HyperfineEigensystem` | Energy levels, eigenvectors, and ESR transitions for a hyperfine system. |
| class | `HyperfineFieldPoint` | One field-sweep contribution from one hyperfine transition. |
| class | `HyperfineFieldSweepResult` | Field-swept ESR spectrum for an electron-nuclear hyperfine system. |
| function | `electron_nuclear_system(hyperfine_hz: Sequence[float] | np.ndarray, *, nuclei: Sequence[NuclearSite] | None = None, g_tensor: float | np.ndarray | list[float] | tuple[float, ...] = 2.00231930436256) -> ElectronNuclearSystem` | Build an electron-nuclear spin system from isotropic hyperfine constants. |
| function | `electron_operator(system: ElectronNuclearSystem, axis: str) -> np.ndarray` | Return an electron spin operator embedded in the full Hilbert space. |
| function | `nuclear_operator(system: ElectronNuclearSystem, nucleus_index: int, axis: str) -> np.ndarray` | Return a nuclear spin operator embedded in the full Hilbert space. |
| function | `zeeman_hamiltonian(system: ElectronNuclearSystem, b0_vector_tesla_g: np.ndarray | Sequence[float]) -> np.ndarray` | Return electron plus nuclear Zeeman Hamiltonian in radians per second. |
| function | `isotropic_hyperfine_hamiltonian(system: ElectronNuclearSystem) -> np.ndarray` | Return isotropic ``S . A . I`` hyperfine Hamiltonian in radians per second. |
| function | `hyperfine_hamiltonian(system: ElectronNuclearSystem, b0_vector_tesla_g: np.ndarray | Sequence[float]) -> np.ndarray` | Return Zeeman plus isotropic hyperfine Hamiltonian. |
| function | `diagonalize_hyperfine_system(system: ElectronNuclearSystem, b0_vector_tesla_g: np.ndarray | Sequence[float], *, strength_tolerance: float = 1e-12, frequency_tolerance_hz: float = 1e-09) -> HyperfineEigensystem` | Diagonalize a hyperfine Hamiltonian and return ESR-active transitions. |
| function | `simulate_hyperfine_field_sweep(system: ElectronNuclearSystem, microwave_frequency_hz: float, *, b0_direction_g: np.ndarray | Sequence[float] = (0.0, 0.0, 1.0), b1_direction_g: np.ndarray | Sequence[float] = (1.0, 0.0, 0.0), broadening_hz: float = 1000000.0, points: int = 1024, span_tesla: float | None = None, fields_tesla: np.ndarray | list[float] | tuple[float, ...] | None = None, lineshape: str = 'gaussian', detection_mode: str = 'absorption', intensity_tolerance: float = 1e-14) -> HyperfineFieldSweepResult` | Return a field-swept ESR spectrum including isotropic hyperfine coupling. |

## `spin_dynamics.esr.hyscore`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `HyscoreSpectrum` | 2D HYSCORE spectrum on centered frequency axes. |
| function | `hyscore_signal(t1_seconds, t2_seconds, coupling: HyperfineCoupling, *, tau_seconds: float) -> np.ndarray` | Return the 2D HYSCORE time-domain signal ``V[t1, t2]``. |
| function | `cross_peak_positions(coupling: HyperfineCoupling) -> tuple[tuple[float, float], ...]` | Return the HYSCORE cross-peak positions in Hz. |
| function | `hyscore_spectrum(t1_seconds, t2_seconds, signal, *, zero_fill: int = 4) -> HyscoreSpectrum` | Return the 2D HYSCORE magnitude spectrum on centered frequency axes. |

## `spin_dynamics.esr.lineshapes`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `gaussian_lineshape(axis: np.ndarray, center: float, width: float, *, derivative: bool = False) -> np.ndarray` | Return a unit-height Gaussian absorption line or its first derivative. |
| function | `lorentzian_lineshape(axis: np.ndarray, center: float, width: float, *, derivative: bool = False) -> np.ndarray` | Return a unit-height Lorentzian absorption line or its first derivative. |
| function | `spectrum_from_lines(axis: np.ndarray, centers: np.ndarray | list[float] | tuple[float, ...], intensities: np.ndarray | list[float] | tuple[float, ...], *, width: float, lineshape: str = 'gaussian', detection_mode: str = 'absorption') -> np.ndarray` | Return a broadened ESR spectrum from weighted line centers. |

## `spin_dynamics.esr.orientations`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `spherical_direction(alpha: float, beta: float) -> np.ndarray` | Return a unit vector from azimuth ``alpha`` and polar angle ``beta``. |
| class | `ESROrientationSample` | One local ``g``-tensor orientation relative to lab static and RF fields. |
| function | `single_crystal_orientation(alpha: float = 0.0, beta: float = 0.0, *, b1_alpha: float | None = None, b1_beta: float | None = None) -> tuple[ESROrientationSample, ...]` | Return a one-sample ESR orientation ensemble. |
| function | `powder_average_grid(n_theta: int = 16, n_phi: int = 32, n_chi: int = 8, *, b1_b0_angle: float = np.pi / 2.0) -> tuple[ESROrientationSample, ...]` | Return a normalized ESR powder grid with correlated lab B0 and B1 axes. |
| function | `normalize_orientations(orientations: tuple[ESROrientationSample, ...] | list[ESROrientationSample]) -> tuple[ESROrientationSample, ...]` | Return ESR orientation samples with weights normalized to unity. |

## `spin_dynamics.esr.pulsed`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `equilibrium_density(levels_hz: np.ndarray) -> np.ndarray` | Return a trace-zero high-temperature ESR density matrix. |
| function | `flip_angle_duration(flip_angle_rad: float, nutation_hz: float) -> float` | Return the rectangular-pulse duration for a spin-1/2 flip angle. |
| function | `rf_operator_eigenbasis(eigensystem: ESREigensystem, direction_g = (1.0, 0.0, 0.0)) -> np.ndarray` | Return ``e1 . S`` for a unit microwave-field direction in the eigenbasis. |
| function | `rotating_indices(levels_hz: np.ndarray, rf_frequency_hz: float) -> np.ndarray` | Return two-level RWA winding numbers for a carrier frequency. |
| function | `static_hamiltonian_rotating(eigensystem: ESREigensystem, rf_frequency_hz: float) -> np.ndarray` | Return the rotating-frame static Hamiltonian in radians per second. |
| function | `pulse_hamiltonian(eigensystem: ESREigensystem, *, nutation_hz: float, rf_frequency_hz: float, phase: float = 0.0, b1_direction_g = (1.0, 0.0, 0.0)) -> np.ndarray` | Return a rectangular microwave-pulse Hamiltonian in the rotating frame. |
| function | `detection_operator(eigensystem: ESREigensystem, rf_frequency_hz: float, rx_direction_g = (1.0, 0.0, 0.0)) -> np.ndarray` | Return the baseband receive observable for the addressed ESR line. |
| class | `ESRFIDResult` | Complex baseband ESR FID from one rectangular excitation pulse. |
| class | `ESRHahnEchoResult` | Complex baseband ESR Hahn echo from one isochromat. |
| function | `simulate_fid(system: ESRSpinSystem, b0_vector_tesla_g, *, nutation_hz: float, pulse_duration_seconds: float, times_seconds, rf_frequency_hz: float | None = None, phase: float = 0.0, b1_direction_g = (1.0, 0.0, 0.0), rx_direction_g = None, t2_seconds: float = np.inf, relaxation: ESRRelaxationModel | None = None, initial_density: np.ndarray | None = None) -> ESRFIDResult` | Simulate a pulsed ESR free-induction decay in the rotating frame. |
| function | `simulate_hahn_echo(system: ESRSpinSystem, b0_vector_tesla_g, *, nutation_hz: float, excitation_duration_seconds: float, refocus_duration_seconds: float, tau_seconds: float, times_seconds, rf_frequency_hz: float | None = None, excitation_phase: float = 0.0, refocus_phase: float = np.pi / 2.0, b1_direction_g = (1.0, 0.0, 0.0), rx_direction_g = None, t2_seconds: float = np.inf, relaxation: ESRRelaxationModel | None = None, initial_density: np.ndarray | None = None) -> ESRHahnEchoResult` | Simulate a two-pulse ESR Hahn echo for one isochromat. |

## `spin_dynamics.esr.relaxation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ESRRelaxationModel` | Phenomenological ESR relaxation model in the energy eigenbasis. |
| function | `matrix_exponential(matrix: np.ndarray, duration: float = 1.0) -> np.ndarray` | Return ``exp(matrix * duration)`` for a small dense matrix. |
| function | `liouville_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray` | Return the commutator Liouvillian for column-stacked density matrices. |
| function | `relaxation_superoperator(dimension: int, model: ESRRelaxationModel) -> np.ndarray` | Return a trace-preserving ESR relaxation superoperator. |
| function | `liouville_superoperator(hamiltonian: np.ndarray, model: ESRRelaxationModel | None = None) -> np.ndarray` | Return Hamiltonian plus optional ESR relaxation Liouvillian. |
| function | `propagate_density_liouville(density: np.ndarray, hamiltonian: np.ndarray, duration: float, *, relaxation: ESRRelaxationModel | None = None) -> np.ndarray` | Propagate a density matrix with Hamiltonian and ESR relaxation. |

## `spin_dynamics.esr.spectra`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ESRLine` | One orientation-resolved ESR transition line. |
| class | `ESRFrequencySpectrumResult` | Fixed-field ESR spectrum on a frequency axis. |
| class | `ESRFieldSweepResult` | Fixed-frequency ESR spectrum on a static-field axis. |
| function | `simulate_frequency_spectrum(system: ESRSpinSystem, b0_tesla: float, *, orientations: OrientationInput = 'single', broadening_hz: float = 1000000.0, points: int = 1024, span_hz: float | None = None, frequencies_hz: np.ndarray | list[float] | tuple[float, ...] | None = None, lineshape: str = 'gaussian', detection_mode: str = 'absorption') -> ESRFrequencySpectrumResult` | Return a broadened fixed-field ESR spectrum on a frequency axis. |
| function | `simulate_field_sweep(system: ESRSpinSystem, microwave_frequency_hz: float, *, orientations: OrientationInput = 'single', broadening_tesla: float = 0.0001, points: int = 1024, span_tesla: float | None = None, fields_tesla: np.ndarray | list[float] | tuple[float, ...] | None = None, lineshape: str = 'gaussian', detection_mode: str = 'absorption') -> ESRFieldSweepResult` | Return a broadened fixed-frequency ESR spectrum on a field axis. |

## `spin_dynamics.esr.systems`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `as_g_tensor(g_tensor: float | np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray` | Return a validated 3 by 3 electron ``g`` tensor. |
| class | `ESRSpinSystem` | Single-electron ESR spin system with an isotropic or anisotropic ``g`` tensor. |
| class | `ESRTransition` | One ESR transition between electron-spin energy eigenstates. |
| class | `ESREigensystem` | Energy levels, eigenvectors, and allowed ESR transitions for one field. |

## `spin_dynamics.exchange`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ExchangeSite` | One magnetically distinct site in a chemical-exchange bath. |
| class | `ExchangeSystem` | A set of exchanging sites with a first-order rate matrix. |
| class | `RelaxationExchange2DResult` | Encode-mix-detect data set for relaxation exchange spectroscopy. |
| function | `two_site_exchange(*, offset_a_hz: float, offset_b_hz: float, k_ab_hz: float, k_ba_hz: float | None = None, population_a: float | None = None, t2_a_seconds: float = np.inf, t2_b_seconds: float = np.inf, t1_a_seconds: float = np.inf, t1_b_seconds: float = np.inf, labels: tuple[str, str] = ('A', 'B'), balance: str = 'warn') -> ExchangeSystem` | Build a two-site exchange system. |
| function | `exchange_generator(system: ExchangeSystem) -> np.ndarray` | Return the kinetic generator ``X`` for magnetization exchange. |
| function | `transverse_generator(system: ExchangeSystem) -> np.ndarray` | Return the complex transverse Bloch-McConnell generator. |
| function | `transverse_propagator(system: ExchangeSystem, duration: float) -> np.ndarray` | Return ``exp(A * duration)`` for the transverse generator. |
| function | `mixing_propagator(system: ExchangeSystem, mixing_time: float, *, include_t1: bool = True) -> np.ndarray` | Return the longitudinal exchange map ``G`` for the mixing interval. |
| function | `simulate_exchange_fid(system: ExchangeSystem, times_seconds: np.ndarray, *, initial_magnetization: np.ndarray | None = None) -> np.ndarray` | Return the complex transverse free-induction decay with exchange. |
| function | `exchange_spectrum(system: ExchangeSystem, *, num_points: int = 4096, dwell_seconds: float | None = None, span_hz: float | None = None, line_broadening_hz: float = 0.0) -> tuple[np.ndarray, np.ndarray]` | Return ``(frequencies_hz, spectrum)`` from an exchange-broadened FID. |
| function | `simulate_relaxation_exchange_2d(system: ExchangeSystem, encode_times: np.ndarray, detect_times: np.ndarray, mixing_time: float, *, include_t1: bool = True) -> RelaxationExchange2DResult` | Simulate an encode-mix-detect (T2-T2) relaxation exchange data set. |

## `spin_dynamics.experiment.cli`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `build_parser() -> argparse.ArgumentParser` |  |
| function | `main(argv: Sequence[str] | None = None) -> int` |  |

## `spin_dynamics.experiment.config`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ConfigError` | Raised when a config document cannot be built into an Experiment. |
| function | `experiment_from_config(data: dict[str, Any]) -> Experiment` | Build an :class:`Experiment` from a friendly config mapping. |
| function | `experiment_to_config(experiment: Experiment) -> dict[str, Any]` | Return a friendly config mapping for an experiment (round-trips). |
| function | `dumps_toml(config: dict[str, Any]) -> str` | Serialize a friendly config mapping to a TOML string. |
| function | `dumps_json(config: dict[str, Any]) -> str` |  |
| function | `save_config(experiment: Experiment, path: str | Path) -> None` | Write an experiment to a ``.toml`` or ``.json`` config file. |
| function | `load_config(path: str | Path) -> Experiment` | Read an experiment from a ``.toml`` or ``.json`` config file. |

## `spin_dynamics.experiment.esr_adapter`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ESRDEERResult` | DEER time-domain form factor and its source distance distribution. |
| function | `require_system(experiment: Any) -> ESRSpinSystem` |  |
| function | `require_b0_vector(experiment: Any) -> tuple[float, float, float]` |  |
| function | `fid_kwargs(experiment: Any) -> dict[str, Any]` |  |
| function | `hahn_kwargs(experiment: Any) -> dict[str, Any]` |  |
| function | `cw_sweep_kwargs(experiment: Any) -> dict[str, Any]` |  |
| function | `deer_kwargs(experiment: Any) -> dict[str, Any]` |  |
| function | `run_deer(**kwargs) -> ESRDEERResult` | Adapt the array-returning DEER engine to a persistable facade result. |

## `spin_dynamics.experiment.estimate`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CostModel` | Abstract cost of one planned workflow execution. |
| class | `RuntimeEstimate` | Advisory runtime/memory prediction for a planned experiment. |
| function | `calibrate(force: bool = False) -> _Calibration` | Measure (or return the cached) affine cost constants for this host. |
| function | `set_calibration(overhead_seconds: float, seconds_per_unit: float, backend: str = 'manual') -> None` | Pin the cost constants (useful for tests and known hosts). |
| function | `estimate_runtime(cost: CostModel) -> RuntimeEstimate` | Convert an abstract cost model to a host-calibrated estimate. |
| function | `format_seconds(seconds: float) -> str` |  |
| function | `format_bytes(num_bytes: int) -> str` |  |

## `spin_dynamics.experiment.hardware`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SolenoidCoil` | Solenoid winding, generated by ``fields.coils.solenoid``. |
| class | `PlanarSpiralCoil` | Planar spiral (surface) coil, generated by ``fields.coils.planar_spiral``. |
| class | `TxCoil` | Transmit coil: geometry plus drive current for the Biot-Savart solve. |
| class | `RxCoil` | Receive coil; sensitivity is solved by reciprocity with unit current. |
| class | `UniformB0` | Uniform static field. |
| class | `ImagingPlane` | Physical placement of the 2-D phantom grid in the coil frame. |

## `spin_dynamics.experiment.io`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `register_result_type(cls: type) -> type` | Register a workflow result dataclass for load-time reconstruction. |
| function | `save_run(record: RunRecord, path: str) -> None` |  |
| class | `ReproductionReport` | Comparison between a saved run and a fresh execution of its spec. |
| class | `LoadedRun` | A run loaded from disk: spec, raw arrays, and best-effort result. |
| function | `load_run(path: str) -> LoadedRun` |  |

## `spin_dynamics.experiment.nqr_adapter`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `require_site(experiment: Any) -> QuadrupolarSite` |  |
| function | `target_transition(site: QuadrupolarSite, label: str) -> NQRTransition` | Resolve a transition label. |
| function | `bare_nutation_hz(site: QuadrupolarSite, label: str, effective_nutation_hz: float) -> float` | Convert the effective two-level Rabi rate to bare gamma*B1/(2*pi). |
| function | `model_selection(experiment: Any) -> NQRModelSelection` | Run (cached) reduced-vs-full model selection for an NQR experiment. |
| function | `resolved_slse_model(experiment: Any) -> str` |  |
| function | `resolve_slse_func(experiment: Any) -> Callable[..., Any]` |  |
| function | `slse_kwargs(experiment: Any) -> dict[str, Any]` |  |
| function | `sorc_kwargs(experiment: Any) -> dict[str, Any]` |  |
| function | `fid_kwargs(experiment: Any) -> dict[str, Any]` |  |
| function | `population_transfer_kwargs(experiment: Any) -> dict[str, Any]` |  |

## `spin_dynamics.experiment.plan`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ExperimentPlan` | Resolved execution plan for an :class:`Experiment`. |
| function | `plan_experiment(experiment: Experiment, *, estimate: bool = True) -> ExperimentPlan` | Resolve, validate, and (optionally) cost a declarative experiment. |

## `spin_dynamics.experiment.registry`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `WorkflowEntry` | One registered workflow behind the facade. |
| function | `register_workflow(entry: WorkflowEntry) -> WorkflowEntry` |  |
| function | `resolve(sequence_type: type, probe: str) -> WorkflowEntry | None` |  |
| function | `available_workflows() -> tuple[WorkflowEntry, ...]` |  |
| function | `probes_for(sequence_type: type) -> tuple[str, ...]` |  |

## `spin_dynamics.experiment.rules`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `RuleFinding` | One outcome from a single rule. |
| function | `rephasing_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]` | Front-load the isochromat-grid rephasing check. |
| function | `noise_spec_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]` | Validate the acquisition noise spec at plan time. |
| function | `hardware_wiring_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]` | Solve requested coil fields at plan time and surface wiring problems. |
| function | `nqr_model_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]` | Run ``select_nqr_model`` at plan time and check the engine dispatch. |
| function | `spectroscopy_inputs_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]` | Resolve spectroscopy sample objects and transition labels at plan time. |
| function | `transport_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]` | Report uniform-flow scale and flag closed reflecting transport. |
| function | `sequence_ir_rule(experiment: Experiment, entry: WorkflowEntry) -> list[RuleFinding]` | Compile general IR at plan time and reject unsupported backend policy. |
| function | `run_rules(experiment: Experiment, entry: WorkflowEntry, rules: Iterable[Rule] = DEFAULT_RULES) -> list[RuleFinding]` |  |

## `spin_dynamics.experiment.runner`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ExperimentPlanError` | Raised when ``run()`` is called on an experiment whose plan has errors. |
| class | `RunRecord` | A completed experiment run: spec, native result, and provenance. |
| function | `run_experiment(experiment: Experiment, **execution) -> RunRecord` | Plan and execute an experiment, delegating to the registered workflow. |

## `spin_dynamics.experiment.serialization`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SerializationError` | Raised when a value cannot be encoded to or decoded from JSON form. |
| function | `register_serializable(cls: type) -> type` | Register a dataclass so tagged dicts can be decoded back into it. |
| function | `registered_types() -> Mapping[str, type]` |  |
| function | `encode(value: Any, *, path: str = '$') -> Any` | Encode a value into JSON-representable form. |
| function | `decode(value: Any) -> Any` | Decode a value produced by :func:`encode`. |

## `spin_dynamics.experiment.specs`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `Phantom` | Spatial sample description for imaging: density plus optional maps. |
| class | `TransportDomain2D` | Density and physical axes for 2-D random-walker transport. |
| class | `UniformFlow2D` | Uniform ``(vx, vz)`` transport velocity in meters per second. |
| class | `DEERDistribution` | Distance grid and non-negative weights for a DEER experiment. |
| class | `SequenceDomain` | Spatial sample and field maps for general SequenceIR execution. |
| class | `SampledB0` | A spatially-varying static field sampled on the imaging plane. |
| class | `Sample` | Sample description. |
| class | `Hardware` | Transmit/receive hardware description. |
| class | `Acquisition` | Offset grid, receiver noise, and rephasing-guard configuration. |
| class | `CPMG` | Asymptotic (infinite-train) CPMG echo, no relaxation. |
| class | `CPMGTrain` | Finite CPMG echo train with relaxation. |
| class | `CPMGIRTrain` | Finite CPMG echo train preceded by an inversion-recovery delay sweep. |
| class | `CPMGImaging` | Phase-encoded CPMG imaging (2-D spin-warp on a phantom). |
| class | `PGSE` | Deterministic pulsed-gradient spin echo diffusion encoding. |
| class | `PGSEWalkers` | Explicit random-walker PGSE with diffusion and optional uniform flow. |
| class | `NQRSLSE` | Spin-lock spin-echo NQR detection train. |
| class | `NQRSORC` | Strong off-resonance comb NQR train (reduced spin-1 engine only). |
| class | `NQRFID` | Single-pulse NQR FID using the full density-matrix engine. |
| class | `NQRPopulationTransfer` | Selective perturbation followed by reduced spin-1 SLSE detection. |
| class | `ESRFID` | Pulsed ESR free-induction decay (rotating frame, single isochromat). |
| class | `ESRHahnEcho` | Two-pulse ESR Hahn echo (single isochromat). |
| class | `ESRCWSweep` | Continuous-wave ESR field sweep at fixed microwave frequency. |
| class | `ESRDEER` | DEER form factor calculated from ``Sample.deer_distribution``. |
| class | `ESRTwoPulseESEEM` | Two-pulse ESEEM trace and frequency spectrum. |
| class | `ESRThreePulseESEEM` | Three-pulse stimulated-echo ESEEM trace and spectrum. |
| class | `ESRHYSCORE` | Two-dimensional HYSCORE time grid and spectrum. |
| class | `ESRDaviesENDOR` | One-dimensional Davies ENDOR radiofrequency sweep. |
| class | `ESRMimsENDOR` | One-dimensional Mims ENDOR sweep with blind-spot weighting. |
| class | `SequenceIRExecution` | Execute a backend-neutral :class:`SequenceIR` through the facade. |
| class | `Experiment` | A complete declarative experiment description. |
| function | `non_default_fields(experiment: Experiment) -> dict[str, Any]` | Return dotted spec-field names whose values differ from the defaults. |

## `spin_dynamics.experiment.wiring`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `uses_hardware_fields(hardware: Hardware) -> bool` | True when the hardware spec requests a coil-field solve. |
| function | `grid_positions_m(shape: tuple[int, int], plane: ImagingPlane) -> np.ndarray` | Physical voxel positions (meters) of a phantom grid on the plane. |
| function | `sampled_b0_from_solution(solution, plane: ImagingPlane, shape: tuple[int, int], carrier_hz: float, nutation_rad_s: float = 1.0) -> SampledB0` | Sample a solved 3-D field onto an imaging plane as a :class:`SampledB0`. |
| function | `solve_imaging_field_maps(phantom: Phantom, hardware: Hardware, *, t1_seconds: float | None = None, t2_seconds: float | None = None) -> ImagingFieldMaps` | Assemble (and cache) the imaging field maps for a phantom + hardware. |
| function | `solve_diagnostics(phantom: Phantom, hardware: Hardware) -> dict[str, float]` | Return the per-coil transmit-efficiency diagnostics (cached solve). |
| function | `solve_for_experiment(experiment: Experiment) -> ImagingFieldMaps` | Solve field maps for an experiment's sample + hardware specs. |

## `spin_dynamics.fields.coil_peec`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `self_partial_inductance(length: float, gmd_radius: float) -> float` | Self partial inductance (H) of a straight round filament. |
| class | `Conductor` | A single current-carrying wire: a polyline centreline plus a cross-section. |
| function | `conductor_from_segments(segments: Sequence[Segment], *, wire_radius: float, material: ConductorMaterial = ANNEALED_COPPER, n_radial: int = 6, n_angular: int = 8, temperature: float | None = None) -> Conductor` | Build a :class:`Conductor` from a connected ``(start, end)`` segment list. |
| function | `filament_self_inductance(filament: list[Segment], gmd_radius: float) -> float` | Self partial inductance (H) of one open filamentary wire following the path. |
| class | `PEECImpedance` | Frequency-swept terminal impedance of a coil from the PEEC solve. |
| function | `extract_impedance(conductor: Conductor, frequencies: Sequence[float], *, formulation: str = 'chain', ground_plane: GroundPlane | GroundedBox | None = None) -> PEECImpedance` | Solve the PEEC system for ``L(w)`` and ``R(w)``. |
| function | `extract_impedance_surface(conductor: Conductor, frequencies: Sequence[float], *, n_perimeter: int = 48, formulation: str = 'chain', shield: GroundPlane | GroundedBox | None = None) -> PEECImpedance` | Surface-impedance (SIBC) solve for the deep-skin (high ``a/delta``) regime. |
| function | `current_distribution(conductor: Conductor, frequency: float, *, formulation: str = 'chain', segment: int | None = None) -> tuple[np.ndarray, np.ndarray]` | Per-sub-filament current magnitude across the cross-section at ``frequency``. |
| class | `GroundedBox` | A grounded rectangular shield enclosing the coil (walls at potential zero). |
| class | `GroundPlane` | A single infinite grounded conducting plane at zero potential (a half-space wall). |
| function | `self_capacitance(conductor: Conductor, *, shield: GroundedBox | GroundPlane | None = None, relative_permittivity: float = 1.0, potential = None) -> float` | Lumped self-capacitance (F) of the coil from an electrostatic energy method. |
| function | `capacitance_to_ground(conductor: Conductor, *, shield: GroundedBox | GroundPlane | None = None, relative_permittivity: float = 1.0) -> float` | Isolated self-capacitance to ground (F): the charge held at unit potential,. |
| function | `helical_solenoid(*, diameter: float, length: float, turns: int, wire_radius: float, material: ConductorMaterial = ANNEALED_COPPER, n_per_turn: int = 16, n_radial: int = 6, n_angular: int = 8, temperature: float | None = None, axis: str = 'z') -> Conductor` | Build a :class:`Conductor` for a helical single-layer solenoid. |
| function | `self_resonant_frequency(conductor: Conductor) -> float` | First self-resonant frequency (Hz) ``1 / (2 pi sqrt(L C))``. |
| function | `radiation_resistance(conductor: Conductor, frequency: float, *, shield: GroundedBox | GroundPlane | None = None) -> float` | First-order (magnetic-dipole) radiation resistance (ohm) of the coil. |
| class | `PEECCoilProperties` | Lumped RF properties of an arbitrary coil from the PEEC solve. |
| function | `coil_properties_peec(conductor: Conductor, frequency: float, *, formulation: str = 'full', include_radiation: bool = True, shield: GroundedBox | GroundPlane | None = None, relative_permittivity: float = 1.0) -> PEECCoilProperties` | Extract lumped RF properties of an arbitrary coil at ``frequency`` via PEEC. |

## `spin_dynamics.fields.coil_properties`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ConductorMaterial` | A conductor's RF-relevant material constants, with a temperature model. |
| function | `medhurst_proximity_factor(l_over_D: float, p_over_d: float) -> float` | Medhurst proximity factor ``Phi`` from the ``(l/D, p/d)`` table. |
| function | `sheath_helix_dispersion(frequency: float, a: float, psi: float) -> tuple[float, float, float]` | Solve the ``n = 0`` sheath-helix dispersion at ``frequency``. |
| class | `CoilProperties` | Lumped RF properties of a single-layer solenoid at a design frequency. |
| function | `solenoid_properties(*, diameter: float, length: float, turns: int, wire_diameter: float, frequency: float, material: ConductorMaterial = ANNEALED_COPPER, temperature: float | None = None) -> CoilProperties` | Extract the lumped RF properties of a single-layer round-wire solenoid. |
| function | `solenoid_field_inductance(*, diameter: float, length: float, turns: int, wire_diameter: float, n_segments: int = 120) -> float` | Independent field-based inductance of the solenoid (Biot-Savart / Neumann). |

## `spin_dynamics.fields.fasthenry_interop`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `to_fasthenry_inp(conductor: Conductor, frequencies, *, nwinc: int | None = None, nhinc: int | None = None, rw: float | None = None, rh: float | None = None) -> str` | Return a FastHenry ``.inp`` deck for ``conductor`` over ``frequencies`` (Hz). |
| class | `FastHenryResult` | FastHenry ``L(f)`` and ``R(f)`` for the (single-port) conductor. |
| function | `run_fasthenry(conductor: Conductor, frequencies, *, nwinc: int | None = None, nhinc: int | None = None, rw: float | None = None, rh: float | None = None, timeout: float = 300.0, workdir: str | None = None) -> FastHenryResult` | Run FastHenry on ``conductor`` and return its ``L(f)``/``R(f)`` (single port). |
| function | `compare_with_fasthenry(conductor: Conductor, frequencies, *, nwinc: int | None = None, nhinc: int | None = None, rw: float | None = None, rh: float | None = None, formulation: str = 'chain')` | Return ``(peec, fasthenry)`` results for the same conductor and frequencies. |

## `spin_dynamics.fields.fastercap_interop`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `to_fastercap_panels(conductor: Conductor, *, n_theta: int = 24, name: str = 'cond') -> str` | Return a FasterCap panel deck for the conductor's (round) wire surface. |
| function | `run_fastercap(conductor: Conductor, *, n_theta: int = 24, tolerance: float = 0.01, timeout: float = 180.0, workdir: str | None = None) -> float` | Panelize ``conductor``, run FasterCap and return its self-capacitance (F). |
| function | `compare_capacitance_with_fastercap(conductor: Conductor, *, n_theta: int = 24, tolerance: float = 0.01) -> tuple[float, float]` | Return ``(peec, fastercap)`` self-capacitance (F) for the same conductor. |

## `spin_dynamics.fields.coils`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `solenoid(*, radius: float, length: float, turns: int, center: Sequence[float] = (0.0, 0.0, 0.0), axis: str = 'z', n_segments: int = 48) -> list[Segment]` | A solenoid modelled as ``turns`` coaxial loops evenly spaced over ``length``. |
| function | `planar_spiral(*, r_inner: float, r_outer: float, turns: int, center: Sequence[float] = (0.0, 0.0, 0.0), axis: str = 'z', n_points: int = 400) -> list[Segment]` | A planar Archimedean spiral from ``r_inner`` to ``r_outer`` over ``turns``. |
| function | `maxwell_pair(*, radius: float, separation: float | None = None, center: Sequence[float] = (0.0, 0.0, 0.0), axis: str = 'z', n_segments: int = 72) -> list[Segment]` | A Maxwell pair (anti-Helmholtz): two coaxial loops with opposed currents. |
| function | `conducting_ring(*, radius: float, center: Sequence[float] = (0.0, 0.0, 0.0), axis: str = 'z', n_segments: int = 144) -> list[Segment]` | A single circular loop -- e.g. a conducting ring for the eddy model. |
| function | `cylindrical_shield(*, radius: float, length: float, n_rings: int, center: Sequence[float] = (0.0, 0.0, 0.0), axis: str = 'z') -> tuple[np.ndarray, np.ndarray]` | Coaxial-ring model of a thin conducting cylinder (e.g. a cryostat bore). |

## `spin_dynamics.fields.domain`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SpatialDomain` | A rectilinear voxel grid of one to three spatial axes. |

## `spin_dynamics.fields.eddy_modes`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `EddyModeSpectrum` | Eddy-mode decomposition of a gradient's residual field at the sample. |
| class | `EddyModes` | Coaxial-ring eddy model of a conducting structure. |

## `spin_dynamics.fields.gradient_coils`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CylindricalWindingSurface` | Regular z-axis cylindrical winding surface. |
| class | `CylindricalGradientSystem` | A cylindrical source mesh and its field sensitivity matrix. |
| class | `GradientCoilDesignResult` | Constrained current solution, stream function, and fit diagnostics. |
| function | `spherical_target_points(radius: float, *, points_per_axis: int = 9, center: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray` | Return a Cartesian point grid clipped to a spherical target volume. |
| function | `linear_gradient_target(points: np.ndarray, gradient: Sequence[float], *, center: Sequence[float] = (0.0, 0.0, 0.0), offset_t: float = 0.0) -> np.ndarray` | Return ``offset + gradient dot (position - center)`` in tesla. |
| function | `build_cylindrical_gradient_system(surface: CylindricalWindingSurface, target_points: np.ndarray, *, field_direction: Sequence[float] = (0.0, 0.0, 1.0), chunk_size: int = 128) -> CylindricalGradientSystem` | Build the projected field-per-ampere matrix for a cylindrical mesh. |
| function | `solve_gradient_coil(system: CylindricalGradientSystem, target_field_t: np.ndarray, *, regularization: float = 0.0, field_weights: np.ndarray | None = None, solver: SolverName = 'auto', atol: float = 1e-10, btol: float = 1e-10, max_iterations: int | None = None) -> GradientCoilDesignResult` | Solve the KCL-constrained, Tikhonov-regularized coil design. |
| function | `design_cylindrical_gradient_coil(surface: CylindricalWindingSurface, target_points: np.ndarray, target_field_t: np.ndarray, *, field_direction: Sequence[float] = (0.0, 0.0, 1.0), regularization: float = 0.0, field_weights: np.ndarray | None = None, chunk_size: int = 128, solver: SolverName = 'auto', atol: float = 1e-10, btol: float = 1e-10, max_iterations: int | None = None) -> GradientCoilDesignResult` | Build and solve a cylindrical gradient-coil problem in one call. |

## `spin_dynamics.fields.interpolate`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `dlinear_sample(values: np.ndarray, axes: Sequence[np.ndarray], positions: np.ndarray) -> np.ndarray` | Multilinearly sample a ``d``-dimensional ``values`` map at ``positions``. |

## `spin_dynamics.fields.magnetostatics`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `BarMagnet` | A 2-D uniformly magnetized rectangular bar (infinite along z). |
| class | `FiniteMagnetRod` | A finite uniformly magnetized rod with its long axis along z. |
| class | `HalbachDipoleFieldMaps` | Sampled 3-D B0 field of a four-rod finite Halbach dipole. |
| function | `bar_array_b0(x: np.ndarray, y: np.ndarray, bars: Sequence[BarMagnet], *, yoke_y: float | None = None) -> tuple[np.ndarray, np.ndarray]` | Return ``(Bx, By)`` (T) of permanent-magnet bars at points ``(x, y)``. |
| function | `finite_magnet_array_b0(points: np.ndarray, rods: Sequence[FiniteMagnetRod], *, n_cross: int = 5, n_length: int = 21, chunk_size: int = 4096) -> np.ndarray` | Return the 3-D B field (T) of finite uniformly magnetized rods. |
| function | `halbach_dipole_magnets(*, center_radius: float = 0.03, length: float = 0.08, remanence: float = 1.3, rod_shape: Literal['cylinder', 'square'] = 'cylinder', rod_radius: float | None = 0.008, rod_width: float | None = None, field_angle: float = 0.0) -> tuple[FiniteMagnetRod, ...]` | Return four rods for the lowest-order finite Halbach dipole. |
| function | `sample_halbach_dipole_field(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray, *, rods: Sequence[FiniteMagnetRod] | None = None, center_radius: float = 0.03, length: float = 0.08, remanence: float = 1.3, rod_shape: Literal['cylinder', 'square'] = 'cylinder', rod_radius: float | None = 0.008, rod_width: float | None = None, field_angle: float = 0.0, n_cross: int = 5, n_length: int = 21, chunk_size: int = 4096, gamma: float = GAMMA_PROTON) -> HalbachDipoleFieldMaps` | Sample a finite four-rod Halbach dipole onto a 3-D grid. |
| function | `biot_savart(points: np.ndarray, segments: Sequence[tuple[Sequence[float], Sequence[float]]], current: float) -> np.ndarray` | Biot-Savart B field (T) of straight current segments at ``points``. |
| function | `segment_field_sensitivity(points: np.ndarray, segments: Sequence[tuple[Sequence[float], Sequence[float]]], *, direction: Sequence[float] = (0.0, 0.0, 1.0), chunk_size: int = 128) -> np.ndarray` | Return projected field per ampere for every straight source segment. |
| function | `circular_loop(center: Sequence[float], radius: float, *, axis: str = 'y', n_segments: int = 72) -> list[tuple[np.ndarray, np.ndarray]]` | Return straight-segment endpoints approximating a circular current loop. |
| class | `MagnetFieldMaps` | Sampled B0/B1 of a magnet+coil assembly on a 2-D ``(x, y)`` grid. |
| function | `sample_magnet_field(x_axis: np.ndarray, y_axis: np.ndarray, bars: Sequence[BarMagnet], *, yoke_y: float | None = None, coil_segments: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None, coil_current: float = 1.0, gamma: float = GAMMA_PROTON) -> MagnetFieldMaps` | Sample a permanent-magnet + RF-coil assembly onto a 2-D grid. |
| function | `nmr_mouse_magnets(*, magnet_width: float = 0.02, magnet_height: float = 0.02, gap: float = 0.012, remanence: float = 1.3, antiparallel: bool = True) -> tuple[list[BarMagnet], float]` | Return the bar magnets and yoke plane of a u-shaped NMR-MOUSE. |

## `spin_dynamics.fields.maps`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SpatialFieldMaps` | Spatial sample and field maps shared by imaging and diffusion workflows. |

## `spin_dynamics.fields.nonlinear_magnetostatics`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `BrauerBH` | Smooth Brauer soft-magnetic reluctivity ``nu(B^2) = bk1 + bk2 exp(bk3 B^2)``. |
| class | `MagneticMaterial` | A magnetic material region: linear or saturable, optionally a magnet. |
| function | `air() -> MagneticMaterial` | Return free space (``mu_r = 1``). |
| function | `linear_material(mu_r: float, *, name: str = 'linear') -> MagneticMaterial` | Return a linear material of relative permeability ``mu_r``. |
| function | `rf_ferrite(mu_r: float = 1000.0, *, bh: BrauerBH | None = None) -> MagneticMaterial` | Return a high-permeability RF ferrite (linear by default, below saturation). |
| function | `soft_iron(curve: str = 'soft_iron') -> MagneticMaterial` | Return a saturable soft-iron material from a named Brauer curve. |
| function | `ndfeb(remanence_t: float = 1.3, *, mu_rec: float = 1.05) -> MagneticMaterial` | Return an NdFeB permanent magnet (recoil permeability ``mu_rec``). |
| class | `PlanarSolution` | Result of a planar magnetostatic solve. |
| class | `PlanarMagnetostatics` | 2D translationally-invariant nonlinear magnetostatics via ``A_z``. |
| class | `AxisymmetricSolution` | Result of an axisymmetric magnetostatic solve. |
| class | `AxisymmetricMagnetostatics` | Rotationally-symmetric nonlinear magnetostatics via ``A_phi``. |

## `spin_dynamics.fields.positions`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `positions_nd(values: np.ndarray, ndim: int | None = None) -> np.ndarray` | Validate and return an ``(num_particles, d)`` float64 position array. |
| function | `velocity_array(velocity, positions: np.ndarray, time: float) -> np.ndarray` | Return a per-particle velocity array matching ``positions``. |
| function | `gradient_offset(positions: np.ndarray, gradient) -> np.ndarray` | Return the Lagrangian gradient-induced offset ``positions @ gradient``. |

## `spin_dynamics.fields.quasistatic`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `vector_potential(points: np.ndarray, segments: Sequence[Segment], current: float = 1.0) -> np.ndarray` | Magnetic vector potential ``A`` (T*m) of straight current segments. |
| function | `induced_efield(points: np.ndarray, segments: Sequence[Segment], dcurrent_dt: float) -> np.ndarray` | Induced (inductive) E-field ``E = -dA/dt`` (V/m) for current slew ``dI/dt``. |
| function | `mutual_inductance(loop_a: Sequence[Segment], loop_b: Sequence[Segment]) -> float` | Mutual inductance ``M`` (H) between two filamentary loops. |
| function | `self_inductance_circular(radius: float, wire_radius: float) -> float` | Self-inductance (H) of a circular loop, ``mu0 a (ln(8a/r_w) - 2)``. |
| function | `eddy_power(e_field: np.ndarray, conductivity: np.ndarray | float, cell_volume: float, *, time_average: bool = True) -> float` | Resistive eddy power ``P = k * integral sigma |E|^2 dV`` (W). |
| class | `EddyResult` | Induced E-field, eddy current and deposited power in a conductive sample. |
| function | `eddy_currents(grid_points: np.ndarray, segments: Sequence[Segment], dcurrent_dt: float, *, conductivity: float, mask: np.ndarray, spacing: Sequence[float], charge_correction: bool = True, time_average: bool = True) -> EddyResult` | First-order eddy currents and power for a conductive sample on a grid. |
| function | `geometric_loss_integral(grid_points: np.ndarray, segments: Sequence[Segment], *, conductivity: float, mask: np.ndarray, spacing: Sequence[float], charge_correction: bool = False) -> float` | ``G = integral sigma |A_unit|^2 dV`` (ohm/(rad/s)^2) over the conductor. |
| function | `reflected_resistance(grid_points: np.ndarray, segments: Sequence[Segment], *, conductivity: float, mask: np.ndarray, spacing: Sequence[float], frequency: float, charge_correction: bool = False) -> float` | Reflected series resistance ``R = omega^2 integral sigma |A_unit|^2 dV`` (ohm). |
| function | `coil_inductance(radii: Sequence[float], centers: np.ndarray, *, wire_radius: float, axis: str = 'z', n_segments: int = 120) -> float` | Series inductance (H) of coaxial circular turns carrying the same current. |
| class | `CoilLoading` | Frequency-swept sample loading of a coil by a conductive medium. |
| function | `coil_loading(grid_points: np.ndarray, segments: Sequence[Segment], *, conductivity: float, mask: np.ndarray, spacing: Sequence[float], frequencies: Sequence[float], inductance: float, coil_resistance: float | np.ndarray, charge_correction: bool = False) -> CoilLoading` | Sweep the sample-loading effect of a conductive medium across frequency. |

## `spin_dynamics.fields.scalar_potential_3d`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ScalarPotentialSolution` | Result of a 3D reduced-scalar-potential magnetostatic solve. |
| class | `ReducedScalarPotential3D` | 3D nonlinear magnetostatics via the reduced scalar potential ``psi``. |

## `spin_dynamics.hyperpolarization.singlet`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `singlet_projector(nspin: int = 2, pair: tuple[int, int] = (0, 1)) -> np.ndarray` | Return the selected pair's singlet projector embedded in ``nspin`` spins. |
| function | `triplet_projector(nspin: int = 2, pair: tuple[int, int] = (0, 1)) -> np.ndarray` | Return the selected pair's total-triplet projector. |
| function | `singlet_order_operator(nspin: int = 2, pair: tuple[int, int] = (0, 1)) -> np.ndarray` | Return trace-zero pair singlet order ``P_S - P_T / 3``. |
| function | `spin_pair_swap_operator(nspin: int = 2, pair: tuple[int, int] = (0, 1)) -> np.ndarray` | Return the unitary operator that swaps the selected spin-1/2 sites. |
| function | `singlet_population(density: np.ndarray, *, pair: tuple[int, int] = (0, 1)) -> float` | Return the selected pair's singlet population from a physical density. |
| function | `triplet_population(density: np.ndarray, *, pair: tuple[int, int] = (0, 1)) -> float` | Return the selected pair's total triplet population. |
| function | `singlet_order_amplitude(density: np.ndarray, *, pair: tuple[int, int] = (0, 1)) -> float` | Project a physical or deviation density onto normalized singlet order. |
| class | `ParahydrogenState` | Physical and deviation density matrices for a para/ortho H2 mixture. |
| function | `parahydrogen_state(para_fraction: float) -> ParahydrogenState` | Return the two-proton state for a specified parahydrogen fraction. |

## `spin_dynamics.hyperpolarization.lls`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `store_singlet_order(density: np.ndarray, duration_seconds: float, singlet_lifetime_seconds: float, *, pair: tuple[int, int] = (0, 1), purge_non_singlet: bool = True) -> np.ndarray` | Store a selected singlet-order mode for a finite duration. |
| class | `SLICLongLivedStateResult` | Stage-resolved result of a two-spin SLIC/store/SLIC workflow. |
| function | `simulate_slic_lls(system: CoupledSpinSystem, storage_times_seconds: Iterable[float] | np.ndarray, *, singlet_lifetime_seconds: float, preparation_time_seconds: float | None = None, readout_time_seconds: float | None = None, nutation_hz: float | None = None, purge_non_singlet: bool = True) -> SLICLongLivedStateResult` | Simulate two-spin SLIC preparation, storage, and reconversion. |

## `spin_dynamics.hyperpolarization.phip`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `PHIPFieldSegment` | One constant-field segment used during PHIP product transport. |
| class | `HydrogenativePHIPState` | Mapped product state following pairwise parahydrogen addition. |
| class | `PHIPAcquisitionResult` | State and complex FID from a hydrogenative PHIP protocol. |
| function | `hydrogenative_phip_state(product_nspin: int, *, para_fraction: float, pairwise_addition_fraction: float = 1.0, product_pair: tuple[int, int] = (0, 1)) -> HydrogenativePHIPState` | Map parahydrogen singlet order into specified product spin sites. |
| function | `evolve_phip_field_trajectory(density: np.ndarray, system: CoupledSpinSystem, segments: Sequence[PHIPFieldSegment]) -> np.ndarray` | Evolve a PHIP product through a piecewise-constant field trajectory. |
| function | `secularize_high_field_product(density: np.ndarray) -> np.ndarray` | Remove nonsecular product-basis coherences after high-field formation. |
| function | `apply_hard_pulse(density: np.ndarray, system: CoupledSpinSystem, flip_angle_radians: float, *, phase_radians: float = 0.0, indices: Iterable[int] | None = None) -> np.ndarray` | Apply an ideal hard pulse to selected product spins. |
| function | `acquire_phip_fid(density: np.ndarray, system: CoupledSpinSystem, times_seconds: Iterable[float] | np.ndarray, *, detect_indices: Iterable[int] | None = None, t2_seconds: float | None = None) -> np.ndarray` | Acquire a high-field weak-coupling PHIP FID from selected spins. |
| function | `simulate_hydrogenative_phip(system: CoupledSpinSystem, times_seconds: Iterable[float] | np.ndarray, *, protocol: PHIPProtocol, para_fraction: float, pairwise_addition_fraction: float = 1.0, product_pair: tuple[int, int] = (0, 1), reaction_time_seconds: float = 0.0, field_trajectory: Sequence[PHIPFieldSegment] | None = None, pulse_flip_angle_radians: float = np.pi / 4.0, pulse_phase_radians: float = 0.0, pulse_indices: Iterable[int] | None = None, detect_indices: Iterable[int] | None = None, t2_seconds: float | None = None) -> PHIPAcquisitionResult` | Simulate hydrogenative PASADENA or trajectory-defined ALTADENA. |

## `spin_dynamics.interference.active`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CompensationActuator` | Analog feedforward path from a commanded field to the field at the primary. |
| class | `ActiveCancellationResult` | Outcome of an analog feedforward cancellation before and after the ADC. |
| function | `feedforward_cancel(primary: np.ndarray, references: np.ndarray, model, actuator: CompensationActuator, sample_rate_hz: float, *, adc_saturation: float | None = None, rng: np.random.Generator | None = None, seed: int | None = None) -> ActiveCancellationResult` | Cancel RFI at the primary with a compensation coil before digitisation. |

## `spin_dynamics.interference.cancellers`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CancellationResult` | Cleaned primary data and the RFI estimate produced by a canceller. |
| class | `LinearCancellerModel` | Fixed multi-reference FIR canceller fitted by gated ridge least squares. |
| function | `fit_gated_ridge_fir(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, *, taps: int = 1, ridge: float = 0.0) -> LinearCancellerModel` | Fit a fixed FIR reference canceller on baseline samples only. |
| function | `gated_ridge_fir_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, *, taps: int = 1, ridge: float = 0.0) -> CancellationResult` | Fit and apply a gated ridge-LS FIR canceller. |
| function | `fit_scalar_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, *, ridge: float = 0.0) -> LinearCancellerModel` | Fit a zero-lag multi-reference scalar canceller on ``fit_mask`` samples. |
| function | `scalar_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, *, ridge: float = 0.0) -> CancellationResult` | Fit and apply a zero-lag multi-reference scalar canceller. |
| function | `fit_robust_fir(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, *, taps: int = 1, ridge: float = 0.0, huber_delta: float = 1.345, max_iter: int = 25, tol: float = 1e-06, scale: float | None = None) -> LinearCancellerModel` | Fit an outlier-robust FIR canceller by Huber IRLS on baseline samples. |
| function | `robust_fir_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, *, taps: int = 1, ridge: float = 0.0, huber_delta: float = 1.345, max_iter: int = 25, tol: float = 1e-06, scale: float | None = None) -> CancellationResult` | Fit and apply a Huber-IRLS robust FIR canceller. |
| function | `windowed_ridge_fir_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, *, taps: int = 1, window_samples: int = 1024, ridge: float = 0.0, smoothness: float = 0.0) -> CancellationResult` | Apply an offline windowed ridge-FIR canceller with smooth coefficients. |
| function | `joint_signal_reference_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, signal_basis: np.ndarray, *, taps: int = 1, reference_ridge: float = 0.0, signal_ridge: float = 0.0) -> CancellationResult` | Fit reference-derived RFI and structured signal terms jointly. |
| function | `windowed_joint_signal_reference_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, signal_basis: np.ndarray, *, taps: int = 1, window_samples: int = 1024, reference_ridge: float = 0.0, smoothness: float = 0.0, signal_ridge: float = 0.0) -> CancellationResult` | Fit windowed reference RFI and structured signal terms jointly. |
| function | `sparse_reference_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray, *, sparse_penalty: float, taps: int = 1, ridge: float = 0.0, max_iter: int = 50, tol: float = 1e-06) -> CancellationResult` | Split the primary into reference-explained RFI plus a sparse impulse term. |
| function | `adaptive_lms_canceller(primary: np.ndarray, references: np.ndarray, update_mask: np.ndarray | None = None, *, taps: int = 1, step: float = 0.1, normalized: bool = True, epsilon: float = 1e-12, leak: float = 0.0, initial_coefficients: np.ndarray | None = None) -> CancellationResult` | Apply mask-aware LMS/NLMS cancellation for time-varying transfer paths. |
| function | `adaptive_rls_canceller(primary: np.ndarray, references: np.ndarray, update_mask: np.ndarray | None = None, *, taps: int = 1, forgetting: float = 0.995, initial_covariance: float = 1000.0, initial_coefficients: np.ndarray | None = None) -> CancellationResult` | Apply a mask-aware recursive least-squares reference canceller. |
| function | `frequency_domain_canceller(primary: np.ndarray, references: np.ndarray, fit_mask: np.ndarray | None = None, *, segment_length: int = 256, hop: int | None = None, ridge: float = 0.0, sample_rate_hz: float = 1.0) -> CancellationResult` | Cancel RFI with a per-frequency multi-reference Wiener transfer function. |

## `spin_dynamics.interference.coils`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ReferenceCoil` | A pickup coil that senses one projection of the RFI field. |
| function | `coil_voltage(coil: ReferenceCoil, sources: list[RFISource] | tuple[RFISource, ...], *, nqr_signal: np.ndarray | None = None, rng: np.random.Generator | None = None, seed: int | None = None) -> np.ndarray` | Return the measured voltage of one coil under the given RFI sources. |
| function | `reference_matrix(coils: list[ReferenceCoil] | tuple[ReferenceCoil, ...], sources: list[RFISource] | tuple[RFISource, ...], *, nqr_signal: np.ndarray | None = None, rng: np.random.Generator | None = None, seed: int | None = None) -> np.ndarray` | Return the stacked reference observation ``X`` with shape ``(K, N)``. |
| function | `coupling_matrix(coils: list[ReferenceCoil] | tuple[ReferenceCoil, ...]) -> np.ndarray` | Return ``C = [c_1, ..., c_K]`` (shape ``(3, K)``) of coil pickup vectors. |

## `spin_dynamics.interference.diagnostics`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `RFISuppressionResult` | RMS RFI level before and after cancellation on a selected mask. |
| class | `MatchedFilterImprovementResult` | Matched-filter SNR before and after cancellation. |
| class | `SignalBiasResult` | Complex gain, amplitude bias, phase bias, and residual error. |
| class | `ResidualLine` | One prominent residual spectral line. |
| class | `ResidualSpectrumResult` | FFT spectrum and the largest residual spectral lines. |
| class | `DesignMatrixDiagnostics` | Rank and conditioning diagnostics for a tapped reference matrix. |
| class | `SaturationDiagnostics` | Samples that exceed a symmetric front-end threshold. |
| class | `ReferenceNoiseInjectionResult` | White noise a canceller injects into the cleaned output via its references. |
| function | `rfi_suppression_db(before: np.ndarray, after: np.ndarray, mask: np.ndarray | None = None, *, clean_signal: np.ndarray | None = None) -> RFISuppressionResult` | Return RMS suppression in dB on ``mask``. |
| function | `matched_filter_snr_improvement(clean_signal: np.ndarray, before_signals: np.ndarray, after_signals: np.ndarray, *, pnoise: np.ndarray | None = None, frequencies: np.ndarray | None = None, offsets: np.ndarray | None = None, noise_scale: float = 1.0, matched_filter: np.ndarray | None = None) -> MatchedFilterImprovementResult` | Estimate matched-filter SNR improvement from cancellation. |
| function | `signal_bias(clean_signal: np.ndarray, cleaned_signal: np.ndarray, mask: np.ndarray | None = None) -> SignalBiasResult` | Estimate amplitude and phase bias of ``cleaned_signal`` vs ``clean_signal``. |
| function | `residual_spectral_lines(residual: np.ndarray, sample_rate_hz: float, mask: np.ndarray | None = None, *, top_n: int = 5) -> ResidualSpectrumResult` | Return FFT amplitudes and the largest residual spectral lines. |
| function | `reference_design_diagnostics(references: np.ndarray, mask: np.ndarray | None = None, *, taps: int = 1, rtol: float | None = None) -> DesignMatrixDiagnostics` | Return rank and condition number of the tapped reference design matrix. |
| function | `saturation_diagnostics(signal: np.ndarray, threshold: float) -> SaturationDiagnostics` | Flag samples whose magnitude reaches or exceeds ``threshold``. |
| function | `reference_noise_injection(coefficients: np.ndarray, reference_noise_sigma: np.ndarray | float) -> ReferenceNoiseInjectionResult` | Estimate the reference-channel noise a canceller injects into its output. |

## `spin_dynamics.interference.masks`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SampleLabel` | Per-sample acquisition state on the shot clock. |
| class | `AcquisitionMask` | Per-sample receive-state labels sharing one shot clock. |
| function | `blank_mask(sample_rate_hz: float, num_samples: int, *, fill: SampleLabel = SampleLabel.SIGNAL) -> AcquisitionMask` | Return a mask with every sample set to ``fill``. |
| function | `mask_from_intervals(sample_rate_hz: float, duration_seconds: float, *, transmit: Sequence[Interval] | None = None, ringdown: Sequence[Interval] | None = None, baseline: Sequence[Interval] | None = None, fill: SampleLabel = SampleLabel.SIGNAL) -> AcquisitionMask` | Build a mask from second-valued intervals on a shot clock. |

## `spin_dynamics.interference.recordings`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `RFIRecording` | A measured RFI record: ADC sample windows plus a reconstructed mask. |
| function | `save_rfi_recording(path: str | Path, recording: RFIRecording) -> None` | Write a recording to a NumPy ``.npz`` archive. |
| function | `load_rfi_recording(path: str | Path) -> RFIRecording` | Load a recording previously written by :func:`save_rfi_recording`. |

## `spin_dynamics.interference.sources`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `RFIWaveform` | A real, scalar interference waveform sampled on a shot clock. |
| function | `tone_waveform(num_samples: int, sample_rate_hz: float, *, frequency_hz: float, amplitude: float = 1.0, phase_rad: float = 0.0) -> RFIWaveform` | Return a single narrowband tone ``A cos(2*pi*f*t + phi)``. |
| function | `am_carrier_waveform(num_samples: int, sample_rate_hz: float, *, carrier_hz: float, modulation_hz: float, modulation_depth: float = 0.5, amplitude: float = 1.0, phase_rad: float = 0.0) -> RFIWaveform` | Return an amplitude-modulated carrier (broadcast-AM model). |
| function | `chirp_waveform(num_samples: int, sample_rate_hz: float, *, start_hz: float, stop_hz: float, amplitude: float = 1.0, phase_rad: float = 0.0) -> RFIWaveform` | Return a linear-frequency chirp sweeping ``start_hz`` -> ``stop_hz``. |
| function | `colored_noise_waveform(num_samples: int, sample_rate_hz: float, *, amplitude: float = 1.0, exponent: float = 0.0, seed: int | None = None, rng: np.random.Generator | None = None) -> RFIWaveform` | Return broadband noise with a power-law spectrum ``S(f) ~ f**(-exponent)``. |
| function | `impulsive_waveform(num_samples: int, sample_rate_hz: float, *, event_rate_hz: float, amplitude: float = 1.0, decay_seconds: float = 1e-05, ring_hz: float = 0.0, seed: int | None = None, rng: np.random.Generator | None = None) -> RFIWaveform` | Return impulsive bursts (a switching-transient model). |
| class | `RFISource` | A vector-field RFI source evaluated at lab-frame positions. |
| class | `UniformPlaneWaveSource` | A spatially uniform (far-field) RFI source. |
| class | `MagneticDipoleSource` | A near-field magnetic-dipole RFI source with a ``1/r**3`` pattern. |
| function | `total_field(sources: list[RFISource] | tuple[RFISource, ...], positions: np.ndarray) -> np.ndarray` | Return the summed field ``(P, 3, N)`` of several sources at ``positions``. |

## `spin_dynamics.interference.trackers`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `kalman_harmonic_canceller(primary: np.ndarray, frequencies_hz: np.ndarray | list[float] | tuple[float, ...], sample_rate_hz: float, *, update_mask: np.ndarray | None = None, process_std: float = 0.001, measurement_std: float = 1.0, initial_amplitude_std: float = 1.0) -> CancellationResult` | Track and subtract drifting narrowband carriers with a Kalman filter. |

## `spin_dynamics.motion`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MotionFieldMaps2D` | Two-dimensional field maps used by moving isochromats. |
| class | `MotionFieldMaps` | N-dimensional (1-, 2-, or 3-D) field maps for moving isochromats. |
| class | `ParticleEnsemble` | Moving isochromat ensemble. |
| function | `make_motion_field_maps_2d(x_axis: Iterable[float] | np.ndarray, z_axis: Iterable[float] | np.ndarray, *, b0_map: Iterable[float] | np.ndarray | None = None, b0_vector_map: Iterable[float] | np.ndarray | None = None, b1_tx_map: Iterable[float] | np.ndarray | None = None, b1_tx_vector_map: Iterable[float] | np.ndarray | None = None, b1_rx_map: Iterable[float] | np.ndarray | None = None, b1_rx_vector_map: Iterable[float] | np.ndarray | None = None) -> MotionFieldMaps2D` | Validate and assemble two-dimensional field maps. |
| function | `make_motion_field_maps(axes: SpatialDomain | Sequence[Iterable[float] | np.ndarray], *, b0_map: Iterable[float] | np.ndarray | None = None, b1_tx_map: Iterable[float] | np.ndarray | None = None, b1_rx_map: Iterable[float] | np.ndarray | None = None) -> MotionFieldMaps` | Assemble 1-, 2-, or 3-D motion field maps over ``axes``. |
| function | `transverse_b1_magnitude(b0_vector_map: Iterable[float] | np.ndarray, b1_vector_map: Iterable[float] | np.ndarray) -> np.ndarray` | Return the local B1 magnitude perpendicular to the local B0 direction. |
| function | `circular_b1_component_magnitude(b0_vector_map: Iterable[float] | np.ndarray, b1_vector_map: Iterable[float] | np.ndarray, *, handedness: int = 1) -> np.ndarray` | Return the resonant circular B1 component for high-field NMR/MRI. |
| function | `initialize_ensemble_from_density(rho: Iterable[float] | np.ndarray, x_axis: Iterable[float] | np.ndarray, z_axis: Iterable[float] | np.ndarray, *, walkers_per_cell: int = 1, diffusion_coefficient: float | Iterable[float] | np.ndarray = 0.0, seed: int | None = None, jitter: bool = False) -> ParticleEnsemble` | Create a walker ensemble from a two-dimensional spin-density map. |
| function | `initialize_ensemble_from_domain(domain: SpatialDomain, rho: Iterable[float] | np.ndarray, *, walkers_per_cell: int = 1, diffusion_coefficient: float | Iterable[float] | np.ndarray = 0.0, seed: int | None = None, jitter: bool = False) -> ParticleEnsemble` | Create a walker ensemble from a 1-, 2-, or 3-D spin-density volume. |
| function | `advect_diffuse_positions(positions: np.ndarray, dt: float, *, velocity: Velocity = None, diffusion_coefficient: float | Iterable[float] | np.ndarray = 0.0, rng: np.random.Generator | None = None, time: float = 0.0, bounds: tuple[tuple[float, float], tuple[float, float]] | None = None, boundary: Boundary = 'reflect') -> np.ndarray` | Advance positions with deterministic advection and Brownian diffusion. |
| function | `move_ensemble(ensemble: ParticleEnsemble, dt: float, *, velocity: Velocity = None, rng: np.random.Generator | None = None, time: float = 0.0, bounds: tuple[tuple[float, float], tuple[float, float]] | None = None, boundary: Boundary = 'reflect') -> ParticleEnsemble` | Return an ensemble with advected/diffused positions. |
| function | `apply_free_precession(ensemble: ParticleEnsemble, dt: float, off_resonance: Iterable[float] | np.ndarray, *, t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0) -> ParticleEnsemble` | Apply relaxation and off-resonance precession to each particle. |
| function | `apply_rf_rotation(ensemble: ParticleEnsemble, duration: float, phase: float, amplitude: float, off_resonance: Iterable[float] | np.ndarray, *, b1_tx: float | Iterable[float] | np.ndarray = 1.0) -> ParticleEnsemble` | Apply a rectangular RF rotation using local B1 transmit scaling. |
| function | `free_precession_with_motion_step(ensemble: ParticleEnsemble, fields: MotionFieldMaps2D, dt: float, *, velocity: Velocity = None, rng: np.random.Generator | None = None, time: float = 0.0, gradient: tuple[float, float] = (0.0, 0.0), t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0, boundary: Boundary = 'reflect') -> ParticleEnsemble` | Move particles and apply a first-order free-precession update. |
| function | `receive_signal(ensemble: ParticleEnsemble, fields: MotionFieldMaps2D | None = None) -> complex` | Sum weighted received transverse magnetization over particles. |
| function | `make_circular_reflector(center: tuple[float, float], radius: float) -> BoundaryFn` | Return a reflecting-wall boundary callback for a circular pore. |
| function | `make_elliptical_reflector(center: tuple[float, float], semi_axes: tuple[float, float]) -> BoundaryFn` | Return a reflecting-wall boundary callback for an elliptical pore. |
| function | `make_semipermeable_plane(interface: float, exchange_rate: float, *, axis: Literal['x', 'z'] = 'x', outer_boundary: BoundaryMode = 'reflect') -> BoundaryFn` | Return a stochastic semi-permeable internal plane boundary. |
| function | `apply_boundary(positions: np.ndarray, bounds: tuple[tuple[float, float], tuple[float, float]], mode: Boundary, *, previous_positions: np.ndarray | None = None, rng: np.random.Generator | None = None, time: float = 0.0, dt: float = 0.0) -> np.ndarray` | Apply boundary conditions to two-dimensional positions. |

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
| function | `tuned_probe_output_noise_density(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, sample: Any | None = None) -> tuple[np.ndarray, np.ndarray]` | Return tuned-probe output-referred noise density and frequencies. |
| function | `untuned_probe_output_noise_density(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, sample: Any | None = None) -> tuple[np.ndarray, np.ndarray]` | Return untuned-probe output-referred noise density and frequencies. |
| function | `matched_probe_output_noise_density(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, tf1: np.ndarray | None = None, sample: Any | None = None) -> tuple[np.ndarray, np.ndarray]` | Return matched-probe output-referred noise density and frequencies. |
| function | `frequency_bin_width(frequencies: np.ndarray) -> float` | Estimate a representative frequency-bin width. |

## `spin_dynamics.nonresonant.field_reversal`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `NonresonantFieldModel` | Two orthogonal switching coils plus a non-reversible background field. |
| class | `IsochromatEnsemble` | Per-isochromat coil directions, field scales, and weights. |
| function | `sample_isochromats(model: NonresonantFieldModel, n: int, *, b_inhomogeneity: float = 0.25, a_inhomogeneity: float = 0.0, direction_tilt_deg: float = 15.0, seed: int = 0) -> IsochromatEnsemble` | Sample an isochromat ensemble with coil field and direction inhomogeneity. |
| function | `rodrigues_rotate(vectors: np.ndarray, axis_unit: np.ndarray, angle) -> np.ndarray` | Rotate ``(N, 3)`` vectors about per-row unit axes by ``angle`` (Rodrigues). |
| class | `FieldSegment` | One piecewise-constant stretch of a nonresonant sequence. |
| function | `sequence_waveform(unit, *, repeats: int = 1)` | Return ``(times, i_a, i_b)`` step arrays of the coil currents over the sequence. |
| function | `evolve_segment(magnetization: np.ndarray, ensemble: IsochromatEnsemble, segment: FieldSegment) -> np.ndarray` | Evolve the ``(N, 3)`` magnetization through one :class:`FieldSegment`. |
| class | `FieldReversalResult` | Echo train from a nonresonant field-reversal sequence. |
| function | `simulate_field_reversal_echoes(model: NonresonantFieldModel, ensemble: IsochromatEnsemble, unit: Sequence[FieldSegment] | Sequence[Sequence[FieldSegment]], *, num_echoes: int, echo_spacing_seconds: float | None = None, t2_seconds: float = np.inf, initial_direction = None, return_magnetization: bool = False) -> FieldReversalResult` | Simulate a repeated nonresonant refocusing unit and read the echo train. |

## `spin_dynamics.nonresonant.sequences`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `basic_reversal_sequence(model: NonresonantFieldModel, *, echo_spacing_seconds: float, tau_rev_seconds: float = 0.0, reversal_steps: int = 16) -> list[FieldSegment]` | The basic nonresonant sequence (Brill 2002 Fig. 1B): periodic ``B_B`` reversal. |
| function | `csar_sequence(model: NonresonantFieldModel, *, echo_spacing_seconds: float, tau_rev_seconds: float = 0.0, free_fraction: float = 0.1, reversal_steps: int = 16, adiabatic_steps: int = 160, sense: int = 1) -> list[FieldSegment]` | A 90-degree CSAR refocusing unit (Brill 2002 Fig. 1D / Fig. 3A). |
| function | `csar_double_reversal_sequence(model: NonresonantFieldModel, *, echo_spacing_seconds: float, tau_rev_seconds: float = 0.0, free_fraction: float = 0.1, reversal_steps: int = 16, adiabatic_steps: int = 80) -> list[FieldSegment]` | A 2-pi CSAR refocusing unit (Brill 2002 Fig. 3B): two reversals per echo period. |
| function | `csar_supercycle_sequence(model: NonresonantFieldModel, *, echo_spacing_seconds: float, tau_rev_seconds: float = 0.0, senses: tuple[int, ...] = (1, 1, -1, -1), **kwargs) -> list[list[FieldSegment]]` | The Fig. 3C supercycle: 90-degree CSAR units with alternating rotation senses. |
| function | `effective_rotation(ensemble: IsochromatEnsemble, unit, isochromat_index: int = 0) -> tuple[np.ndarray, float]` | Return the ``(axis, angle)`` of one isochromat's echo-to-echo net rotation. |

## `spin_dynamics.nqr.crossover`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CrossoverOrientation` | Static-field, transmit, and receive directions for one crystallite. |
| class | `CrossoverTransition` | One observable transition between energy manifolds. |
| class | `CrossoverSpectrumResult` | Exact equilibrium spectrum across the NQR-to-NMR crossover. |
| class | `CrossoverFieldSweepResult` | Overlap-tracked eigenstates and transition responses over a B0 sweep. |
| class | `PowderCrossoverSweepResult` | Powder-averaged spectra on one frequency axis over a B0 sweep. |
| function | `boltzmann_populations(levels_hz: np.ndarray | Sequence[float], temperature_kelvin: float) -> np.ndarray` | Return normalized Boltzmann populations for energies supplied in Hz. |
| function | `crossover_transitions_from_eigensystem(eigensystem: NQREigensystem, orientation: CrossoverOrientation, *, orientation_index: int = 0, temperature_kelvin: float = 300.0, degeneracy_tolerance_hz: float | None = None, coupling_tolerance: float = 1e-12) -> tuple[CrossoverTransition, ...]` | Return degeneracy-safe transition responses for one orientation. |
| function | `track_crossover_field_sweep(site: QuadrupolarSite, b0_tesla: np.ndarray | Sequence[float], *, orientation: CrossoverOrientation | None = None, temperature_kelvin: float = 300.0, backend: str = 'numpy') -> CrossoverFieldSweepResult` | Track eigenstate character and all state-pair responses over increasing B0. |
| function | `simulate_crossover_powder_sweep(site: QuadrupolarSite, b0_tesla: np.ndarray | Sequence[float], *, n_theta: int = 4, n_phi: int = 8, n_chi: int = 4, b1_b0_angle: float = np.pi / 2.0, temperature_kelvin: float = 300.0, broadening_hz: float = 1000.0, frequency_points: int = 512, frequency_range_hz: tuple[float, float] | None = None, lineshape: str = 'gaussian', normalize_each_field: bool = True, backend: str = 'numpy') -> PowderCrossoverSweepResult` | Return optional powder-averaged crossover spectra over several fields. |
| function | `simulate_crossover_spectrum(site: QuadrupolarSite, b0_tesla: float, *, orientations: CrossoverOrientationInput = 'single', temperature_kelvin: float = 300.0, broadening_hz: float = 100.0, points: int = 2048, frequency_range_hz: tuple[float, float] | None = None, lineshape: str = 'gaussian', normalize: bool = False, degeneracy_tolerance_hz: float | None = None, coupling_tolerance: float = 1e-12, backend: str = 'numpy') -> CrossoverSpectrumResult` | Simulate every observable transition of ``H_Q + H_Z`` exactly. |

## `spin_dynamics.nqr.crossover_sequences`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CrossoverSLSEResult` | Single-crystal SLSE echo train across an arbitrary static-field regime. |
| class | `PowderCrossoverSLSEResult` | Orientation-averaged exact-pulse SLSE echo train. |
| function | `powder_carrier_frequency_hz(site: QuadrupolarSite, b0_tesla: float, orientations: Sequence[OrientationSample], *, nutation_hz: float) -> float` | Choose one intensity-weighted carrier from a powder orientation set. |
| function | `select_powder_frequency_slice(site: QuadrupolarSite, b0_tesla: float, orientations: Sequence[OrientationSample], *, carrier_frequency_hz: float, half_width_hz: float) -> tuple[tuple[OrientationSample, ...], float]` | Select crystallites with an RF-active transition in a receiver slice. |
| function | `matched_filter_echo_waveforms(coherent_waveforms: np.ndarray, acquisition_offsets_seconds: Sequence[float] | np.ndarray, *, receiver_bandwidth_hz: float = np.inf) -> tuple[np.ndarray, np.ndarray]` | Filter coherent echo waveforms and project them onto the first echo. |
| function | `simulate_crossover_slse(site: QuadrupolarSite, b0_vector_tesla_pas: Sequence[float] | np.ndarray, *, nutation_hz: float, excitation_duration_seconds: float, refocus_duration_seconds: float, echo_spacing_seconds: float, num_echoes: int, relaxation: RelaxationSuperoperator, rf_frequency_hz: float | None = None, excitation_phase_radians: float = 0.0, refocus_phase_radians: float = np.pi / 2.0, b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0), receive_quadrature_direction_pas: Sequence[float] | np.ndarray | None = None, detection_mode: str = 'baseband', floquet_sidebands: int = 4, initial_density: np.ndarray | None = None, acquisition_offsets_seconds: Sequence[float] | np.ndarray | None = None, pulse_model: str = 'floquet') -> CrossoverSLSEResult` | Simulate an SLSE train with Floquet pulses and static-field dissipation. |
| function | `simulate_crossover_slse_powder(site: QuadrupolarSite, b0_tesla: float, *, nutation_hz: float, excitation_duration_seconds: float, refocus_duration_seconds: float, echo_spacing_seconds: float, num_echoes: int, relaxation: RelaxationSuperoperator, rf_frequency_hz: float | None = None, excitation_phase_radians: float = 0.0, refocus_phase_radians: float = np.pi / 2.0, n_theta: int = 3, n_phi: int = 6, n_chi: int = 4, b1_b0_angle_radians: float = np.pi / 2.0, floquet_sidebands: int = 4, acquisition_duration_seconds: float | None = None, acquisition_points: int = 129, receiver_bandwidth_hz: float = np.inf, orientations: Sequence[OrientationSample] | None = None, pulse_model: str = 'floquet', num_workers: int | None = 1, parallel_backend: str = 'thread', retain_local_results: bool = True) -> PowderCrossoverSLSEResult` | Powder-average an exact-pulse SLSE train using one global carrier. |

## `spin_dynamics.nqr.floquet`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `FloquetRFResult` | Final state from a finite-sideband monochromatic Floquet calculation. |
| function | `linear_rf_floquet_hamiltonian(site: QuadrupolarSite, b0_vector_tesla_pas: Sequence[float] | np.ndarray, *, nutation_hz: float, rf_frequency_hz: float, phase_radians: float = 0.0, b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0), sidebands: int = 3) -> np.ndarray` | Return the finite Sambe-space Hamiltonian for a linear cosine field. |
| function | `simulate_floquet_rf(site: QuadrupolarSite, b0_vector_tesla_pas: Sequence[float] | np.ndarray, *, nutation_hz: float, rf_frequency_hz: float, pulse_duration_seconds: float, phase_radians: float = 0.0, b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0), sidebands: int = 3, initial_density: np.ndarray | None = None, temperature_kelvin: float = 300.0) -> FloquetRFResult` | Propagate one constant-envelope linear RF pulse with Floquet sidebands. |

## `spin_dynamics.nqr.field_relaxation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `FieldEquilibriumResult` | Exact Gibbs state of one quadrupolar site at a specified static field. |
| class | `FieldRelaxationResult` | Density and spin-expectation trajectories under a fixed static field. |
| class | `FieldSweepHistoryResult` | History-dependent density trajectory through a vector static-field ramp. |
| function | `field_dependent_equilibrium(site: QuadrupolarSite, b0_vector_tesla_pas: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0), *, temperature_kelvin: float = 300.0) -> FieldEquilibriumResult` | Return the normalized Gibbs state of the complete ``H_Q + H_Z``. |
| class | `FieldDependentRelaxationModel` | Completely-positive relaxation toward the Gibbs state of ``H_Q + H_Z``. |
| class | `FieldDependentDaviesRelaxationModel` | Thermal secular relaxation with field-dependent transition rates. |
| class | `FieldDependentNonsecularRelaxationModel` | Unified-GKLS relaxation for clusters of unresolved Bohr frequencies. |
| function | `simulate_field_relaxation(site: QuadrupolarSite, b0_vector_tesla_pas: Sequence[float] | np.ndarray, times_seconds: Sequence[float] | np.ndarray, *, relaxation: FieldDependentRelaxationModel | FieldDependentDaviesRelaxationModel, initial_density: np.ndarray | None = None) -> FieldRelaxationResult` | Propagate a density matrix while relaxing at one fixed static field. |
| function | `simulate_field_sweep_history(site: QuadrupolarSite, times_seconds: Sequence[float] | np.ndarray, b0_vectors_tesla_pas: Sequence[Sequence[float]] | np.ndarray, *, relaxation: RelaxationSuperoperator | None = None, temperature_kelvin: float | None = None, initial_density: np.ndarray | None = None, substeps_per_interval: int = 1) -> FieldSweepHistoryResult` | Carry one density matrix through a piecewise-linear vector-field history. |

## `spin_dynamics.nqr.full_dynamics`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `rf_operator_eigenbasis(eigensystem: NQREigensystem, direction) -> np.ndarray` | Return ``e1 . I`` for unit direction ``e1`` in the energy eigenbasis. |
| function | `rotating_indices(levels_hz: np.ndarray, rf_frequency_hz: float) -> np.ndarray` | Return RWA winding numbers ``round((nu_i - min nu) / nu_rf)`` per level. |
| function | `static_hamiltonian_rotating(eigensystem: NQREigensystem, rf_frequency_hz: float) -> np.ndarray` | Return the rotating-frame static Hamiltonian (rad/s) in the eigenbasis. |
| function | `pulse_hamiltonian(eigensystem: NQREigensystem, *, nutation_hz: float, rf_frequency_hz: float, phase: float = 0.0, b1_direction_pas = (1.0, 0.0, 0.0)) -> np.ndarray` | Return the rotating-frame RWA pulse Hamiltonian (rad/s) in the eigenbasis. |
| function | `detection_operator(eigensystem: NQREigensystem, rf_frequency_hz: float, rx_direction_pas = (1.0, 0.0, 0.0)) -> np.ndarray` | Return the baseband receive observable ``M`` with ``s = Tr(rho M)``. |
| class | `CoilDrive` | One linearly-polarized RF coil in a multi-axis excitation set. |
| function | `multi_axis_pulse_hamiltonian(eigensystem: NQREigensystem, coils: Sequence[CoilDrive], *, rf_frequency_hz: float) -> np.ndarray` | Return the rotating-frame RWA Hamiltonian for several RF coils at once. |
| function | `circular_pulse_hamiltonian(eigensystem: NQREigensystem, *, nutation_hz: float, rf_frequency_hz: float, axis1_pas = (1.0, 0.0, 0.0), axis2_pas = (0.0, 1.0, 0.0), helicity: int = 1, phase: float = 0.0) -> np.ndarray` | Return the pulse Hamiltonian for a circularly-polarized (quadrature) drive. |
| function | `quadrature_detection_operator(eigensystem: NQREigensystem, rf_frequency_hz: float, axis1_pas = (1.0, 0.0, 0.0), axis2_pas = (0.0, 1.0, 0.0), helicity: int = 1) -> np.ndarray` | Return the matched quadrature (circular) receive observable ``M``. |
| class | `FullNQRFIDResult` | Complex baseband FID from the full density-matrix model. |
| class | `FullNQREchoResult` | Complex baseband echo from a full density-matrix two-pulse sequence. |
| function | `simulate_full_fid(site: QuadrupolarSite, *, nutation_hz: float, pulse_duration_seconds: float, times_seconds: np.ndarray, rf_frequency_hz: float | None = None, phase: float = 0.0, b1_direction_pas = (1.0, 0.0, 0.0), rx_direction_pas = None, b0_vector_tesla_pas = None, relaxation: NQRRelaxationLike | None = None, initial_density: np.ndarray | None = None) -> FullNQRFIDResult` | Simulate a single-pulse full density-matrix NQR FID. |
| function | `simulate_full_echo(site: QuadrupolarSite, *, nutation_hz: float, excitation_duration_seconds: float, refocus_duration_seconds: float, echo_spacing_seconds: float, times_seconds: np.ndarray, rf_frequency_hz: float | None = None, excitation_phase: float = 0.0, refocus_phase: float = np.pi / 2, b1_direction_pas = (1.0, 0.0, 0.0), rx_direction_pas = None, b0_vector_tesla_pas = None, relaxation: NQRRelaxationLike | None = None) -> FullNQREchoResult` | Simulate a full density-matrix two-pulse (Hahn-style) NQR echo. |
| class | `FullNQRSLSEResult` | Spin-lock spin-echo (SLSE) train from the full density-matrix model. |
| function | `simulate_full_slse(site: QuadrupolarSite, *, nutation_hz: float, excitation_duration_seconds: float, refocus_duration_seconds: float, echo_spacing_seconds: float, num_echoes: int, rf_frequency_hz: float | None = None, excitation_phase: float = 0.0, refocus_phase: float = np.pi / 2.0, orientations: OrientationInput = 'single', b0_tesla: float = 0.0, b1_direction_pas = (1.0, 0.0, 0.0), rx_direction_pas = None, relaxation: NQRRelaxationLike | None = None, t2e_seconds: float = np.inf) -> FullNQRSLSEResult` | Simulate a full density-matrix SLSE echo train (valid for spin-3/2). |

## `spin_dynamics.nqr.hamiltonians`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `quadrupole_frequency_scale_hz(site: QuadrupolarSite) -> float` | Return the Hamiltonian prefactor matching the public frequency parameter. |
| function | `quadrupole_hamiltonian(site: QuadrupolarSite) -> np.ndarray` | Return the zero-field quadrupole Hamiltonian in radians per second. |
| function | `zeeman_hamiltonian(site: QuadrupolarSite, b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray` | Return the Zeeman Hamiltonian in radians per second. |
| function | `nqr_hamiltonian(site: QuadrupolarSite, b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float] | None = None) -> np.ndarray` | Return the quadrupole plus optional Zeeman Hamiltonian. |
| function | `diagonalize_site(site: QuadrupolarSite, b0_vector_tesla_pas: np.ndarray | list[float] | tuple[float, float, float] | None = None, *, strength_tolerance: float = 1e-12, frequency_tolerance_hz: float = 1e-09) -> NQREigensystem` | Diagonalize a site Hamiltonian and return transition metadata. |
| function | `batched_nqr_hamiltonians(site: QuadrupolarSite, b0_vectors_tesla_pas: np.ndarray | list[list[float]]) -> np.ndarray` | Return one Hamiltonian per static-field vector, shape ``(N, dim, dim)``. |
| function | `diagonalize_sites_over_b0(site: QuadrupolarSite, b0_vectors_tesla_pas: np.ndarray | list[list[float]], *, backend: str = 'numpy', strength_tolerance: float = 1e-12, frequency_tolerance_hz: float = 1e-09) -> tuple[NQREigensystem, ...]` | Diagonalize one site across many static-field vectors with one ``eigh``. |

## `spin_dynamics.nqr.inhomogeneity`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `EFGIsochromat` | One static EFG variant in an inhomogeneous NQR ensemble. |
| class | `EFGDistribution` | Normalized collection of static quadrupolar-site variants. |
| class | `NQRFIDDistributionResult` | Time-domain and spectral NQR signal from an EFG distribution. |
| class | `SLSEDistributionResult` | SLSE echo train summed over a static EFG distribution. |
| class | `WindowDeconvolutionResult` | Regularized deconvolution of a finite acquisition-window spectrum. |
| class | `SLSEAcquisitionSpectrumResult` | Finite-window SLSE echo acquisition and spectrum. |
| class | `EFGRephasingAnalysis` | Discretization estimate for a static EFG frequency grid. |
| function | `analyze_efg_rephasing(frequencies_hz: np.ndarray | list[float] | tuple[float, ...], max_time_seconds: float, safety_factor: float = 1.25) -> EFGRephasingAnalysis` | Estimate whether an EFG isochromat grid may discretely rephase. |
| function | `check_efg_rephasing(frequencies_hz: np.ndarray | list[float] | tuple[float, ...], max_time_seconds: float, safety_factor: float = 1.25, action: str = 'warn') -> EFGRephasingAnalysis` | Warn or raise when an EFG grid may produce rephasing artifacts. |
| function | `gaussian_efg_distribution(base_site: QuadrupolarSite, *, quadrupole_std_hz: float = 0.0, eta_std: float = 0.0, samples: int = 31, sigma_span: float = 3.0) -> EFGDistribution` | Return a Gaussian static distribution of ``nu_Q`` and optionally ``eta``. |
| function | `temperature_efg_distribution(base_site: QuadrupolarSite, temperatures_kelvin: np.ndarray | list[float] | tuple[float, ...], *, weights: np.ndarray | list[float] | tuple[float, ...] | None = None, reference_temperature_kelvin: float = 293.15, quadrupole_slope_hz_per_kelvin: float = 0.0, eta_slope_per_kelvin: float = 0.0) -> EFGDistribution` | Map a static temperature distribution onto ``nu_Q`` and ``eta`` shifts. |
| function | `fid_spectrum(signal: np.ndarray, times: np.ndarray, *, zero_fill_factor: int = 2, window: str = 'hann') -> tuple[np.ndarray, np.ndarray]` | Return a centered FFT spectrum for a uniformly sampled complex FID. |
| function | `deconvolve_acquisition_window(spectrum: np.ndarray, frequencies_hz: np.ndarray, acquisition_times_seconds: np.ndarray, *, regularization_strength: float = 0.01) -> WindowDeconvolutionResult` | Deconvolve the finite acquisition window with Tikhonov regularization. |
| function | `efg_line_spectrum(distribution: EFGDistribution, transition_label: str, *, carrier_frequency_hz: float | None = None, amplitudes: np.ndarray | list[complex] | tuple[complex, ...] | None = None, linewidth_hz: float = 100.0, points: int = 1024, span_hz: float | None = None) -> tuple[np.ndarray, np.ndarray]` | Return a smoothed line spectrum for a static EFG distribution. |
| function | `simulate_fid_efg_distribution(distribution: EFGDistribution, transition_label: str, times_seconds: np.ndarray | list[float] | tuple[float, ...], *, excitation: SelectivePulse, carrier_frequency_hz: float | None = None, orientations: OrientationInput = 'single', t2_seconds: float = np.inf, zero_fill_factor: int = 2, window: str = 'hann', rephase_action: str = 'warn', rephase_safety_factor: float = 1.25) -> NQRFIDDistributionResult` | Simulate a selective-pulse NQR FID from a static EFG distribution. |
| function | `simulate_slse_efg_distribution(distribution: EFGDistribution, sequence, *, orientations: OrientationInput = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf, relaxation = None, rephase_action: str = 'warn', rephase_safety_factor: float = 1.25) -> SLSEDistributionResult` | Simulate an SLSE echo train summed over a static EFG distribution. |
| function | `simulate_slse_acquisition_spectrum(distribution: EFGDistribution, sequence, *, acquisition_duration_seconds: float, acquisition_points: int = 256, echo_index: int = -1, carrier_frequency_hz: float | None = None, orientations: OrientationInput = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf, relaxation = None, zero_fill_factor: int = 2, spectrum_window: str = 'none', noise: NoiseSpec | Mapping | float | int | None = None, deconvolution_strength: float | None = None, rephase_action: str = 'warn', rephase_safety_factor: float = 1.25) -> SLSEAcquisitionSpectrumResult` | Simulate the spectrum of one finite-window acquired SLSE echo. |

## `spin_dynamics.nqr.interference`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `slse_acquisition_mask(sequence: SLSESequence, sample_rate_hz: float, *, ringdown_seconds: float = 0.0, pre_baseline_seconds: float = 0.0, post_baseline_seconds: float = 0.0, baseline_intervals: Sequence[Interval] | None = None) -> AcquisitionMask` | Return a transmit/ringdown/signal/baseline mask for an SLSE train. |
| function | `sorc_acquisition_mask(sequence: SORCSequence, sample_rate_hz: float, *, ringdown_seconds: float = 0.0, pre_baseline_seconds: float = 0.0, post_baseline_seconds: float = 0.0, baseline_intervals: Sequence[Interval] | None = None, initial_gap_is_baseline: bool = True) -> AcquisitionMask` | Return a transmit/ringdown/signal/baseline mask for a SORC train. |
| function | `slse_mask_from_metadata(num_samples: int, sample_rate_hz: float, *, echo_spacing_seconds: float, detection_duration_seconds: float, num_echoes: int, ringdown_seconds: float = 0.0, start_offset_seconds: float = 0.0, post_baseline_seconds: float = 0.0, baseline_intervals: Sequence[Interval] | None = None) -> AcquisitionMask` | Reconstruct an SLSE acquisition mask over a window of ``num_samples`` ADC samples. |
| function | `sorc_mask_from_metadata(num_samples: int, sample_rate_hz: float, *, half_spacing_seconds: float, detection_duration_seconds: float, num_pulses: int, ringdown_seconds: float = 0.0, start_offset_seconds: float = 0.0, post_baseline_seconds: float = 0.0, baseline_intervals: Sequence[Interval] | None = None, initial_gap_is_baseline: bool = True) -> AcquisitionMask` | Reconstruct a SORC acquisition mask over a window of ``num_samples`` ADC samples. |
| function | `nqr_recording_from_samples(primary: np.ndarray, references: np.ndarray | None, sample_rate_hz: float, *, sequence: str = 'slse', **timing) -> RFIRecording` | Pair measured ADC windows with a mask reconstructed from sequence timing. |

## `spin_dynamics.nqr.lab_frame`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `LabFrameRFResult` | Density trajectory and detected signal after each RF waveform segment. |
| function | `sample_linear_rf_pulse(duration_seconds: float, time_step_seconds: float, amplitude_tesla: float, carrier_hz: float, *, phase_radians: float = 0.0, direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]` | Sample a linearly polarized cosine RF pulse at segment midpoints. |
| function | `simulate_lab_frame_rf(site: QuadrupolarSite, b0_vector_tesla_pas: Sequence[float] | np.ndarray, segment_durations_seconds: Sequence[float] | np.ndarray, rf_fields_tesla_pas: Sequence[Sequence[float]] | np.ndarray, *, initial_density: np.ndarray | None = None, temperature_kelvin: float = 300.0, receive_direction_pas: Sequence[complex] | np.ndarray = (1.0, 0.0, 0.0), relaxation: RelaxationSuperoperator | None = None) -> LabFrameRFResult` | Propagate an arbitrary sampled RF waveform without an RWA. |

## `spin_dynamics.nqr.model_selection`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `NQRModelSelection` | Recommendation and diagnostics for the reduced-vs-full modeling choice. |
| function | `select_nqr_model(site: QuadrupolarSite, target: str | NQRTransition, *, nutation_hz: float, pulse_duration_seconds: float, b1_direction_pas = (1.0, 0.0, 0.0), linewidth_hz: float = 0.0, b0_vector_tesla_pas = None, rf_frequency_hz: float | None = None, isolation_threshold: float = 5.0, coupling_tolerance: float = 0.01) -> NQRModelSelection` | Recommend the reduced or full NQR model for a pulse on one transition. |

## `spin_dynamics.nqr.operators`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `validate_spin(spin: float) -> float` | Return a validated integer or half-integer spin quantum number. |
| function | `spin_dimension(spin: float) -> int` | Return the Hilbert-space dimension for one spin. |
| function | `is_half_integer_spin(spin: float) -> bool` | Return ``True`` for half-integer spin (1/2, 3/2, 5/2, ...). |
| function | `fundamental_operator_gap(spin: float) -> float` | Return the ``eta=0`` fundamental-transition gap ``d`` of the quadrupole operator. |
| class | `SpinMatrices` | Dense single-spin angular momentum matrices. |
| function | `spin_matrices(spin: float) -> SpinMatrices` | Return dense angular-momentum matrices for one spin. |

## `spin_dynamics.nqr.orientations`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `spherical_direction(alpha: float, beta: float) -> np.ndarray` | Return a unit vector from azimuth `alpha` and polar angle `beta`. |
| class | `OrientationSample` | One local EFG orientation relative to lab RF and static fields. |
| function | `single_crystal_orientation(alpha: float, beta: float, *, b0_alpha: float | None = None, b0_beta: float | None = None) -> tuple[OrientationSample, ...]` | Return a one-sample orientation ensemble. |
| function | `powder_average_grid(n_theta: int = 16, n_phi: int = 32) -> tuple[OrientationSample, ...]` | Return a normalized spherical powder-average grid. |
| class | `OrientationFrame` | A full crystallite orientation as an orthonormal lab-to-PAS frame. |
| function | `b0_b1_powder_average_grid(n_theta: int = 12, n_phi: int = 24, n_chi: int = 8, *, b1_b0_angle: float = np.pi / 2.0) -> tuple[OrientationSample, ...]` | Return a powder grid with correlated lab B0 and RF B1 directions. |
| function | `b0_b1_powder_average_halton(num_samples: int, *, start_index: int = 1, b1_b0_angle: float = np.pi / 2.0) -> tuple[OrientationSample, ...]` | Return a low-discrepancy SO(3) powder sequence for correlated B0/B1. |
| function | `powder_frame_grid(n_theta: int = 12, n_phi: int = 24, n_chi: int = 8) -> tuple[OrientationFrame, ...]` | Return a normalized SO(3) powder grid of full lab-to-PAS frames. |
| function | `b0_powder_average_grid(n_theta: int = 16, n_phi: int = 32, *, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0)) -> tuple[OrientationSample, ...]` | Return a powder grid over static-field directions in the PAS. |
| function | `normalize_orientations(orientations: tuple[OrientationSample, ...] | list[OrientationSample]) -> tuple[OrientationSample, ...]` | Return orientation samples with weights normalized to unity. |

## `spin_dynamics.nqr.piezo_detection`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `PiezoelectricCrystal` | Rectangular piezoelectric sample between full-area electrodes. |
| class | `PiezoNQRLine` | One quadrupolar transition used by the detection model. |
| class | `PiezoNQRCoupling` | Strain-to-transition coupling and drive geometry. |
| class | `PiezoDetectionResult` | Detection estimate for one piezoelectrically driven NQR line. |
| function | `glycine_crystal_from_cif(path: str | Path, *, thickness_m: float = 0.0005, electrode_area_m2: float = 2.5e-05, d_eff_m_per_v: float = 6.1e-12, relative_permittivity: float = 10.0, sound_velocity_m_s: float = 2500.0, mechanical_q: float = 100.0, mode_shape_factor: float = 0.5) -> PiezoelectricCrystal` | Build glycine crystal parameters from CIF density and formula data. |
| function | `glycine_site_from_qcc(qcc_hz: float = 1193000.0, eta: float = 0.528) -> QuadrupolarSite` | Return the spin-1 ``14N`` glycine site using the NQRDatabase convention. |
| function | `default_glycine_nqr_lines() -> tuple[PiezoNQRLine, ...]` | Return glycine ``14N`` lines from the bundled NQRDatabase values. |
| function | `load_glycine_nqr_lines_from_sqlite(path: str | Path) -> tuple[PiezoNQRLine, ...]` | Load deduplicated glycine ``14N`` lines from an NQRDatabase SQLite export. |
| function | `resonant_strain_peak(crystal: PiezoelectricCrystal, voltage_rms: float) -> float` | Return peak strain at a resonantly enhanced acoustic antinode. |
| function | `simulate_piezoelectric_nqr_detection(crystal: PiezoelectricCrystal, line: PiezoNQRLine, coupling: PiezoNQRCoupling | None = None, *, voltage_rms: float = 10.0, detuning_hz: float = 0.0, temperature_k: float = 300.0, spin_temperature_enhancement: float = 1.0, readout_efficiency: float = 1.0, power_noise_density_w_per_sqrt_hz: float = 1e-15, fractional_noise_density_per_sqrt_hz: float = 1e-06, integration_time_seconds: float = 1.0) -> PiezoDetectionResult` | Estimate electrical/acoustic detectability for one NQR transition. |
| function | `acoustic_strain_energy(crystal: PiezoelectricCrystal, strain_peak: float) -> float` | Return peak standing-wave strain energy for the active volume. |
| function | `nuclear_absorbed_power(spin_count: float, frequency_hz: float, t1_seconds: float, *, temperature_k: float = 300.0, spin_temperature_enhancement: float = 1.0, saturation_parameter: float = 1.0, spin_dimension: int = 3) -> float` | Return steady resonant power absorbed by a high-temperature transition. |

## `spin_dynamics.nqr.polarization_enhancement`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `PolarizationEnhancedNQRSample` | Sample and spin parameters for polarization-enhanced NQR. |
| class | `CylindricalSampleGeometry` | Cylindrical sample volume moved through the magnet. |
| class | `LinearTransportMotion` | Constant-velocity sample motion along one lab axis. |
| class | `HalbachPrepolarizationMagnet` | Finite four-rod Halbach magnet for transport simulations. |
| class | `PolarizationTransferResult` | Result of a polarization-enhanced NQR transport simulation. |
| function | `ideal_spin1_enhancement_factors(line_frequencies_hz: Sequence[float], *, max_b0_tesla: float, protons_per_molecule: float, nitrogens_per_molecule: float, gamma_hz_per_t: float = PROTON_GAMMA_HZ_PER_T) -> np.ndarray` | Return ideal spin-1 enhancement factors from the Glickstein model. |
| function | `simulate_adiabatic_polarization_transfer(magnet: HalbachPrepolarizationMagnet | Callable[[np.ndarray], np.ndarray], sample: PolarizationEnhancedNQRSample, sample_geometry: CylindricalSampleGeometry, motion: LinearTransportMotion, *, prepolarization_time_seconds: float = np.inf, path_points: int = 501, gamma_hz_per_t: float = PROTON_GAMMA_HZ_PER_T, adiabatic_scale: float = 1.0) -> PolarizationTransferResult` | Simulate polarization transfer while a sample moves through a gradient. |

## `spin_dynamics.nqr.pulses`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SelectivePulse` | A rectangular selective RF pulse applied to one NQR transition. |
| function | `transition_drive_scale(transition: NQRTransition, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float]) -> float` | Return the relative RF coupling for a transition and B1 orientation. |
| function | `selective_pulse_hamiltonian(dimension: int, transition: NQRTransition, *, nutation_hz: float, phase: float = 0.0, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0), detuning_hz: float = 0.0) -> np.ndarray` | Return an embedded two-level RF Hamiltonian in radians per second. |
| function | `apply_selective_pulse(density: np.ndarray, transition: NQRTransition, pulse: SelectivePulse, *, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray` | Apply a selective pulse to a density matrix in the energy basis. |

## `spin_dynamics.nqr.relaxation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ZeroFieldRedfieldEFGModel` | Non-diagonal zero-field Redfield model for fluctuating EFGs. |
| class | `SpectralOverlapRelaxationFit` | Linear fit of a background rate plus spectral-overlap relaxation. |
| function | `transition_rms_linewidth_hz(frequency_offsets_hz: Iterable[float] | np.ndarray, intensities: Iterable[float] | np.ndarray, *, intrinsic_sigma_hz: float = 0.0) -> float` | Return the intensity-weighted RMS transition width. |
| function | `spectral_overlap_factors(linewidths_hz: Iterable[float] | np.ndarray, *, reference_index: int = 0, exponent: float = 1.0) -> np.ndarray` | Return normalized overlap factors from transition linewidths. |
| function | `fit_spectral_overlap_relaxation(t2_seconds: Iterable[float] | np.ndarray, overlap_factors: Iterable[float] | np.ndarray, *, rate_standard_errors_per_second: Iterable[float] | np.ndarray | None = None) -> SpectralOverlapRelaxationFit` | Fit ``R2 = R_floor + R_cross * S`` by linear least squares. |

## `spin_dynamics.nqr.rwa_validation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `RWAComparisonResult` | One exact laboratory-frame versus single-band RWA pulse comparison. |
| class | `RWAValidityMap` | RWA error over static-interaction and RF-strength ratios. |
| function | `compare_rwa_to_lab_frame(site: QuadrupolarSite, b0_vector_tesla_pas: Sequence[float] | np.ndarray, *, nutation_hz: float, rf_frequency_hz: float, pulse_duration_seconds: float, phase_radians: float = 0.0, b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0), temperature_kelvin: float = 300.0, samples_per_carrier_cycle: int = 80) -> RWAComparisonResult` | Compare identical linear RF pulses in the exact lab frame and RWA. |
| function | `scan_rwa_validity(site: QuadrupolarSite, interaction_ratios: Sequence[float] | np.ndarray, rf_strength_ratios: Sequence[float] | np.ndarray, *, b0_direction_pas: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0), b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0), detuning_hz: float = 0.0, phase_radians: float = 0.0, duration_in_carrier_cycles: float = 5.0, temperature_kelvin: float = 300.0, samples_per_carrier_cycle: int = 60) -> RWAValidityMap` | Map RWA error while automatically following the strongest RF line. |

## `spin_dynamics.nqr.sequences`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SLSESequence` | Spin-lock spin-echo detection sequence. |
| class | `SORCSequence` | Strong off-resonance comb sequence, ``(tau - phi - tau)^N``. |
| function | `slse_sequence(transition_label: str, *, pulse_duration_seconds: float, nutation_hz: float, echo_spacing_seconds: float, num_echoes: int, phase: float = 0.0, rf_frequency_hz: float | None = None) -> SLSESequence` | Build a rectangular-pulse SLSE sequence. |
| function | `sorc_sequence(transition_label: str, *, pulse_duration_seconds: float, nutation_hz: float, half_spacing_seconds: float, num_pulses: int, phase: float = 0.0, rf_frequency_hz: float | None = None) -> SORCSequence` | Build a rectangular-pulse SORC sequence. |

## `spin_dynamics.nqr.simulation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SLSEResult` | Simulated SLSE echo train. |
| class | `SORCResult` | Simulated strong off-resonance comb observation train. |
| class | `PopulationTransferResult` | Perturbation plus SLSE detection result. |
| class | `SLSESweepResult` | SLSE response as one pulse-sequence parameter is swept. |
| function | `equilibrium_density(levels_hz: np.ndarray) -> np.ndarray` | Return a trace-zero high-temperature density matrix in the energy basis. |
| function | `transition_signal(density: np.ndarray, transition: NQRTransition, *, b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float]) -> complex` | Return the complex single-coil signal for a transition coherence. |
| function | `sorc_powder_theory_signal(frequency_offsets_hz: np.ndarray | list[float] | tuple[float, ...], half_spacing_seconds: float, flip_angle_radians: float, *, quadrature_points: int = 512, normalize: bool = False) -> np.ndarray` | Return the Konnai/Mikhaltsevitch-Rudakov SORC powder response. |
| function | `sorc_powder_pathway_signal(frequency_offsets_hz: np.ndarray | list[float] | tuple[float, ...], half_spacing_seconds: float | np.ndarray | list[float] | tuple[float, ...], flip_angle_radians: float | np.ndarray | list[float] | tuple[float, ...], num_pulses: int, *, quadrature_points: int = 512, normalize: bool = False) -> np.ndarray` | Return the finite-pulse SORC powder pathway recurrence response. |
| function | `fid_powder_theory_signal(flip_angle_radians: np.ndarray | list[float] | tuple[float, ...], *, normalize: bool = False, absolute: bool = False) -> np.ndarray` | Return the spin-1 powder FID pulse-width response used for SORC checks. |
| function | `simulate_slse(site: QuadrupolarSite, sequence: SLSESequence, *, orientations: OrientationInput = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf, initial_density: np.ndarray | None = None, relaxation: NQRRelaxationLike | None = None, backend: str = 'numpy') -> SLSEResult` | Simulate a selective-pulse SLSE echo train. |
| function | `simulate_sorc(site: QuadrupolarSite, sequence: SORCSequence, *, orientations: OrientationInput = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf, initial_density: np.ndarray | None = None, backend: str = 'numpy', model: str = 'pathway') -> SORCResult` | Simulate a spin-1 SORC train. |
| function | `simulate_slse_offset_sweep(site: QuadrupolarSite, transition_label: str, offsets_hz: np.ndarray | list[float] | tuple[float, ...], *, pulse_duration_seconds: float, nutation_hz: float, echo_spacing_seconds: float, num_echoes: int = 16, phase: float = 0.0, orientations: OrientationInput = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf, relaxation: NQRRelaxationLike | None = None, echo_index: int = -1, backend: str = 'numpy') -> SLSESweepResult` | Sweep irradiation offset and return SLSE amplitude and decay estimates. |
| function | `simulate_slse_spacing_sweep(site: QuadrupolarSite, transition_label: str, echo_spacing_seconds: np.ndarray | list[float] | tuple[float, ...], *, pulse_duration_seconds: float, nutation_hz: float, num_echoes: int = 16, phase: float = 0.0, rf_offset_hz: float = 0.0, orientations: OrientationInput = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf, relaxation: NQRRelaxationLike | None = None, echo_index: int = -1, backend: str = 'numpy') -> SLSESweepResult` | Sweep SLSE pulse period and return amplitude plus effective decay. |
| function | `simulate_population_transfer(site: QuadrupolarSite, perturbation: SelectivePulse, detection_sequence: SLSESequence, *, orientations: OrientationInput = 'powder', b0_tesla: float = 0.0, t2e_seconds: float = np.inf) -> PopulationTransferResult` | Simulate a perturbation pulse followed by SLSE detection. |

## `spin_dynamics.nqr.structure_coupling`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CIFAtom` | One atom from a CIF atom-site loop. |
| class | `CIFStructure` | Minimal CIF crystal structure with atom sites and symmetry operations. |
| class | `ProtonDipolarCoupling` | One quadrupolar-nucleus to proton dipolar coupling estimate. |
| class | `ProtonCouplingEstimate` | Nearby-proton coupling summary for one quadrupolar nucleus. |
| function | `load_cif_structure(path: str | Path) -> CIFStructure` | Load atom sites, cell parameters, and symmetry operations from a CIF. |
| function | `estimate_proton_dipolar_couplings_from_cif(path: str | Path, target_label: str, *, proton_radius_angstrom: float = 3.0, gamma_quadrupolar_hz_per_t: float = GAMMA_14N_HZ_PER_T, gamma_proton_hz_per_t: float = PROTON_GAMMA_HZ_PER_T, field_direction: Sequence[float] | None = None, max_periodic_images: int | None = None, max_results: int | None = None) -> ProtonCouplingEstimate` | Estimate nearby-proton dipolar couplings for ``target_label`` in a CIF. |
| function | `estimate_proton_dipolar_couplings(structure: CIFStructure, target_label: str, *, proton_radius_angstrom: float = 3.0, gamma_quadrupolar_hz_per_t: float = GAMMA_14N_HZ_PER_T, gamma_proton_hz_per_t: float = PROTON_GAMMA_HZ_PER_T, field_direction: Sequence[float] | None = None, max_periodic_images: int | None = None, max_results: int | None = None) -> ProtonCouplingEstimate` | Estimate nearby-proton dipolar couplings in a loaded CIF structure. |
| function | `dipolar_coupling_hz(distance_angstrom: float, *, gamma_a_hz_per_t: float = GAMMA_14N_HZ_PER_T, gamma_b_hz_per_t: float = PROTON_GAMMA_HZ_PER_T) -> float` | Return the point-dipole coupling prefactor in Hz. |

## `spin_dynamics.nqr.systems`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `QuadrupolarSite` | One quadrupolar nucleus in its EFG principal-axis system. |
| class | `NQRTransition` | One transition between quadrupolar energy eigenstates. |
| class | `NQREigensystem` | Energy levels, eigenvectors, and allowed transitions for one site. |

## `spin_dynamics.nqr.zeeman`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `WeakB0Transition` | One transition from an exact weak-field NQR diagonalization. |
| class | `WeakB0SpectrumResult` | Powder/single-crystal weak-B0 transition spectrum. |
| function | `zeeman_frequency_hz(site: QuadrupolarSite, b0_tesla: float | np.ndarray | Sequence[float]) -> float` | Return ``|gamma B0|`` in Hz for a scalar or vector static field. |
| function | `weak_field_ratio(site: QuadrupolarSite, b0_tesla: float | np.ndarray | Sequence[float], *, reference_frequency_hz: float | None = None) -> float` | Return ``|gamma B0| / nu_ref`` for weak-field validity checks. |
| function | `simulate_weak_b0_spectrum(site: QuadrupolarSite, b0_tesla: float, *, orientations: OrientationInput = 'single', transition_label: str | None = None, broadening_hz: float = 100.0, points: int = 1024, span_hz: float | None = None, selection_window_hz: float | None = None, intensity_tolerance: float = 1e-14, weak_ratio_action: str = 'warn', weak_ratio_threshold: float = 0.05, backend: str = 'numpy') -> WeakB0SpectrumResult` | Return a broadened transition spectrum for ``H_Q + H_Z`` in weak B0. |

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

## `spin_dynamics.phase_cycling`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `PhaseCycledSequenceBranch` | One phase-programmed :class:`~spin_dynamics.sequences.SequenceIR` branch. |
| class | `PhaseStep` | One scan branch in a phase cycle. |
| class | `PhaseCycle` | A reusable phase-cycle scan table. |
| function | `phase_cycle_sequence_ir(sequence: SequenceIR, phase_cycle: PhaseCycle, *, pulse_blocks: Mapping[str, int | str | Sequence[int | str]] | None = None) -> tuple[PhaseCycledSequenceBranch, ...]` | Create one independently executable ``SequenceIR`` per phase-cycle step. |
| function | `cpmg_two_step_phase_cycle(*, excitation_name: str = 'excitation', excitation_phase_rad: float = np.pi / 2.0) -> PhaseCycle` | Return the default two-step CPMG/PAP excitation phase cycle. |
| function | `pgste_stimulated_echo_phase_cycle() -> PhaseCycle` | Return the selected-pathway PGSTE stimulated-echo phase table. |
| function | `eseem_stimulated_echo_phase_cycle(n_phase: int = 4) -> PhaseCycle` | Return the phase cycle selecting the three-pulse ESEEM stimulated echo. |
| function | `diff_stebp_phase_cycle() -> PhaseCycle` | Return the 16-step Bruker ``diff_stebp`` 13-interval bipolar PGSTE cycle. |

## `spin_dynamics.optimal_control.control_response`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `TailSpec` | How long to propagate past the commanded window for the ring-down/eddy tail. |
| class | `EnvelopeResponse` | Base class for an RF probe modelled as an LTI filter on the control envelope. |
| class | `TunedProbeResponse` | Single-resonance (tuned) probe: a one-pole envelope lowpass with ring-down. |
| class | `UntunedProbeResponse` | Untuned (low-``Q``, broadband) probe: a near-flat envelope with mild phase. |
| class | `MatchedProbeResponse` | Impedance-matched probe: a two-pole envelope, distinct from a tuned probe. |
| class | `GradientDriverResponse` | Gradient driver: an ``L/R`` slew pole in series with eddy-current shelves. |
| function | `eddy_terms_from_step_response(times: Sequence[float], response: Sequence[float], *, n_terms: int = 1, steady_value: float | None = None) -> tuple[tuple[float, float], ...]` | Fit eddy ``(alpha_k, tau_k)`` terms from a measured/simulated step response. |
| class | `ReceiverResponse` | Output-side LTI filter for objectives on the *detected* signal. |
| function | `build_control_delivery(n_segments, dt, *, rf_response = None, gradient_response = None)` | Build the commanded-to-delivered control transform for GRAPE objectives. |

## `spin_dynamics.optimal_control.detection_objective`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `detector_noise_grid(detector, n_acq, dt_acq, *, center_hz = None) -> np.ndarray` | Field-noise PSD ``N^2(f)`` of a detector on the acquisition FFT grid. |
| function | `detected_field_snr_jax(signal, dt_acq, noise_psd_grid, *, xp = None)` | Field-referred matched-filter SNR of a baseband echo against ``N^2(f)``. |
| function | `make_detected_snr_objective(*, drift_batch: Sequence[np.ndarray], hx_batch: Sequence[np.ndarray], hy_batch: Sequence[np.ndarray], psi0: np.ndarray, detection_operator: np.ndarray, weights: np.ndarray, offsets_hz: np.ndarray, dt: float, n_segments: int, amplitude_template: np.ndarray, acquisition_points: int, acquisition_dt: float, detector = None, noise_psd_grid: np.ndarray | None = None, noise_center_hz: float | None = None, optimize_amplitude: bool = False, per_rf_power: bool = False, rf_response = None, propagator: str = 'expm') -> Callable[[np.ndarray], tuple[float, np.ndarray]]` | Build a ``value_and_grad`` maximizing detector-referred detected SNR. |

## `spin_dynamics.optimal_control.diffusion`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `gradient_moments(gradient, dt, *, encode_end, coherence_sign = None)` | q-vector moments of a (delivered) piecewise-constant gradient waveform. |
| function | `detected_echo_signal(coherences, weights, offsets_hz, *, n_acq, dt_acq)` | Acquire the coherently-summed echo over a readout window. |
| function | `detected_echo_snr(signal, dt_acq, *, rx_response = None, noise = 1.0)` | Matched-filter detected amplitude of an acquired echo over a noise floor. |
| function | `make_pgse_objective(*, drift_batch: Sequence[np.ndarray], hx_batch: Sequence[np.ndarray], hy_batch: Sequence[np.ndarray], hgrad_batch: Sequence[np.ndarray], psi0: np.ndarray, detection_operator: np.ndarray, weights: np.ndarray, offsets_hz: np.ndarray, dt: float, n_segments: int, amplitude_template: np.ndarray, gradient_mask: np.ndarray, coherence_sign: np.ndarray | None = None, encode_end_segment: int, q_target: float, acquisition_points: int, acquisition_dt: float, noise: float = 1.0, q_weight: float = 1.0, refocus_weight: float = 1.0, rf_response = None, gradient_response = None, rx_response = None, propagator: str = 'expm') -> Callable[[np.ndarray], tuple[float, np.ndarray]]` | Build the weighted PGSE ``value_and_grad`` closure. |

## `spin_dynamics.optimal_control.drivers`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `GrapeMultiStartResult` | Result of repeated random-phase-start GRAPE optimization. |
| function | `run_grape_multistart(model: Any, n_segments: int, *, num_starts: int = 24, seed: int | None = None, rng: np.random.Generator | None = None, initial_phases: np.ndarray | None = None, phase_bounds: tuple[float, float] = (-np.pi, np.pi), **grape_optimize_kwargs) -> GrapeMultiStartResult` | Run repeated random-phase-start GRAPE optimization and rank by fidelity. |

## `spin_dynamics.optimal_control.hamiltonians`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ControlHamiltonianModel` | Fixed operator set for a piecewise-constant-control Hamiltonian. |
| function | `gradient_control_operator(system: CoupledSpinSystem, *, control_indices: Iterable[int] | None = None) -> np.ndarray` | Return the base gradient-control operator ``TAU * Iz`` for a spin system. |
| function | `position_gradient_batch(h_grad_base: np.ndarray, positions: Iterable[float]) -> list[np.ndarray]` | Scale a base gradient operator by each ensemble member's position. |
| function | `coupled_spin_control_model(system: CoupledSpinSystem, *, control_indices: Iterable[int] | None = None, include_j_coupling: bool = True, include_gradient: bool = False) -> ControlHamiltonianModel` | Build a GRAPE control model for a scalar-coupled spin-1/2 system. |
| function | `nqr_site_control_model(site, *, rf_frequency_hz: float | None = None, b1_direction_pas: Sequence[float] = (1.0, 0.0, 0.0), b0_vector_tesla_pas: Sequence[float] | np.ndarray | None = None) -> ControlHamiltonianModel` | Build a GRAPE control model for a quadrupolar (NQR) site. |
| function | `nqr_fundamental_states(site, *, b0_vector_tesla_pas: Sequence[float] | np.ndarray | None = None) -> tuple[int, int]` | Return the ``(lower, upper)`` eigen-indices of the fundamental NQR line. |
| function | `nqr_powder_control_batch(site, orientations, *, rf_frequency_hz: float | None = None, b0_vector_tesla_pas: Sequence[float] | np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]` | Per-orientation ``(h_x, h_y)`` drive operators for a powder ensemble. |

## `spin_dynamics.optimal_control.multi_axis`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MultiAxisSLSEConfig` | Configuration for a tri-axial parametric SLSE optimization. |
| function | `control_bounds(config: MultiAxisSLSEConfig, *, amp_max: float = 4.0) -> list[tuple[float, float]]` | Per-parameter ``(lower, upper)`` bounds for one control vector. |
| function | `make_multi_axis_slse_objective(config: MultiAxisSLSEConfig)` | Return ``value_and_grad(x) -> (energy, grad)`` for the SLSE train energy. |
| function | `slse_train_amplitudes(config: MultiAxisSLSEConfig, x: np.ndarray) -> np.ndarray` | Return the complex powder echo-train (num_echoes,) for a control vector. |
| class | `MultiAxisSLSEResult` | Result of a multistart tri-axial SLSE optimization. |
| function | `simultaneous_seed(config: MultiAxisSLSEConfig, *, amp_fraction: float = 1.5, excite_len: float = 0.5, refocus_len: float = 1.0) -> np.ndarray` | A circular-style warm start: all coils simultaneous, quadrature phases. |
| function | `optimize_multi_axis_slse(config: MultiAxisSLSEConfig, *, n_starts: int = 8, seed: int = 0, amp_max: float = 4.0, include_simultaneous_seed: bool = True, options: dict | None = None) -> MultiAxisSLSEResult` | Multistart-optimize the tri-axial rectangular SLSE pulse parameters. |

## `spin_dynamics.optimal_control.objectives`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `state_transfer_fidelity(psi_final, psi_target)` | ``|<target|final>|^2`` for normalized state vectors. |
| function | `average_gate_fidelity(u_final, u_target)` | Nielsen average-gate fidelity ``(|Tr(U_target^H U_final)|^2 + d) / (d(d+1))``. |
| function | `su2_effective_axis(u)` | Extract the effective rotation (axis, angle) of a 2x2 SU(2) unitary. |
| function | `bloch_vector_to_ket(axis)` | Return the spin-1/2 ket aligned with a Bloch-sphere unit vector. |
| function | `robust_ensemble_fidelity(fidelity_per_case, *, reduction: Literal['mean', 'worst_case'] = 'mean')` | Reduce a ``(batch,)`` array of per-case fidelities to a scalar score. |
| function | `make_grape_objective(model, *, n_segments: int, dt, target, psi0 = None, mode: Literal['state_transfer', 'gate'] = 'state_transfer', optimize_amplitude: bool = False, fixed_amplitude: float | np.ndarray | None = None, optimize_gradient: bool = False, fixed_gradient: float | np.ndarray | None = None, gradient_operator_batch: Sequence[np.ndarray] | None = None, control_operator_batch: Sequence[tuple[np.ndarray, np.ndarray]] | None = None, hamiltonian_batch: Sequence[np.ndarray] | None = None, ensemble_reduction: Literal['mean', 'worst_case'] = 'mean', phase_smoothness_weight: float = 0.0, gradient_smoothness_weight: float = 0.0, propagator: Literal['eigh', 'expm'] = 'eigh', rf_response = None, gradient_response = None) -> Callable[[np.ndarray], tuple[float, np.ndarray]]` | Build a ``value_and_grad`` callable for a GRAPE fidelity objective. |

## `spin_dynamics.optimal_control.parameterization`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ControlBounds` | Per-parameter box constraints matching a flattened GRAPE control vector. |
| function | `phase_only_bounds(n_segments: int, *, phase_bound_rad: float = 4 * np.pi) -> ControlBounds` | Loose symmetric phase bounds. |
| function | `amplitude_phase_bounds(n_segments: int, *, amplitude_max_hz: float, phase_bound_rad: float = 4 * np.pi) -> ControlBounds` | Box bounds for joint amplitude+phase control. |
| function | `gradient_bounds(n_segments: int, *, gradient_max: float) -> ControlBounds` | Symmetric bipolar box bounds for a gradient waveform. |
| function | `assemble_control_bounds(n_segments: int, *, optimize_amplitude: bool = False, amplitude_max_hz: float | None = None, optimize_gradient: bool = False, gradient_max: float | None = None, phase_bound_rad: float = 4 * np.pi) -> ControlBounds` | Assemble box bounds for the full ``concat([amplitude?, phase, gradient?])`` layout. |
| function | `constant_gradient_seed(n_segments: int, *, gradient: float) -> np.ndarray` | Constant gradient waveform, e.g. the fixed-gradient slice-selective baseline. |
| function | `rectangular_seed_phase(n_segments: int, *, phase_rad: float = 0.0) -> np.ndarray` | Constant-phase baseline waveform. |
| function | `random_phase_starts(num_starts: int, n_segments: int, *, bounds: tuple[float, float] = (-np.pi, np.pi), seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray` | Reproducible random phase starts. |
| function | `export_to_pulse_shape(*, duration_seconds: np.ndarray | float, amplitude_hz: np.ndarray | float, phase_rad: np.ndarray)` | Export an optimized GRAPE waveform as an ``absolute_phase.PulseShape``. |

## `spin_dynamics.optimal_control.solvers`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `GrapeOptimizationResult` | Result of a GRAPE control-waveform optimization. |
| function | `grape_optimize(model: ControlHamiltonianModel, initial_phase: np.ndarray, *, dt: float | np.ndarray, target: np.ndarray, psi0: np.ndarray | None = None, mode: Literal['state_transfer', 'gate'] = 'state_transfer', optimize_amplitude: bool = False, fixed_amplitude: float | np.ndarray | None = None, initial_amplitude: np.ndarray | None = None, amplitude_max_hz: float | None = None, optimize_gradient: bool = False, fixed_gradient: float | np.ndarray | None = None, initial_gradient: np.ndarray | None = None, gradient_max: float | None = None, gradient_operator_batch: Sequence[np.ndarray] | None = None, control_operator_batch: Sequence[tuple[np.ndarray, np.ndarray]] | None = None, phase_bound_rad: float = 4 * np.pi, hamiltonian_batch: Sequence[np.ndarray] | None = None, ensemble_reduction: Literal['mean', 'worst_case'] = 'mean', phase_smoothness_weight: float = 0.0, gradient_smoothness_weight: float = 0.0, propagator: Literal['eigh', 'expm'] = 'eigh', rf_response = None, gradient_response = None, scipy_method: str = 'L-BFGS-B', scipy_options: dict[str, object] | None = None) -> GrapeOptimizationResult` | Optimize a piecewise-constant RF (and optionally gradient) waveform. |
| function | `grape_optimize_phase_only(model: ControlHamiltonianModel, initial_phase: np.ndarray, *, fixed_amplitude: float | np.ndarray, dt: float | np.ndarray, target: np.ndarray, psi0: np.ndarray | None = None, mode: Literal['state_transfer', 'gate'] = 'state_transfer', phase_bound_rad: float = 4 * np.pi, hamiltonian_batch: Sequence[np.ndarray] | None = None, ensemble_reduction: Literal['mean', 'worst_case'] = 'mean', phase_smoothness_weight: float = 0.0, scipy_method: str = 'L-BFGS-B', scipy_options: dict[str, object] | None = None) -> GrapeOptimizationResult` | Explicit phase-only GRAPE entry point. |

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
| class | `InverseExcitationValidation` | Quantitative cancellation evidence for one excitation/inverse pair. |
| function | `validate_inverse_excitation_pair(target_mrx: np.ndarray, inverse_mrx: np.ndarray, del_w: np.ndarray, *, target_snr: float | None = None, inverse_snr: float | None = None) -> InverseExcitationValidation` | Measure broadband cancellation, peak residual, phase, and SNR balance. |
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

## `spin_dynamics.prepolarization`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `PrepolarizedMagnetization` | Prepared longitudinal magnetization and sequence equilibrium arrays. |
| function | `longitudinal_recovery(initial_magnetization: float | Iterable[float] | np.ndarray, equilibrium_magnetization: float | Iterable[float] | np.ndarray, duration_seconds: float | Iterable[float] | np.ndarray, t1_seconds: float | Iterable[float] | np.ndarray) -> np.ndarray` | Return longitudinal magnetization after finite-time T1 recovery. |
| function | `field_ratio_equilibrium(polarizing_field_tesla: float | Iterable[float] | np.ndarray, detection_field_tesla: float, *, detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0) -> np.ndarray` | Return polarizing-field equilibrium in detection-field units. |
| function | `prepolarized_magnetization(polarizing_field_tesla: float | Iterable[float] | np.ndarray, detection_field_tesla: float, prepolarization_time_seconds: float | Iterable[float] | np.ndarray, t1_seconds: float | Iterable[float] | np.ndarray, *, initial_magnetization: float | Iterable[float] | np.ndarray = 0.0, detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0) -> np.ndarray` | Return prepared ``m0`` after relaxing in a polarizing field. |
| function | `prepolarized_state(polarizing_field_tesla: float | Iterable[float] | np.ndarray, detection_field_tesla: float, prepolarization_time_seconds: float | Iterable[float] | np.ndarray, t1_seconds: float | Iterable[float] | np.ndarray, *, initial_magnetization: float | Iterable[float] | np.ndarray = 0.0, detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0) -> PrepolarizedMagnetization` | Return prepared ``m0``, sequence ``mth``, and enhancement arrays. |
| function | `residence_time_seconds(path_length_meters: float | Iterable[float] | np.ndarray, speed_meters_per_second: float | Iterable[float] | np.ndarray) -> np.ndarray` | Return residence time for transport through a prepolarizing region. |
| function | `prepolarized_flow_state(polarizing_field_tesla: float | Iterable[float] | np.ndarray, detection_field_tesla: float, path_length_meters: float | Iterable[float] | np.ndarray, speed_meters_per_second: float | Iterable[float] | np.ndarray, t1_seconds: float | Iterable[float] | np.ndarray, *, initial_magnetization: float | Iterable[float] | np.ndarray = 0.0, detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0) -> PrepolarizedMagnetization` | Return a prepolarized state for flow through a finite polarizer. |
| function | `apply_prepolarization_to_parameters(params: Mapping[str, Any] | Any, prepared: PrepolarizedMagnetization) -> dict[str, Any]` | Return a shallow parameter copy with ``m0`` and ``mth`` replaced. |

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

## `spin_dynamics.pulse_diagnostics`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ProbePulseShapeDiagnostics` | Solved rotating-frame probe pulse shape and diagnostic metrics. |
| class | `ProbePulseShapeSweep` | Set of solved probe pulse shapes over absolute RF phase. |
| function | `solve_probe_pulse_shape(*, probe: str, absolute_phase_rad: float, pulse_kind: str = 'refocusing', numpts: int = 17, maxoffs: float = 10.0, q_value: float | None = None, mistuning_offset: float | None = None, rotating_phase_rad: float | None = None, pulse_duration_seconds: float | None = None, pulse_amplitude: float = 1.0, delay_seconds: float | None = None) -> ProbePulseShapeDiagnostics` | Solve one probe pulse shape for a requested absolute RF phase. |
| function | `solve_probe_pulse_shape_sweep(*, probe: str, absolute_phase_rad: Sequence[float] | np.ndarray, pulse_kind: str = 'refocusing', numpts: int = 17, maxoffs: float = 10.0, q_value: float | None = None, mistuning_offset: float | None = None, rotating_phase_rad: float | None = None, pulse_duration_seconds: float | None = None, pulse_amplitude: float = 1.0, delay_seconds: float | None = None) -> ProbePulseShapeSweep` | Solve a probe pulse-shape sweep over absolute RF phase. |
| function | `build_probe_pulse_shape_library(*, probe: str, absolute_phase_rad: Sequence[float] | np.ndarray, pulse_kind: str = 'refocusing', numpts: int = 17, maxoffs: float = 10.0, q_value: float | None = None, mistuning_offset: float | None = None, rotating_phase_rad: float | None = None, pulse_duration_seconds: float | None = None, pulse_amplitude: float = 1.0, delay_seconds: float | None = None) -> PulseShapeLibrary` | Build a probe-solved absolute-phase pulse-shape library. |

## `spin_dynamics.relaxation`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SingleSpinMatrices` | Dense angular-momentum matrices for one spin quantum number. |
| function | `spin_dimension(spin: float) -> int` | Return the Hilbert-space dimension for one spin. |
| function | `single_spin_matrices(spin: float) -> SingleSpinMatrices` | Return dense angular-momentum matrices for one spin. |
| function | `quadrupolar_tesseral_operators(spin: float) -> tuple[np.ndarray, ...]` | Return five Hermitian rank-2 spin-tensor components. |
| class | `BPPRelaxationRates` | Temperature-dependent BPP rates, times, and spectral densities. |
| class | `ArrheniusFit` | Log-linear Arrhenius fit for a positive measured quantity. |
| class | `BPPRelaxationModel` | Configurable BPP relaxation model with Arrhenius correlation time. |
| class | `PhenomenologicalRelaxationModel` | Phenomenological relaxation model in the Hamiltonian energy basis. |
| class | `RelaxationSuperoperator` | Protocol for relaxation models that build Hamiltonian-aware Liouvillians. |
| class | `MotionalAveragingModel` | Protocol for motional regimes used by microscopic relaxation models. |
| class | `DipolarRelaxationSource` | One fluctuating dipolar bath spin coupled to a target spin. |
| class | `RigidSolidMotionalAveraging` | Rigid-lattice dipolar fluctuations for solid-state relaxation. |
| class | `IsotropicLiquidMotionalAveraging` | Isotropic rotational averaging for liquid-state dipolar relaxation. |
| class | `VibrationalMotionalAveraging` | Exponentially damped motion modulated by one vibrational frequency. |
| class | `RedfieldEFGRelaxationModel` | Secular Redfield model for isotropic fluctuations of an EFG tensor. |
| class | `RedfieldDipolarRelaxationModel` | Secular Redfield relaxation model from fluctuating dipolar couplings. |
| class | `WallCollisionRelaxationModel` | Gas-wall collision relaxation from a stochastic spin map. |
| function | `spectral_density_lorentzian(angular_frequency_rad_per_s: float | Iterable[float] | np.ndarray, correlation_time_seconds: float | Iterable[float] | np.ndarray) -> np.ndarray` | Return the isotropic rotational spectral density ``2 tau/(1+w^2 tau^2)``. |
| function | `arrhenius_correlation_time(temperature_kelvin: float | Iterable[float] | np.ndarray, *, tau_ref_seconds: float, reference_temperature_kelvin: float = 298.15, activation_energy_j_per_mol: float = 0.0) -> np.ndarray` | Return ``tau_c(T)`` using an Arrhenius activation energy. |
| function | `fit_arrhenius_observable(temperature_kelvin: float | Iterable[float] | np.ndarray, values: float | Iterable[float] | np.ndarray, *, relative_uncertainty: float | Iterable[float] | np.ndarray | None = None) -> ArrheniusFit` | Fit ``ln(values)`` against ``1 / temperature``. |
| function | `stokes_einstein_debye_correlation_time(hydrodynamic_radius_m: float | Iterable[float] | np.ndarray, viscosity_pa_s: float | Iterable[float] | np.ndarray, temperature_kelvin: float | Iterable[float] | np.ndarray, *, slip_factor: float = 1.0) -> np.ndarray` | Return the rank-2 rotational correlation time from Stokes-Einstein-Debye. |
| function | `tau_c_from_t1_minimum(angular_frequency_rad_per_s: float, *, r1_coefficients: tuple[float, float, float] = (0.0, 1.0, 4.0)) -> float` | Return the correlation time at the BPP ``T1`` minimum for a Larmor freq. |
| function | `gas_mean_speed_m_per_s(temperature_kelvin: float | Iterable[float] | np.ndarray, mass_amu: float) -> np.ndarray` | Return Maxwell-Boltzmann mean molecular speed for a gas species. |
| function | `wall_collision_rate_per_second(surface_to_volume_per_m: float | Iterable[float] | np.ndarray, *, temperature_kelvin: float, mass_amu: float, accommodation_probability: float = 1.0) -> np.ndarray` | Return gas-wall encounter rate ``vbar * (S/V) / 4``. |
| function | `sphere_surface_to_volume_per_m(diameter_m: float | Iterable[float] | np.ndarray) -> np.ndarray` | Return ``S/V`` for a sphere from its diameter. |
| function | `cube_surface_to_volume_per_m(edge_m: float | Iterable[float] | np.ndarray) -> np.ndarray` | Return ``S/V`` for a cube from its edge length. |
| function | `cylinder_surface_to_volume_per_m(diameter_m: float | Iterable[float] | np.ndarray, *, aspect: float) -> np.ndarray` | Return ``S/V`` for a closed cylinder from diameter and ``length/diameter``. |
| function | `bpp_relaxation_rates(*, angular_frequency_rad_per_s: float | Iterable[float] | np.ndarray, correlation_time_seconds: float | Iterable[float] | np.ndarray, temperature_kelvin: float | Iterable[float] | np.ndarray | None = None, coupling_scale_per_second2: float = 1.0, r1_coefficients: tuple[float, float, float] = (0.0, 1.0, 4.0), r2_coefficients: tuple[float, float, float] = (1.5, 2.5, 1.0), baseline_r1_per_second: float = 0.0, baseline_r2_per_second: float = 0.0) -> BPPRelaxationRates` | Return BPP relaxation rates from ``J(0)``, ``J(w0)``, and ``J(2w0)``. |
| class | `PrepolarizedT1rhoExperiment` | Prepolarized spin-lock (``T1rho``) readout over a correlation-time grid. |
| function | `t1rho_relaxation_rate(spin_lock_angular_rad_per_s: float | Iterable[float] | np.ndarray, larmor_angular_rad_per_s: float, correlation_time_seconds: float | Iterable[float] | np.ndarray, *, coupling_scale_per_second2: float = 1.0, coefficients: tuple[float, float, float] = (1.5, 2.5, 1.0), lock_harmonic: float = 2.0, baseline_rate_per_second: float = 0.0) -> np.ndarray` | Return the on-resonance spin-lock relaxation rate ``R1rho``. |
| function | `simulate_prepolarized_t1rho(*, spin_lock_angular_rad_per_s: Iterable[float] | np.ndarray, larmor_angular_rad_per_s: float, correlation_time_seconds: Iterable[float] | np.ndarray, spin_lock_time_seconds: float, coupling_scale_per_second2: float, polarizing_field_tesla: float | Iterable[float] | np.ndarray, detection_field_tesla: float, prepolarization_time_seconds: float | Iterable[float] | np.ndarray, t1_seconds: float | Iterable[float] | np.ndarray, coefficients: tuple[float, float, float] = (1.5, 2.5, 1.0), lock_harmonic: float = 2.0, baseline_rate_per_second: float = 0.0, initial_magnetization: float | Iterable[float] | np.ndarray = 0.0, detection_equilibrium_magnetization: float | Iterable[float] | np.ndarray = 1.0) -> PrepolarizedT1rhoExperiment` | Simulate a prepolarized spin-lock (``T1rho``) readout. |
| class | `QuadrupolarRelaxationRates` | Quadrupolar ``R1``/``R2`` rates (and ``T1``/``T2``) for one nucleus. |
| function | `quadrupolar_relaxation_rates(quadrupole_coupling_hz: float | Iterable[float] | np.ndarray, spin: float, correlation_time_seconds: float | Iterable[float] | np.ndarray, *, asymmetry: float = 0.0, larmor_angular_rad_per_s: float = 0.0) -> QuadrupolarRelaxationRates` | Return quadrupolar ``R1``/``R2`` from the coupling constant and ``tau_c``. |
| function | `apply_relaxation_to_parameters(params: Mapping[str, Any] | Any, rates: BPPRelaxationRates) -> dict[str, Any]` | Return a shallow parameter copy with ``T1`` and ``T2`` replaced. |
| function | `dipolar_coupling_hz(distance_angstrom: float, *, gamma_a_hz_per_t: float = 3076600.0, gamma_b_hz_per_t: float = PROTON_GAMMA_HZ_PER_T) -> float` | Return the point-dipole coupling prefactor in Hz. |
| function | `dipolar_coupling_tensor(vector_angstrom: Sequence[float] | np.ndarray, *, coupling_hz: float) -> np.ndarray` | Return ``2*pi*d*(I - 3 n n^T)`` for a point dipolar coupling. |
| function | `matrix_exponential(matrix: np.ndarray, duration: float = 1.0) -> np.ndarray` | Return ``exp(matrix * duration)`` for a small dense matrix. |
| function | `liouville_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray` | Return the commutator Liouvillian for column-stacked density matrices. |
| function | `relaxation_superoperator(dimension: int, model: RelaxationModelLike, *, hamiltonian: np.ndarray | None = None) -> np.ndarray` | Return a trace-preserving relaxation superoperator. |
| function | `liouville_superoperator(hamiltonian: np.ndarray, model: RelaxationModelLike | None = None) -> np.ndarray` | Return Hamiltonian plus optional relaxation Liouvillian. |
| function | `propagate_density_liouville(density: np.ndarray, hamiltonian: np.ndarray, duration: float, *, relaxation: RelaxationModelLike | None = None) -> np.ndarray` | Propagate a density matrix with Hamiltonian and optional relaxation. |
| function | `cycle_superoperator(steps: tuple[tuple[np.ndarray, float], ...] | list[tuple[np.ndarray, float]], *, relaxation: RelaxationModelLike | None = None) -> np.ndarray` | Return the Liouville propagator for one repeated pulse-sequence cycle. |
| function | `effective_decay_time(eigenvalues: np.ndarray, cycle_duration_seconds: float, *, steady_tolerance: float = 1e-10) -> float` | Estimate the dominant non-steady decay time from cycle eigenvalues. |

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

## `spin_dynamics.sequences.compiler`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CompiledADC` | Receive samples on the absolute sequence timeline. |
| class | `CompiledSequence` | Piecewise-constant RF/gradient timeline plus exact ADC sample times. |
| function | `compile_sequence(sequence: SequenceIR, *, system_frequency_hz: float | None = None) -> CompiledSequence` | Compile an IR into piecewise-constant intervals. |
| function | `compiled_to_motion_steps(compiled: CompiledSequence, *, spatial_dimensions: int = 2, gradient_axes: tuple[int, ...] | None = None)` | Adapt compiled intervals to the existing moving-isochromat engine. |

## `spin_dynamics.sequences.ir`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `HardwareEffectsPolicy` | Declare whether execution must realize transmit and receive hardware. |
| class | `RFPulse` | Sampled complex RF envelope in hertz. |
| class | `GradientWaveform` | Sampled gradient waveform in hertz per meter. |
| class | `ADCEvent` | Uniform receive-sampling event. |
| class | `SequenceBlock` | Concurrent events with an explicit duration. |
| class | `SequenceIR` | A complete backend-neutral pulse sequence. |

## `spin_dynamics.sequences.motion`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MotionSequenceStep` | One interval in a moving-isochromat pulse sequence. |
| class | `MotionSequenceResult` | Result from a moving-isochromat sequence simulation. |
| function | `run_motion_sequence(ensemble: ParticleEnsemble, fields: MotionFields, steps: Sequence[MotionSequenceStep], *, velocity: Velocity = None, rng: np.random.Generator | None = None, t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0, boundary: Boundary = 'reflect', default_substeps: int = 1, detuning_waveform: DetuningWaveform = None) -> MotionSequenceResult` | Run a sequence while moving particles through sampled field maps. |
| function | `make_motion_cpmg_sequence(num_echoes: int, echo_spacing: float, *, excitation_duration: float, refocusing_duration: float, excitation_phase: float = np.pi / 2, refocusing_phase: float = 0.0, gradient: tuple[float, float] = (0.0, 0.0), substeps_per_interval: int = 1) -> tuple[MotionSequenceStep, ...]` | Build a rectangular-pulse CPMG sequence for moving isochromats. |
| function | `make_motion_udd_sequence(num_pulses: int, total_duration: float, *, excitation_duration: float, refocusing_duration: float, excitation_phase: float = np.pi / 2, refocusing_phase: float = 0.0, gradient: tuple[float, float] = (0.0, 0.0), substeps_per_interval: int = 1) -> tuple[MotionSequenceStep, ...]` | Build a rectangular-pulse UDD sequence for moving isochromats. |
| function | `run_motion_cpmg_sequence(ensemble: ParticleEnsemble, fields: MotionFields, *, num_echoes: int, echo_spacing: float, excitation_duration: float, refocusing_duration: float, gradient: tuple[float, float] = (0.0, 0.0), velocity: Velocity = None, rng: np.random.Generator | None = None, t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0, boundary: Boundary = 'reflect', substeps_per_interval: int = 1, detuning_waveform: DetuningWaveform = None) -> MotionSequenceResult` | Run a rectangular-pulse CPMG sequence with moving isochromats. |
| function | `run_motion_udd_sequence(ensemble: ParticleEnsemble, fields: MotionFields, *, num_pulses: int, total_duration: float, excitation_duration: float, refocusing_duration: float, gradient: tuple[float, float] = (0.0, 0.0), velocity: Velocity = None, rng: np.random.Generator | None = None, t1: float | Iterable[float] | np.ndarray = np.inf, t2: float | Iterable[float] | np.ndarray = np.inf, mth: float | Iterable[float] | np.ndarray = 1.0, boundary: Boundary = 'reflect', substeps_per_interval: int = 1, detuning_waveform: DetuningWaveform = None) -> MotionSequenceResult` | Run a rectangular-pulse UDD sequence with moving isochromats. |

## `spin_dynamics.sequences.plotting`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SequencePlotData` | Scaled arrays used by :func:`plot_sequence`. |
| function | `sequence_plot_data(sequence: SequenceIR | CompiledSequence, *, system_frequency_hz: float | None = None, time_unit: TimeUnit = 'auto', max_points: int = 50000) -> SequencePlotData` | Return plotting arrays without importing Matplotlib. |
| function | `plot_sequence(sequence: SequenceIR | CompiledSequence, *, system_frequency_hz: float | None = None, time_unit: TimeUnit = 'auto', show_blocks: bool = True, max_points: int = 50000, figure = None)` | Plot RF I/Q, gradient channels, and ADC on one aligned time axis. |

## `spin_dynamics.sequences.pulseq`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `PulseqFormatError` | Raised when a Pulseq file is malformed or uses an unsupported feature. |
| function | `read_pulseq(path: str | Path) -> SequenceIR` | Read a Pulseq ``.seq`` text file. |
| function | `write_pulseq(sequence: SequenceIR, path: str | Path, *, definitions: Mapping[str, Any] | None = None, create_signature: bool = True) -> None` | Write ``sequence`` as a core Pulseq 1.5.0 ``.seq`` text file. |
| function | `serialize_pulseq(sequence: SequenceIR, *, definitions: Mapping[str, Any] | None = None, create_signature: bool = True) -> str` | Serialize a raster-aligned sequence as core Pulseq 1.5.0 text. |
| function | `parse_pulseq(text: str, *, source_name: str = '<string>') -> SequenceIR` | Parse Pulseq 1.4/1.5 text into :class:`SequenceIR`. |

## `spin_dynamics.susceptibility`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CylindricalInclusion` | One infinitely long cylindrical inclusion perpendicular to the map plane. |
| class | `SusceptibilityField` | Internal off-resonance field from a susceptibility-contrast geometry. |
| class | `InternalGradientDistribution` | Pore-space distribution of the internal-gradient magnitude (T/m). |
| function | `susceptibility_offresonance_map(x_axis: Iterable[float] | np.ndarray, z_axis: Iterable[float] | np.ndarray, inclusions: Iterable[CylindricalInclusion], *, b0_tesla: float, susceptibility_difference: float = 0.0, gamma: float = PROTON_GAMMA, b0_in_plane_angle: float = 0.0, interior_fill: str = 'uniform') -> SusceptibilityField` | Return the internal off-resonance field for cylindrical inclusions. |
| function | `make_susceptibility_field_maps(field: SusceptibilityField, *, b1_tx_map: Iterable[float] | np.ndarray | None = None, b1_rx_map: Iterable[float] | np.ndarray | None = None) -> MotionFieldMaps2D` | Wrap a ``SusceptibilityField`` as motion field maps. |
| function | `internal_gradient_maps(field: SusceptibilityField) -> tuple[np.ndarray, np.ndarray, np.ndarray]` | Return ``(g_x, g_z, g_magnitude)`` internal-gradient maps in tesla/metre. |
| function | `internal_gradient_distribution(field: SusceptibilityField, *, weights: Iterable[float] | np.ndarray | None = None, restrict_to_pore_space: bool = True, bins: int = 64, range_max: float | None = None) -> InternalGradientDistribution` | Summarize the pore-space internal-gradient magnitude (T/m). |

## `spin_dynamics.workflows.acquisition`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `calc_macq_ideal_probe_relax4(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, num_workers: int | None = 1, rephase_max_time: float | None = None, rephase_safety_factor: float = 1.25, rephase_action: str = 'ignore', radiation_damping: RadiationDampingSpec | None = None) -> np.ndarray` | Calculate acquired spectra for an ideal-probe arbitrary sequence. |
| function | `calc_macq_tuned_probe_relax4(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, num_workers: int | None = 1, rephase_max_time: float | None = None, rephase_safety_factor: float = 1.25, rephase_action: str = 'ignore', radiation_damping: RadiationDampingSpec | None = None) -> tuple[np.ndarray, np.ndarray]` | Calculate finite acquisition for a tuned probe. |
| function | `calc_macq_untuned_probe_relax4(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, num_workers: int | None = 1, rephase_max_time: float | None = None, rephase_safety_factor: float = 1.25, rephase_action: str = 'ignore', radiation_damping: RadiationDampingSpec | None = None) -> tuple[np.ndarray, np.ndarray]` | Calculate finite acquisition for an untuned probe. |
| function | `calc_macq_matched_probe_relax4(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, num_workers: int | None = 1, rephase_max_time: float | None = None, rephase_safety_factor: float = 1.25, rephase_action: str = 'ignore', radiation_damping: RadiationDampingSpec | None = None) -> tuple[np.ndarray, np.ndarray]` | Calculate finite acquisition for a tuned-and-matched probe. |

## `spin_dynamics.workflows.bipolar`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `ToggleInterval` | One piecewise-constant interval in the toggling (coherence) frame. |
| class | `GradientMoments` | Toggling-frame diffusion moments for a gradient waveform. |
| class | `BipolarPGSTEResult` | Moment-model result for a stimulated-echo diffusion sequence. |
| function | `toggling_frame_moments(intervals: Iterable[ToggleInterval], *, gamma: float = GAMMA) -> GradientMoments` | Integrate the toggling-frame wavevector moments for a gradient waveform. |
| function | `cotts_thirteen_interval_intervals(*, gradient_amplitude: float, gradient_duration: float, half_echo_time: float, storage_time: float) -> tuple[ToggleInterval, ...]` | Build the 13-interval bipolar APGSTE toggling-frame intervals. |
| function | `monopolar_pgste_intervals(*, gradient_amplitude: float, gradient_duration: float, half_echo_time: float, storage_time: float) -> tuple[ToggleInterval, ...]` | Build an ordinary monopolar PGSTE for comparison. |
| function | `run_cotts_thirteen_interval_moment(*, gradient_amplitude: float = 0.05, gradient_duration: float = 0.002, half_echo_time: float = 0.006, storage_time: float = 0.04, diffusion_coefficient: float = 2.3e-09, background_gradient: float = 0.0, initial_signal: complex = 1.0 + 0j, gamma: float = GAMMA) -> BipolarPGSTEResult` | Run the 13-interval bipolar APGSTE moment model. |
| function | `run_monopolar_pgste_moment(*, gradient_amplitude: float = 0.05, gradient_duration: float = 0.002, half_echo_time: float = 0.006, storage_time: float = 0.04, diffusion_coefficient: float = 2.3e-09, background_gradient: float = 0.0, initial_signal: complex = 1.0 + 0j, gamma: float = GAMMA) -> BipolarPGSTEResult` | Run an ordinary monopolar PGSTE moment model for comparison. |
| class | `BipolarPGSTEWalkerResult` | Random-walker result for the bipolar 13-interval PGSTE. |
| function | `run_cotts_thirteen_interval_walkers(*, rho: Iterable[float] | np.ndarray | None = None, x_axis: Iterable[float] | np.ndarray | None = None, z_axis: Iterable[float] | np.ndarray | None = None, fields: MotionFieldMaps2D | None = None, gradient_amplitude: float = 0.05, gradient_duration: float = 0.002, half_echo_time: float = 0.006, storage_time: float = 0.04, diffusion_coefficient: float = 2.3e-09, gamma: float = GAMMA, gradient_axis: GradientAxis = 'x', background_gradient: float = 0.0, walkers_per_cell: int = 128, seed: int | None = None, jitter: bool = False, excitation_duration: float = 0.0001, refocusing_duration: float = 0.0002, spoiler_gradient: float = 0.2, spoiler_axis: GradientAxis = 'x', t1_seconds: float = np.inf, t2_seconds: float = np.inf, velocity: Velocity = None, boundary: Boundary = 'reflect', substeps_per_interval: int = 8) -> BipolarPGSTEWalkerResult` | Run the bipolar 13-interval PGSTE with explicit random-walker diffusion. |

## `spin_dynamics.workflows.cpmg`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CPMGResult` | Common result object for ideal and probe-aware CPMG workflows. |
| class | `CPMGTrainResult` | Finite ideal CPMG acquisition result. |
| function | `calc_masy_ideal(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any) -> np.ndarray` | Calculate ideal CPMG asymptotic magnetization. |
| function | `ideal_cpmg_train_max_time(pp0: Any, num_echoes: int) -> float` | Normalized total evolution time for the ideal finite CPMG train. |
| function | `probe_cpmg_train_max_time(pp0: Any, num_echoes: int) -> float` | Normalized total evolution time for the probe-aware finite CPMG trains. |
| function | `run_ideal_cpmg_train(numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, *, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', noise: NoiseSpec | Mapping[str, Any] | float | int | None = None, absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None) -> CPMGTrainResult` | Run a finite ideal CPMG echo train with relaxation. |
| function | `run_tuned_cpmg_train(numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, *, q_value: float | None = None, mistuning_offset: float | None = None, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', noise: NoiseSpec | Mapping[str, Any] | float | int | None = None, radiation_damping: RadiationDampingSpec | Mapping[str, Any] | None = None, absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None) -> CPMGTrainResult` | Run a finite tuned-probe CPMG echo train with relaxation. |
| function | `run_untuned_cpmg_train(numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, *, q_value: float | None = None, mistuning_offset: float | None = None, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', noise: NoiseSpec | Mapping[str, Any] | float | int | None = None, absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None) -> CPMGTrainResult` | Run a finite untuned-probe CPMG echo train with relaxation. |
| function | `run_matched_cpmg_train(numpts: int = 101, maxoffs: float = 10.0, num_echoes: int = 8, t1_seconds: float = 2.0, t2_seconds: float = 2.0, *, q_value: float | None = None, mistuning_offset: float | None = None, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', noise: NoiseSpec | Mapping[str, Any] | float | int | None = None, radiation_damping: RadiationDampingSpec | Mapping[str, Any] | None = None, absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None) -> CPMGTrainResult` | Run a finite matched-probe CPMG echo train with relaxation. |
| function | `run_ideal_cpmg(numpts: int = 101, maxoffs: float = 10.0, *, noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGResult` | Run the validated ideal no-probe CPMG workflow. |
| function | `run_tuned_cpmg(numpts: int = 101, maxoffs: float = 10.0, *, noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGResult` | Run the original/reference tuned-probe CPMG workflow. |
| function | `run_untuned_cpmg(numpts: int = 101, maxoffs: float = 10.0, *, noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGResult` | Run the original/reference untuned-probe CPMG workflow. |
| function | `run_matched_cpmg(numpts: int = 101, maxoffs: float = 10.0, *, noise: NoiseSpec | Mapping[str, Any] | float | int | None = None) -> CPMGResult` | Run the original/reference matched-probe CPMG workflow. |

## `spin_dynamics.workflows.cpmg_ir`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `CPMGIRTrainResult` | Finite CPMG-IR echo train over inversion delays. |
| class | `MatchedCPMGIRTrainResult` | Finite matched-probe CPMG-IR echo train over inversion delays. |
| function | `default_ir_tauvect(tauvect: Iterable[float] | np.ndarray | None = None) -> np.ndarray` | Resolve an optional inversion-delay list to the workflow's tau vector. |
| function | `cpmg_ir_train_max_time(pp0: Any, num_echoes: int, echo_spacing_seconds: float, tau: np.ndarray) -> float` | Normalized total evolution time for a finite CPMG-IR echo train. |
| function | `run_ideal_cpmg_ir_train(num_echoes: int = 10, echo_spacing_seconds: float = 0.0005, tauvect: Iterable[float] | np.ndarray | None = None, t1_seconds: float = 0.005, t2_seconds: float = 0.005, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1, tau_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGIRTrainResult` | Run a compact ideal-probe CPMG-IR finite echo train. |
| function | `run_tuned_cpmg_ir_train(num_echoes: int = 10, echo_spacing_seconds: float = 0.0005, tauvect: Iterable[float] | np.ndarray | None = None, t1_seconds: float = 0.005, t2_seconds: float = 0.005, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1, tau_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGIRTrainResult` | Run a compact tuned-probe CPMG-IR finite echo train. |
| function | `run_untuned_cpmg_ir_train(num_echoes: int = 10, echo_spacing_seconds: float = 0.0005, tauvect: Iterable[float] | np.ndarray | None = None, t1_seconds: float = 0.005, t2_seconds: float = 0.005, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1, tau_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> CPMGIRTrainResult` | Run a compact untuned-probe CPMG-IR finite echo train. |
| function | `run_matched_cpmg_ir_train(num_echoes: int = 10, echo_spacing_seconds: float = 0.0005, tauvect: Iterable[float] | np.ndarray | None = None, t1_seconds: float = 0.005, t2_seconds: float = 0.005, *, numpts: int = 101, maxoffs: float = 10.0, num_workers: int | None = 1, tau_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn') -> MatchedCPMGIRTrainResult` | Run a compact matched-probe CPMG-IR finite echo train. |

## `spin_dynamics.workflows.diffusion`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MatchedDiffusionCPMGResult` | Matched-probe diffusion-aware finite CPMG result. |
| class | `TunedDiffusionCPMGResult` | Tuned-probe diffusion-aware finite CPMG result. |
| class | `MatchedDiffusionQSweepResult` | Q sweep result for matched-probe diffusion-aware CPMG. |
| function | `check_matched_diffusion_q_stability(q_value: float, *, action: str = 'warn') -> bool` | Check the compact matched-diffusion Q validation boundary. |
| function | `calc_macq_matched_probe_relax_diffusion(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, apply_receiver: bool = True, num_workers: int | None = 1) -> tuple[np.ndarray, np.ndarray]` | Calculate diffusion-aware matched-probe finite acquisition. |
| function | `calc_macq_tuned_probe_relax_diffusion(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any, *, apply_receiver: bool = True, num_workers: int | None = 1) -> tuple[np.ndarray, np.ndarray]` | Calculate diffusion-aware tuned-probe finite acquisition. |
| function | `run_matched_diffusion_cpmg(num_echoes: int = 5, echo_spacing_seconds: float = 0.001, t1_seconds: float = 0.1, t2_seconds: float = 0.1, dz: float = 0.001, diffusion_time: float = 0.001, diffusion_coefficient: float | None = None, t90_seconds: float = 0.0001, q_value: float = 50.0, *, numpts: int = 101, apply_receiver: bool = False, num_workers: int | None = 1, q_stability_action: str = 'warn', auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None) -> MatchedDiffusionCPMGResult` | Run a compact matched-probe diffusion-aware CPMG train. |
| function | `run_tuned_diffusion_cpmg(num_echoes: int = 5, echo_spacing_seconds: float = 0.001, t1_seconds: float = 0.1, t2_seconds: float = 0.1, dz: float = 0.001, diffusion_time: float = 0.001, diffusion_coefficient: float | None = None, gradient: float = 1.0, t90_seconds: float = 0.0001, q_value: float | None = None, *, numpts: int = 101, apply_receiver: bool = True, num_workers: int | None = 1, auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None) -> TunedDiffusionCPMGResult` | Run a compact tuned-probe diffusion-aware CPMG train. |
| function | `run_matched_diffusion_q_sweep(q_values: Iterable[float] | np.ndarray | None = None, *, num_echoes: int = 5, echo_spacing_seconds: float = 0.001, diffusion_coefficient: float | None = None, diffusion_time: float = 0.001, dz: float = 0.001, t90_seconds: float = 0.0001, numpts: int = 101, num_workers: int | None = 1, sweep_workers: int | None = 1, q_stability_action: str = 'warn', auto_refine_grid: bool = False, rephase_safety_factor: float = 1.25, rephase_action: str = 'warn', absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None) -> MatchedDiffusionQSweepResult` | Sweep matched-probe Q for the compact diffusion CPMG workflow. |

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

## `spin_dynamics.workflows.imaging_3d`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `MultiSliceImagingResult` | Result of a 2-D multi-slice 3-D imaging simulation. |
| function | `run_multislice_imaging(rho, *, slice_gradient: float, slice_axis: int = 1, fov: tuple[float, float, float] = (0.02, 0.02, 0.02), slice_positions = None, b0_map = None, b1_tx_map = None, b1_rx_map = None, t1_map = None, t2_map = None, slice_duration: float = 0.001, flip_angle: float = np.pi / 2, time_bandwidth: float = 4.0, num_substeps: int = 48, window: Window = 'hamming', rephase: bool = True, rephase_fraction: float = 0.5, readout_time: float = 0.002, phase_time: float = 0.0004, refocusing_duration: float = 0.0001, substeps_per_interval: int = 1) -> MultiSliceImagingResult` | Acquire a 3-D volume by true 3-D slice-selective multi-slice imaging. |
| function | `run_multislice_imaging_separable(rho, *, slice_gradient: float, slice_axis: int = 1, fov: tuple[float, float, float] = (0.02, 0.02, 0.02), slice_positions = None, slice_duration: float = 0.001, flip_angle: float = np.pi / 2, time_bandwidth: float = 4.0, num_substeps: int = 48, window: Window = 'hamming', rephase: bool = True, rephase_fraction: float = 0.5, **in_plane_kwargs) -> MultiSliceImagingResult` | Fast separable multi-slice approximation (uniform in-plane field). |

## `spin_dynamics.workflows.imaging_frequency`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `FrequencyEncodedImagingResult` | Result of a frequency-encoded (spin-warp or RARE) imaging simulation. |
| class | `SliceSensitivityResult` | Real-space sensitive slice of an excitation in a non-uniform field. |
| function | `run_rare_imaging(rho, *, t1_map = None, t2_map = None, b0_map = None, b1_tx_map = None, b1_rx_map = None, fov: tuple[float, float] = (0.02, 0.02), echo_train_length: int = 8, phase_encode_order: PhaseEncodeOrder = 'linear', readout_time: float = 0.002, phase_time: float = 0.0004, excitation_duration: float = 5e-05, refocusing_duration: float = 0.0001, num_offsets: int = 1, offset_spread: float = 0.0, gamma: float = 267500000.0, substeps_per_interval: int = 1) -> FrequencyEncodedImagingResult` | Simulate a RARE / fast-spin-echo frequency-encoded image. |
| function | `run_spin_warp_imaging(rho, **kwargs) -> FrequencyEncodedImagingResult` | Simulate a spin-warp image (one spin echo per phase-encode line). |
| function | `imaging_slice_sensitivity(rho, *, center_frequency: float = 0.0, excitation_flip: float = np.pi / 2, excitation_duration: float = 0.0001, refocusing: bool = False, refocusing_flip: float = np.pi, refocusing_duration: float = 0.0002, t1_map = None, t2_map = None, b0_map = None, b1_tx_map = None, b1_rx_map = None, fov: tuple[float, float] = (0.02, 0.02)) -> SliceSensitivityResult` | Map the real-space sensitive slice of an excitation in a non-uniform field. |

## `spin_dynamics.workflows.imaging_types`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `IdealCPMGImagingResult` | Ideal-probe CPMG imaging result. |
| class | `ProbeCPMGImagingResult` | Probe-aware CPMG imaging result. |
| class | `ImagingEchoFitResult` | Voxel-wise mono-exponential fit of reconstructed echo magnitudes. |
| class | `ImagingNoiseStatistics` | Repeated-trial image-domain noise summary. |
| class | `ImagingFieldMaps` | Spatial sample and field maps for CPMG imaging workflows. |

## `spin_dynamics.workflows.pgse`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `PGSEMomentResult` | Deterministic PGSE result from gradient-moment diffusion attenuation. |
| class | `PGSEWalkerResult` | Random-walker PGSE result from explicit diffusive displacement. |
| class | `PGSTEWalkerResult` | Random-walker stimulated-echo PGSE (PGSTE) result. |
| class | `DDEWalkerResult` | Random-walker double diffusion encoding (DDE / double-PGSE) result. |
| class | `OGSEWalkerResult` | Random-walker oscillating-gradient spin-echo (OGSE) result. |
| function | `pgse_b_value(gradient_amplitude: float, gradient_duration: float, diffusion_time: float, *, gamma: float = 267500000.0) -> float` | Return the rectangular Stejskal-Tanner PGSE b-value. |
| function | `gradient_moment_b_value(segments: Iterable[tuple[float, float]], *, gamma: float = 267500000.0) -> float` | Integrate ``q(t)^2`` for a piecewise-constant effective gradient. |
| function | `run_pgse_moment(*, num_echoes: int = 1, gradient_amplitude: float = 0.05, gradient_duration: float = 0.002, diffusion_time: float = 0.02, diffusion_coefficient: float = 2.3e-09, t2_seconds: float = np.inf, first_echo_time_seconds: float | None = None, echo_spacing_seconds: float | None = None, initial_signal: complex = 1.0 + 0j, gamma: float = 267500000.0) -> PGSEMomentResult` | Run a fast ideal PGSE or PGSE-prepared CPMG calculation. |
| function | `run_pgse_walkers(*, rho: Iterable[float] | np.ndarray | None = None, x_axis: Iterable[float] | np.ndarray | None = None, z_axis: Iterable[float] | np.ndarray | None = None, fields: MotionFieldMaps2D | None = None, num_echoes: int = 1, gradient_amplitude: float = 0.05, gradient_duration: float = 0.002, diffusion_time: float = 0.02, diffusion_coefficient: float = 2.3e-09, gamma: float = 267500000.0, gradient_axis: PGSEDirection = 'x', walkers_per_cell: int = 128, seed: int | None = None, jitter: bool = False, excitation_duration: float = 0.0001, refocusing_duration: float = 0.0002, echo_spacing_seconds: float | None = None, t1_seconds: float = np.inf, t2_seconds: float = np.inf, velocity: Velocity = None, boundary: Boundary = 'reflect', substeps_per_interval: int = 8) -> PGSEWalkerResult` | Run PGSE with explicit random-walker diffusion. |
| function | `run_pgste_walkers(*, rho: Iterable[float] | np.ndarray | None = None, x_axis: Iterable[float] | np.ndarray | None = None, z_axis: Iterable[float] | np.ndarray | None = None, fields: MotionFieldMaps2D | None = None, gradient_amplitude: float = 0.05, gradient_duration: float = 0.002, diffusion_time: float = 0.02, diffusion_coefficient: float = 2.3e-09, gamma: float = 267500000.0, gradient_axis: PGSEAxis = 'x', walkers_per_cell: int = 128, seed: int | None = None, jitter: bool = False, excitation_duration: float = 0.0001, encode_delay: float = 0.0, spoiler_gradient: float = 0.2, spoiler_axis: PGSEAxis = 'x', t1_seconds: float = np.inf, t2_seconds: float = np.inf, velocity: Velocity = None, boundary: Boundary = 'reflect', substeps_per_interval: int = 8) -> PGSTEWalkerResult` | Run a pulsed-gradient stimulated-echo (PGSTE) walker simulation. |
| function | `run_dde_walkers(*, rho: Iterable[float] | np.ndarray | None = None, x_axis: Iterable[float] | np.ndarray | None = None, z_axis: Iterable[float] | np.ndarray | None = None, fields: MotionFieldMaps2D | None = None, gradient_amplitude: float = 0.05, gradient_duration: float = 0.002, diffusion_time: float = 0.02, mixing_time: float = 0.0, angle1: float = 0.0, angle2: float = 0.0, diffusion_coefficient: float = 2.3e-09, gamma: float = 267500000.0, walkers_per_cell: int = 128, seed: int | None = None, jitter: bool = False, excitation_duration: float = 0.0001, refocusing_duration: float = 0.0002, t1_seconds: float = np.inf, t2_seconds: float = np.inf, velocity: Velocity = None, boundary: Boundary = 'reflect', substeps_per_interval: int = 8) -> DDEWalkerResult` | Run a double diffusion encoding (DDE / double-PGSE) walker simulation. |
| function | `run_ogse_walkers(*, rho: Iterable[float] | np.ndarray | None = None, x_axis: Iterable[float] | np.ndarray | None = None, z_axis: Iterable[float] | np.ndarray | None = None, fields: MotionFieldMaps2D | None = None, gradient_amplitude: float = 0.05, oscillation_frequency: float = 100.0, num_periods: int = 2, samples_per_period: int = 16, diffusion_coefficient: float = 2.3e-09, gamma: float = 267500000.0, gradient_axis: PGSEAxis = 'x', walkers_per_cell: int = 128, seed: int | None = None, jitter: bool = False, excitation_duration: float = 0.0001, refocusing_duration: float = 0.0002, t1_seconds: float = np.inf, t2_seconds: float = np.inf, velocity: Velocity = None, boundary: Boundary = 'reflect', substeps_per_interval: int = 4) -> OGSEWalkerResult` | Run an oscillating-gradient spin-echo (OGSE) walker simulation. |
| function | `run_pgse(*, backend: PGSEBackend = 'moment', **kwargs) -> PGSEMomentResult | PGSEWalkerResult` | Dispatch to the moment or random-walker PGSE backend. |

## `spin_dynamics.workflows.qspace`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `QSpaceReconstructionResult` | Image-domain result reconstructed from a centered q-space grid. |
| class | `QSpacePhaseRetrievalResult` | Constrained pore-shape estimate from magnitude-only q-space samples. |
| class | `QSpaceShapeMetrics` | Shift/reflection-invariant quality metrics for a reconstructed pore. |
| class | `PGSEQSpaceWalkerResult` | Finite-pulse PGSE response sampled on a centered two-dimensional q grid. |
| function | `acquire_pgse_qspace_walkers(rho: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray, qx_axis: np.ndarray, qz_axis: np.ndarray, *, gradient_duration: float = 0.0005, diffusion_time: float = 0.02, diffusion_coefficient: float = 2.3e-09, gamma: float = 267500000.0, walkers_per_cell: int = 32, seed: int | None = None, jitter: bool = True, excitation_duration: float = 0.0001, refocusing_duration: float = 0.0002, t1_seconds: float = np.inf, t2_seconds: float = np.inf, velocity: Velocity = None, fields: MotionFieldMaps2D | None = None, boundary: Boundary = 'reflect', substeps_per_interval: int = 8) -> PGSEQSpaceWalkerResult` | Acquire a finite-pulse restricted-diffusion response on a q-space grid. |
| function | `qspace_axes_from_real_space(x_axis: np.ndarray, z_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]` | Return centered angular q axes compatible with a real-space grid. |
| function | `real_space_axes_from_qspace(qx_axis: np.ndarray, qz_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]` | Return centered real-space axes for a uniformly sampled q-space grid. |
| function | `pore_form_factor_from_density(density: np.ndarray, *, normalize: bool = True) -> np.ndarray` | Return the centered complex pore form factor of a 2D density map. |
| function | `qspace_sampling_mask(qx_axis: np.ndarray, qz_axis: np.ndarray, *, qmax_fraction: float = 1.0, missing_fraction: float = 0.0, seed: int | None = None) -> np.ndarray` | Build a reproducible radial-window and random-dropout sampling mask. |
| function | `add_qspace_intensity_noise(intensity: np.ndarray, *, snr: float, seed: int | None = None, sample_mask: np.ndarray | None = None) -> tuple[np.ndarray, float]` | Add Gaussian intensity noise and return the clipped data and sigma. |
| function | `threshold_qspace_intensity(intensity: np.ndarray, *, noise_sigma: float, threshold_sigma: float = 2.0, sample_mask: np.ndarray | None = None) -> np.ndarray` | Suppress a known additive intensity-noise floor before phase retrieval. |
| function | `qspace_shape_metrics(estimate: np.ndarray, reference: np.ndarray, *, threshold: float = 0.2) -> QSpaceShapeMetrics` | Compare pore shapes modulo translation and axis reflections. |
| function | `reconstruct_qspace_image(response: np.ndarray, qx_axis: np.ndarray, qz_axis: np.ndarray, *, data_kind: QSpaceDataKind = 'complex', clip_negative: bool = False, normalize: bool = True) -> QSpaceReconstructionResult` | Reconstruct an image or autocorrelation from centered q-space samples. |
| function | `phase_retrieve_qspace_magnitude(magnitude: np.ndarray, qx_axis: np.ndarray, qz_axis: np.ndarray, *, support: np.ndarray | None = None, iterations: int = 300, beta: float = 0.8, seed: int | None = None, input_is_intensity: bool = False, er_iterations: int = 40, sample_mask: np.ndarray | None = None) -> QSpacePhaseRetrievalResult` | Estimate a non-negative pore image from magnitude-only q-space data. |

## `spin_dynamics.workflows.single_sided`

| Kind | Name | Summary |
| --- | --- | --- |
| class | `SampleLayer` | A depth layer of the sample (depth measured along the through-plane axis). |
| class | `LayeredSample` | A stack of :class:`SampleLayer` ordered by depth. |
| class | `MouseCPMGResult` | One simulated CPMG measurement at a fixed excitation frequency. |
| class | `MouseDepthProfileResult` | A depth profile assembled from per-frequency CPMG measurements. |
| class | `MouseFieldSource` | A B0/B1 field source for the single-sided (NMR-MOUSE) workflow. |
| class | `AnalyticMouseField` | Analytic bar-magnet field source with an optional image yoke (the default). |
| class | `SolvedMouseField` | Field source backed by a solved 3-D :class:`ScalarPotentialSolution`. |
| function | `resonant_depth(source, frequency_hz: float, *, yoke_y: float | None = None, depth_range: tuple[float, float] | None = None, gamma: float = GAMMA_PROTON) -> float` | Return the on-axis depth where the proton Larmor frequency equals ``frequency_hz``. |
| function | `simulate_mouse_cpmg(source, sample: LayeredSample, frequency_hz: float, *, yoke_y: float | None = None, echo_time: float = 0.0002, num_echoes: int = 64, excitation_duration: float = 1e-05, depth_halfwidth: float = 0.0005, lateral_halfwidth: float = 0.003, n_depth: int = 121, n_lateral: int = 3, walkers_per_cell: int = 12, substeps_per_interval: int = 4, coil_segments: Sequence | None = None, diffusion_scale: float = 1.0, depth_range: tuple[float, float] | None = None, gamma: float = GAMMA_PROTON, seed: int = 0) -> MouseCPMGResult` | Simulate one CPMG measurement at ``frequency_hz`` in the real magnet field. |
| class | `MouseDiffusionResult` | Diffusion measured at one depth from the diffusion-on/off echo ratio. |
| function | `measure_diffusion_at_depth(source, sample: LayeredSample, frequency_hz: float, *, echo_time: float = 0.00012, num_echoes: int = 40, n_seeds: int = 4, min_ratio: float = 0.1, gamma: float = GAMMA_PROTON, **cpmg_kwargs) -> MouseDiffusionResult` | Measure D at the slice by the diffusion-on / diffusion-off echo ratio. |
| function | `mouse_depth_profile(source, sample: LayeredSample, frequencies_hz: Sequence[float], *, yoke_y: float | None = None, **cpmg_kwargs) -> MouseDepthProfileResult` | Profile a sample in depth by sweeping the excitation frequency. |

## `spin_dynamics.workflows.slice_selective`

| Kind | Name | Summary |
| --- | --- | --- |
| function | `make_slice_selective_excitation(*, duration: float, slice_gradient: float, flip_angle: float = np.pi / 2, slice_axis: int = 0, ndim: int = 2, time_bandwidth: float = 4.0, num_substeps: int = 48, phase: float = 0.0, window: Window = 'hamming', rephase: bool = True, rephase_fraction: float = 0.5) -> tuple[MotionSequenceStep, ...]` | Build a slice-selective excitation as a tuple of motion-sequence steps. |
| function | `slice_excitation_weights(positions: np.ndarray, *, duration: float, slice_gradient: float, center: float = 0.0, flip_angle: float = np.pi / 2, time_bandwidth: float = 4.0, num_substeps: int = 48, window: Window = 'hamming', rephase: bool = True, rephase_fraction: float = 0.5) -> np.ndarray` | Return the complex transverse weight a slice pulse imprints at ``positions``. |
| function | `slice_profile_table(*, slice_gradient: float, off_resonance_max: float, duration: float, flip_angle: float = np.pi / 2, num: int = 1201, time_bandwidth: float = 4.0, num_substeps: int = 48, window: Window = 'hamming', rephase: bool = True, rephase_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]` | Tabulate the excited transverse magnetization versus off-resonance. |
| class | `SliceProfileResult` | Through-slice magnetization profile of a slice-selective pulse. |
| function | `simulate_slice_profile(*, duration: float, slice_gradient: float, flip_angle: float = np.pi / 2, time_bandwidth: float = 4.0, num_substeps: int = 48, window: Window = 'hamming', rephase: bool = True, rephase_fraction: float = 0.5, extent: float | None = None, num_positions: int = 201) -> SliceProfileResult` | Excite a uniform line of spins and return the through-slice profile. |

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
