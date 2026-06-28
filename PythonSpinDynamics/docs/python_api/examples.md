# Examples

Examples live in `examples/`. They can be run from `PythonSpinDynamics` or from
inside `PythonSpinDynamics/examples`; each script adds the local `src` directory
to `sys.path` when the package has not been installed yet.

## Ideal CPMG

```powershell
python examples\ideal_cpmg.py --numpts 101
```

## Ideal FID

```powershell
python examples\ideal_fid.py --numpts 101
```

## Finite Ideal CPMG Train

```powershell
python examples\ideal_cpmg_train.py --numpts 101 --num-echoes 8
```

## UDD vs CPMG Filter Functions

This plotting example compares ideal instantaneous-pulse UDD and CPMG timing
for the same number of refocusing pulses. It shows pulse placement, sinusoidal
detuning response, and cumulative pickup from low-frequency fluctuations.

```powershell
python examples\plot_udd_cpmg_filter.py --pulses 8 --output results\udd_cpmg_filter.png
```

## Ideal Time-Varying CPMG

```powershell
python examples\ideal_time_varying_cpmg.py --numpts 101 --num-echoes 16
```

## Compare CPMG and FID

```powershell
python examples\compare_cpmg_fid.py --numpts 101
```

Optionally save arrays:

```powershell
python examples\compare_cpmg_fid.py --numpts 101 --save-npz results\ideal_compare.npz
```

## Export Validation Arrays

```powershell
python examples\export_validation_arrays.py results\validation_arrays.npz --numpts 101
```

## Plot Ideal Workflows

This example requires Matplotlib.

```powershell
python examples\plot_ideal_workflows.py --numpts 201 --output results\ideal_workflows.png
```

The plotting example narrows the FID offset range by default so the FID panel is
readable. To reproduce the raw MATLAB default FID range, use:

```powershell
python examples\plot_ideal_workflows.py --fid-maxoffs 10 --raw-fid-scale
```

## Plot Ideal Imaging

This example uses the `flower.png` phantom from the MATLAB reference tree. It
uses Matplotlib when available and falls back to Pillow for writing the output
PNG.

```powershell
python examples\plot_ideal_imaging.py --pixels 6 --ny 7 --output results\ideal_imaging.png
```

Use `--image-mode single`, `echo-sum`, `fit-rho`, or `fit-t2` to choose how
the echo stack is converted into the displayed image. The fitting modes require
at least two echoes.
Use `--t1-encoded --inversion-time 5e-4` with the ideal probe path to add an
inversion-recovery preparation before phase encoding and CPMG.

## Plot Custom Imaging Fields

This example builds a small synthetic phantom with custom B0, transmit-B1, and
receive-B1 maps, then plots the input maps, k-space, and reconstruction. It
requires Matplotlib.

```powershell
python examples\plot_custom_imaging_fields.py --pixels 8 --ny 7 --output results\custom_imaging_fields.png
```

The same `--image-mode` option is available here for comparing selected-echo,
echo-summed, fitted-rho, and fitted-T2 displays under custom field maps.
The same ideal-probe `--t1-encoded` option can be combined with those display
modes for synthetic T1 contrast examples.

## RARE Imaging (Frequency-Encoded)

This example contrasts the two frequency-encoded workflows, `run_spin_warp_imaging`
and `run_rare_imaging`, on a synthetic two-T2 phantom. Spin-warp uses one spin
echo per phase-encode line (the reference image); RARE reads a different k-space
line on each echo of a CPMG train, so it needs far fewer excitations at the cost
of a T2 weighting across the phase-encode lines that blurs the image. The four
panels show the phantom, the spin-warp reference, a single-shot RARE
reconstruction, and the k-space T2 weighting that drives the blurring.

```powershell
python examples\plot_rare_imaging.py --pixels 32 --output results\rare_imaging.png
```

Use `--echo-train-length` to set the RARE acceleration (default: a full single
shot), `--readout-time` for the frequency-encode duration, and `--t2-long` /
`--t2-short` for the phantom relaxation contrast. Only Matplotlib is required.

## Imaging Inhomogeneity (Frequency-Encoded)

This example evaluates how field inhomogeneity affects a frequency-encoded
(spin-warp) image, since `run_spin_warp_imaging` and `run_rare_imaging` accept
the same B0/B1 maps as the phase-encoded workflows. One phantom is imaged four
ways: uniform fields (reference), a linear B0 gradient (geometric distortion
along the readout axis), a sub-voxel B0 spread (`num_offsets` / `offset_spread`,
the T2* point-spread function that blurs along readout), and a transmit-B1
shading.

```powershell
python examples\plot_imaging_inhomogeneity.py --pixels 32 --output results\imaging_inhomogeneity.png
```

Use `--b0-gradient-hz`, `--b0-spread-hz` / `--num-offsets`, and `--b1-min` to set
each effect, and `--readout-time` (a shorter readout resists B0 distortion). Only
Matplotlib is required.

## Sensitive Slice in a Non-Uniform Field

A practical issue in inhomogeneous-field imaging is that the excited slice is
neither flat nor uniform in real space. This example uses
`imaging_slice_sensitivity` on a single-sided-like field (B0 rising with depth
and curving across the probe, plus a surface-coil B1) to map the sensitive
slice. The four panels show the B0 field with curved iso-frequency contours, the
sensitive slice at one excitation frequency (a curved band shaded by B1),
several slices at different frequencies (curved bands at different depths), and
the slice center and peak intensity versus position -- making explicit that the
slice is curved (not flat) and shaded (not uniform).

```powershell
python examples\plot_sensitive_slice.py --pixels 61 --output results\sensitive_slice.png
```

Use `--b0-depth-hz` / `--b0-curvature-hz` to shape the field, and
`--excitation-duration` to set the slice thickness (bandwidth ~ 1/duration). Only
Matplotlib is required.

## Multi-Slice 3D Imaging in a Halbach Field

This example acquires a structured 3D phantom (spheres at different depths plus an
in-plane bar) with the true-3D slice-selective workflow `run_multislice_imaging`
in a *mild* Halbach `(B0, B1)` field -- a smooth B0 saddle across the bore and a
gentle B1 falloff. Because the slice is selected by total off-resonance, the B0
saddle gently curves and shifts the slices and distorts the readout. The figure
shows a few acquired slices next to the ground-truth slices, the Halbach B0 and
B1 maps, and a 3D voxel rendering of the reconstructed volume.

```powershell
python examples\plot_multislice_halbach_imaging.py --pixels 16 --slices 5 --output results\multislice_halbach.png
```

Use `--b0-inhomogeneity-hz` and `--b1-inhomogeneity` to set the field
variation, and `--slice-thickness-voxels` to set the slice gradient. The engine
path costs roughly `slices x pixels` full-ensemble sequence runs, so keep the
grid modest; `run_multislice_imaging_separable` is the fast flat-slice
approximation for larger volumes. Only Matplotlib is required.

## Finite Halbach Dipole Field

Samples the 3D field of the lowest-order finite Halbach dipole: four transverse,
diametrically magnetized cylindrical or square rods. The example plots the
mid-plane Larmor map, vector field, uniformity, axial finite-length falloff, and
field-gradient map. It is a direct visualization of
`spin_dynamics.fields.magnetostatics.sample_halbach_dipole_field`.

```powershell
python examples\plot_halbach_dipole_field.py --rod-shape square --output results\halbach_dipole.png
```

Use `--center-radius`, `--rod-radius` / `--rod-width`, `--length`, and
`--remanence` for the magnet, and `--n-cross` / `--n-length` for the point-dipole
cubature resolution. Only Matplotlib is required.

## NMR-MOUSE Field Maps and Sensitive Slice

Builds the field of a single-sided NMR-MOUSE from first principles
(`spin_dynamics.fields.magnetostatics`): the analytic B0 of two antiparallel
permanent-magnet bars plus a soft-iron return yoke (method of images), and the
Biot-Savart B1 of a surface coil. It shows the `|B0|`/Larmor map with
iso-frequency contours (each a depth slice), the static gradient, the transverse
B1, and the depth-resolved sensitive slice from `imaging_slice_sensitivity`. The
field reproduces the device's regime: `~5-23 MHz` Larmor, `G ~ 7-28 T/m`.

```powershell
python examples\plot_nmr_mouse_fields.py --pixels 121 --output results\nmr_mouse_fields.png
```

Use `--gap`, `--magnet-mm`, and `--remanence` for the magnet, `--coil-radius` for
the coil, and `--excitation-duration` to set the slice bandwidth. Only Matplotlib
is required.

## NMR-MOUSE Depth-Resolved Relaxation and Diffusion

Drives the moving-isochromat engine with the magnet's real B0 field: walkers
seeded around the frequency-selected sensitive slice diffuse through the actual
static gradient, so the slice selection, the echo train, and its diffusion
attenuation all emerge from spins moving through the spatially structured field.
On a layered phantom (water / gap / gel) the figure shows the depth profile of
CPMG signal (the gap reads as a hole), the apparent T2 vs depth (diffusion-
shortened where the gradient is strong), the diffusion coefficient vs depth from
the diffusion-on/off ratio method (resolving the fast and slow layers), and the
single-sided echo trains.

```powershell
python examples\plot_nmr_mouse_depth_profile.py --num-depths 13 --num-d-depths 5 --seeds 4 --output results\nmr_mouse_depth.png
```

This is a moving-walker Monte-Carlo (stochastic; ~1-2 minutes). Use `--seeds` and
`--walkers` to trade speed for noise. Only Matplotlib is required.

## Tuned-Probe CPMG

```powershell
python examples\tuned_probe_cpmg.py --numpts 101
```

## Probe CPMG Comparison

```powershell
python examples\probe_cpmg_compare.py --numpts 101
```

## Probe Parameter Sweeps

```powershell
python examples\probe_parameter_sweeps.py --numpts 101
```

## Matched CPMG-IR Finite Train

```powershell
python examples\matched_cpmg_ir_train.py --numpts 21 --num-echoes 4 --num-tau 4
```

## Finite Probe Train Sweeps

```powershell
python examples\finite_probe_train_sweeps.py --numpts 21 --num-echoes 3
```

## Absolute-Phase CPMG Examples

These plotting examples reproduce the pulse-shape simulation strategy from
Mandal 2015 in compact form. They use the finite CPMG `absolute_phase` workflow
to solve the tuned, untuned, or matched probe waveform for each refocusing
pulse's absolute RF phase, discretize the rotating-frame shape into small pulse
segments, and compare the matched-filter echo amplitude with a synchronized
reference.

```powershell
python examples\plot_mandal2015_phase_step_sweep.py --output results\mandal2015_phase_step_sweep.png
python examples\plot_mandal2015_echo_modulation.py --output results\mandal2015_echo_modulation.png
python examples\plot_mandal2015_pulse_shapes.py --output results\mandal2015_pulse_shapes.png
```

The two finite-train plots accept `--phase-bins N` to quantize refocusing
absolute phases and reuse the corresponding pulse-shape solves. The result
metadata still stores the scheduled phase for every echo plus the quantized
matrix phase and exported refocusing pulse-shape library.
By default, the finite-train plots also enable `auto_refine_grid` and use
`--rephase-action raise`, so a too-coarse fixed isochromat grid is corrected or
reported instead of becoming an artificial echo modulation. Use
`--no-auto-refine-grid --rephase-action warn` only for deliberate diagnostics.
The pulse-shape plotting example uses the public
`spin_dynamics.pulse_diagnostics.solve_probe_pulse_shape` API, which can also
be used directly in notebooks or debugging scripts.

## Matched Diffusion CPMG

```powershell
python examples\matched_diffusion_cpmg.py --numpts 21 --num-echoes 3
```

Add `--phase-step 0.25 --phase-bins 16` to run the same compact diffusion
case with absolute-phase-resolved matched-probe pulse shapes.

```powershell
python examples\plot_diffusion_absolute_phase_compare.py --output results\diffusion_absolute_phase_compare.png
python examples\plot_tuned_diffusion_absolute_phase_compare.py --output results\tuned_diffusion_absolute_phase_compare.png
```

These plots compare four CPMG echo decays: synchronized RF without diffusion,
diffusion only, absolute-phase advance only, and the combined case.
The diffusion examples use a narrow default `--dz-um` and auto-refine the
offset grid so the plotted echo decays are not dominated by discrete-grid
rephasing. Each script prints the effective number of offsets after refinement.
The matched-probe comparison also prints the matched-probe absolute-phase
residual; the current matched pulse-shape solver is often nearly
phase-invariant. The tuned-probe comparison is the higher-contrast example for
probe-solved absolute-phase sensitivity combined with diffusion.

## PGSE D-T2 Inverse Laplace

This plotting example requires SciPy and Matplotlib for the non-negative
2D inverse Laplace transform. It builds a PGSE b-axis with the new moment
backend, simulates a PGSE-prepared CPMG echo matrix for a two-component
D-T2 distribution, adds Gaussian noise, and recovers the map with
`invert_d_t2`.

```powershell
python examples\plot_pgse_d_t2.py --output results\pgse_d_t2.png
```

Use `--snr`, `--regularization`, and `--regularization-order` to inspect
conditioning. If SciPy is absent the script falls back to an unconstrained
least-squares preview, but the intended production path is the default NNLS
solve from the `opt` extra.

## PGSE Restricted Diffusion in a Pore

This plotting example uses the stochastic random-walker PGSE backend
(`run_pgse_walkers`) to model diffusion confined to a slab pore with reflecting
walls. Passing explicit field maps whose bounds coincide with the pore makes the
walkers bounce off the walls, which the analytical moment backend cannot
represent. The three panels reproduce the canonical restricted-diffusion
signatures: the echo attenuation `E(b)` bends below the free `exp(-b D)` line as
the pore shrinks, the apparent diffusion coefficient `D_app = -ln(E)/b` falls
with increasing diffusion time, and the walker displacement histogram shows the
Gaussian free spread clamped to the pore width.

```powershell
python examples\plot_pgse_restricted_diffusion.py --output results\pgse_restricted.png
```

Use `--diffusion-time` to set the b-sweep diffusion time, and
`--walkers-per-cell` / `--substeps` to trade runtime for smoother, more accurate
stochastic curves. Only Matplotlib is required; SciPy is not used here.

## PGSE Diffusive Diffraction in a Circular Pore

This example extends restricted diffusion to a genuinely two-dimensional
geometry: walkers confined to a disc by a reflecting circular wall, supplied by
the new `spin_dynamics.motion.make_circular_reflector` callback (the motion
engine now accepts a callable boundary in addition to the rectangular
`reflect`/`periodic`/`clip` modes). In the narrow-pulse, long-mixing q-space
regime the normalized echo stops decaying monotonically and instead shows
*diffusive diffraction* minima at the zeros of the disc structure factor
`|2 J1(q a)/(q a)|^2`, i.e. at `q_ang a = 3.83, 7.02, ...`. The x-axis uses the
Callaghan reciprocal-space convention `q = gamma G delta / (2*pi)`; the angular
alternative is `q_ang = gamma G delta` (note the factor-of-`2*pi` ambiguity
between the two conventions in the literature).

```powershell
python examples\plot_pgse_circular_pore_diffraction.py --output results\pgse_diffraction.png
```

This is the heaviest example in the suite (the q sweep re-runs the walker
ensemble per gradient); the defaults take a couple of minutes. Use `--num-q`,
`--grid`, `--walkers-per-cell`, and `--substeps` to trade runtime for sharper,
deeper fringe minima, and `--pore-radius` / `--diffusion-time` to move the
diffraction features. SciPy is optional and only used to overlay the Bessel
form-factor theory.

## q-Space Pore Imaging

This example is the inverse counterpart to the circular-pore diffraction plot.
It starts from a two-dimensional q-space diffraction response in the ideal
short-gradient-pulse, fully sampled-pore limit and reconstructs real-space
structure. A complex form factor can be inverted directly; the usual
magnitude-squared response `|F(q)|^2` instead gives a pore autocorrelation, so
the example then applies finite-support, non-negative phase retrieval to
estimate the pore image itself. The default figure includes both an ideal
response and a finite-SNR intensity measurement with additive q-space noise.

```powershell
python examples\plot_pgse_qspace_pore_imaging.py --output results\qspace_pore_imaging.png
```

Use `--snr`, `--support-factor`, `--iterations`, and `--seed` to explore noise
sensitivity and the magnitude-only phase-retrieval ambiguity. Only NumPy and
Matplotlib are needed.

## PGSTE Stimulated-Echo Diffusion

This example uses the stimulated-echo backend (`run_pgste_walkers`) to show why
PGSTE is preferred for slow diffusion and short-`T2` samples. The encoding is
split across three 90-degree pulses so the magnetization is stored along the
longitudinal axis during the (long) diffusion delay, decaying with `T1` instead
of `T2`. The left panel confirms the stimulated echo follows Stejskal-Tanner
`exp(-b D)` at half the spin-echo amplitude; the right panel sweeps the
diffusion time at fixed `b` with a short `T2` and long `T1`, where the PGSE spin
echo collapses while the stimulated echo survives.

```powershell
python examples\plot_pgste_stimulated_echo.py --output results\pgste.png
```

Use `--t1` / `--t2` to set the relaxation contrast, `--fixed-b` for the
diffusion-time panel weighting, and `--walkers-per-cell` / `--substeps` for
convergence. Only Matplotlib is required.

## Phase-Cycled Stimulated Echo

This example uses the public `PhaseCycle` abstraction directly on a deterministic
Bloch ensemble in an inhomogeneous static field. It contrasts a single uncycled
three-90 scan, a simple two-step readout cycle, and the full three-pulse
stimulated-echo phase table. Storage relaxation creates a prompt last-pulse FID
from recovered longitudinal magnetization; the two-step cycle cannot separate it
from the stimulated echo, while the full receiver signature
`-phi1 + phi2 + phi3` keeps the refocused stimulated echo at `tau`.

```powershell
python examples\plot_phase_cycled_stimulated_echo.py --output results\phase_cycled_ste.png
```

Use `--offset-span-hz` / `--offset-sigma-hz` to shape the B0 distribution,
`--tau-ms` / `--storage-ms` to set the sequence timing, and
`--t1-ms` / `--t2-ms` to control storage relaxation. Only Matplotlib is required.

## Double Diffusion Encoding in an Elliptical Pore

This example uses the double-PGSE backend (`run_dde_walkers`) with an elliptical
reflecting pore (`make_elliptical_reflector`) to demonstrate the unique
capability of double diffusion encoding (DDE): detecting *microscopic*
anisotropy. Two gradient pairs separated by a mixing time encode displacement
along directions at a relative angle `psi`; the echo then carries a `cos 2*psi`
modulation whose amplitude reports the local anisotropy. Because it depends only
on the relative angle, it survives powder averaging, so it reveals shape
anisotropy even in an orientationally disordered sample where single-PGSE
diffusion looks isotropic. The example contrasts the ellipse with an equal-area
circle: the `cos 2*psi` term is present for the ellipse (single orientation and
powder-averaged) and absent for the circle.

```powershell
python examples\plot_pgse_double_encoding_elliptical_pore.py --output results\dde.png
```

The `cos 2*psi` term is a higher-order (`q^4`) effect, so the default uses strong
diffusion weighting (`--gradient-amplitude 1.0`); the powder panel re-runs the
walker ensemble at several orientations, so the defaults take a couple of
minutes. Use `--semi-major` / `--semi-minor` to set the eccentricity,
`--mixing-time` for the inter-block delay, and `--num-angles` /
`--num-orientations` / `--walkers-per-cell` to trade runtime for smoother curves.
Only Matplotlib is required.

## DEXSY Exchange Map

This example uses the semi-permeable membrane boundary to simulate a
two-compartment DEXSY / diffusion-diffusion correlation experiment with an
explicit finite-pulse sequence. Walkers start in narrow and wide slab
compartments separated by an internal membrane, undergo a finite PGSE block,
exchange during a mixing interval, then undergo a second finite PGSE block and
acquisition. The script sweeps the two gradient amplitudes to build `S(b1, b2)`
and inverts it with `invert_laplace_2d` into a D-D map. The diagonal peaks report
spins that stayed in the same apparent-diffusion compartment; off-diagonal peaks
report exchange between compartments.

```powershell
python examples\plot_dexsy_exchange.py --output results\dexsy_exchange.png
```

Use `--exchange-rate` and `--mixing-time` to tune the off-diagonal peak
amplitudes, and `--walkers-per-cell` / `--substeps` to trade runtime for
smoother Monte Carlo data. The intended inversion path uses SciPy-backed NNLS
from the `opt` extra; without SciPy the script falls back to an unconstrained
preview. Matplotlib is required for plotting.

## T2-T2 Relaxation Exchange (REXSY)

This example is the relaxation-domain analogue of the DEXSY map. It builds an
analytic two-site Bloch-McConnell system with `spin_dynamics.exchange`,
simulates the encode-mix-detect data with `simulate_relaxation_exchange_2d`, and
inverts it with `invert_t2_t2` into a `T2`-`T2` map. Diagonal peaks report spins
whose `T2` is unchanged across the mixing interval; off-diagonal cross peaks
report spins that changed site, and their intensity grows with the exchange
rate. The script also prints the longitudinal mixing propagator so the exchanged
fraction is explicit.

```powershell
python examples\plot_t2_t2_exchange.py --output results\t2_t2_exchange.png
```

Use `--exchange-rate`, `--mixing-time-ms`, and `--population-fast` to tune the
cross peaks. Non-negative inversion uses SciPy-backed NNLS from the `opt` extra;
without SciPy the script falls back to an unconstrained preview. Matplotlib is
required for plotting.

## Internal / Susceptibility Gradients

This example builds the internal field of a packed-grain pore space with
`spin_dynamics.susceptibility`, summarizes the pore-space internal-gradient
distribution, and demonstrates the diagnostic that actually distinguishes an
internal gradient from an applied one. The internal off-resonance map shows the
`cos(2 phi)` dipole lobes around each cylindrical grain. Echo-spacing-dependent
CPMG decay is *not* by itself such a signature -- it occurs for a uniform
applied gradient too -- so the third panel instead runs a fixed-`T_E` CPMG train
at several `B0` values and recovers the `B0**2` scaling of the diffusion-induced
decay rate, which follows from `G_internal` being proportional to
`delta_chi * B0` (an applied gradient would be `B0`-independent).

```powershell
python examples\plot_internal_gradients.py --output results\internal_gradients.png
```

Use `--susceptibility`, `--b0-tesla`, and `--grain-radius-um` to set the
internal-gradient strength, and `--b0-values-tesla` to choose the fields in the
scaling sweep. Matplotlib is required for plotting; the field and gradient
calculations need only NumPy.

## Bipolar 13-Interval PGSTE

This example demonstrates the Cotts 13-interval bipolar PGSTE (Bruker
`diff_stebp`) suppressing the diffusion bias from a constant background gradient.
The left panel traces the toggling-frame applied and background wavevectors for
the 13-interval sequence; the right panel sweeps the background gradient and
plots the apparent diffusion coefficient recovered from a b-value sweep, for the
monopolar stimulated echo (biased) and the 13-interval sequence (flat at the true
value). A third panel runs the explicit moving-walker runner
(`run_cotts_thirteen_interval_walkers`) at two background gradients and overlays
the echoes on `exp(-b D)`, confirming the full simulation reproduces the
moment-model b-value with the suppression intact. It prints the 16-step
`diff_stebp` phase cycle it uses.

```powershell
python examples\plot_bipolar_pgste.py --output results\bipolar_pgste.png
```

Use `--storage-time-ms`, `--half-echo-time-ms`, and `--gradient-duration-ms` to
set the timing, `--max-background` for the swept background range, and
`--walkers-per-cell` / `--walker-points` for the walker panel. The moment-model
calculation needs only NumPy; Matplotlib is required for plotting.

## OGSE Frequency-Resolved Diffusion

This example uses the oscillating-gradient spin-echo backend
(`run_ogse_walkers`) to map the diffusion spectrum `D(omega)`. Where PGSE varies
the diffusion *time*, OGSE varies the oscillation *frequency* of a
cosine-modulated gradient, reaching much shorter effective diffusion times. The
first panel shows the cosine waveform; the second sweeps the frequency and plots
the apparent diffusion coefficient for free diffusion and for reflecting slab
pores of two widths. Free diffusion is flat at the bulk value, while the
restricted curves rise from the low-frequency (tortuosity-limited) value toward
bulk as the frequency increases -- and the smaller pore stays restricted to
higher frequency (transition `f_c ~ D / L^2`).

```powershell
python examples\plot_ogse_frequency_diffusion.py --output results\ogse.png
```

The gradient amplitude is scaled with frequency to hold the b-value fixed (the
cosine-OGSE b-value falls steeply with frequency). Use `--slab-widths`,
`--freq-min` / `--freq-max` / `--num-freqs`, and `--walkers-per-cell` /
`--substeps` to explore. Only Matplotlib is required.

## Received Signal Noise

This non-plotting example compares opt-in white noise and probe-colored
receiver noise while preserving the clean deterministic result fields.

```powershell
python examples\received_signal_noise.py --numpts 51
```

Use `--save-npz results\received_signal_noise.npz` to save selected clean and
noisy CPMG echoes plus imaging k-space arrays.

## J-Coupling Examples

These examples exercise the scalar-coupled spin-1/2 extension layer. The first
prints a compact heteronuclear J-editing fit, while the plotting examples
visualize mixture modulation curves, TANGO-B filter selectivity, and a two-spin
SLIC dip.

```powershell
python examples\heteronuclear_j_editing.py --points 33
python examples\coupled_isochromat_fields.py --points 21
python examples\plot_j_editing_spectrum.py --output results\j_editing_spectrum.png
python examples\plot_j_editing_field_spread.py --output results\j_editing_field_spread.png
python examples\plot_tango_filter.py --target 160 --output results\tango_filter.png
python examples\plot_slic_two_spin.py --j-hz 7 --delta-hz 0.7 --output results\slic_two_spin.png
```

## BPP Relaxation vs Temperature

This plotting example uses `spin_dynamics.relaxation.BPPRelaxationModel` to
compute `T1` and `T2` from an Arrhenius rotational correlation time and the
spectral densities `J(0)`, `J(w0)`, and `J(2w0)`. The default parameters put
`omega0 tau_c` near order unity across the sweep so the BPP turnover is visible.

```powershell
python examples\plot_bpp_relaxation_temperature.py --output results\bpp_relaxation_temperature.png
```

Use `--larmor-mhz`, `--tau-ref-ns`, `--activation-energy-kj-mol`, and
`--coupling-scale` to tune the curve shape and absolute relaxation rates. Only
Matplotlib is required.

## 129Xe Wall-Collision Relaxation

This plotting example uses `spin_dynamics.relaxation.WallCollisionRelaxationModel`
to show how gas-wall relaxation grows with container surface-to-volume ratio.
The model computes the kinetic wall encounter rate from temperature, isotope
mass, and geometry, then applies a per-collision spin depolarization quantum map
in Liouville space. The reported surface relaxivity is derived from those inputs
rather than supplied as the primary fit parameter.

```powershell
python examples\plot_wall_relaxation_xe.py --output results\wall_relaxation_xe.png
```

Use `--depolarization-probability`, `--temperature-k`,
`--accommodation-probability`, and `--selected-diameters-mm` to explore the
microscopic assumptions behind the equivalent wall-limited `T1`.

## Prepolarized T1rho Dispersion

This plotting example combines the prepolarization and BPP relaxation helpers:
it computes a prepared longitudinal magnetization after relaxation in a
polarizing field, applies a simple on-resonance spin-lock `T1rho` dispersion
model with configurable `J(w1)`, `J(w0)`, and `J(2w0)` coefficients, and plots
the remaining locked signal after a fixed spin-lock time.

```powershell
python examples\plot_t1rho_prepolarized_dispersion.py --output results\t1rho_prepolarized_dispersion.png
```

Use `--spin-lock-min-khz` / `--spin-lock-max-khz` for the dispersion axis,
`--spin-lock-time-s` for the readout duration, and
`--prepolarizing-field-t` / `--prepolarization-time-s` for the preparation.
Only Matplotlib is required.

## Earth's-Field NMR with Prepolarization

This plotting example models a low-field proton FID after electromagnet
prepolarization. It computes buildup in a strong prepolarizing field, relaxation
during the field switch or transfer delay, and a demodulated Earth's-field FID
with Gaussian field-spread dephasing. The signal scale is normalized to the
thermal magnetization at Earth's field, so the prepolarization gain is explicit.

```powershell
python examples\plot_earth_field_prepolarized_nmr.py --output results\earth_field_prepolarized_nmr.png
```

Use `--earth-field-ut`, `--prepolarizing-field-mt`,
`--prepolarization-time-s`, `--transfer-delay-s`, and `--field-spread-hz` to
explore the tradeoff between polarization buildup, transfer loss, and
low-field linewidth. Only Matplotlib is required.

## ESR Examples

These examples exercise the first single-electron ESR surface. The
single-crystal example rotates the static field through an anisotropic
`g`-tensor frame and plots the effective `g` and resonant field. The powder
example compares conventional fixed-frequency field sweeps with fixed-field
frequency sweeps for the same orientation grid, and includes options for
derivative CW display, Lorentzian broadening, diagonal `g` strain, and field
strain. The pulsed example shows rectangular-pulse calibration, an on-resonance
FID, and a Hahn echo from a detuned isochromat ensemble. The relaxation example
uses the Liouville-space `ESRRelaxationModel` to compare FID T2 decay,
Hahn-echo T2 decay, and T1 population relaxation. The hyperfine example shows
the classic one-nucleus ESR doublet from an isotropic coupling.

```powershell
python examples\plot_esr_single_crystal.py --output results\esr_single_crystal.png
python examples\plot_esr_powder_spectrum.py --output results\esr_powder_spectrum.png
python examples\plot_esr_powder_spectrum.py --detection-mode derivative --g-strain 0 0 0.005 --output results\esr_derivative_strain.png
python examples\plot_esr_pulsed_echo.py --output results\esr_pulsed_echo.png
python examples\plot_esr_relaxation.py --output results\esr_relaxation.png
python examples\plot_esr_hyperfine_doublet.py --output results\esr_hyperfine_doublet.png
```

## NQR Examples

These examples exercise the early quadrupolar extension. The current pulsed
examples are explicitly spin-1 examples using the `x`, `y`, and `z` transition
labels. The powder nutation example sweeps the nominal SLSE detection pulse
angle, the population transfer example builds a compact two-frequency
perturbation/detection map, and the SLSE relaxation examples sweep RF offset
and pulse period with the Liouville-space relaxation model. The SLSE relaxation
plots use powder averaging by default; pass `--orientation single` with
`--alpha` and `--beta` to inspect one fixed EFG orientation. The EFG broadening
examples use static isochromat distributions to show both the time-domain
response and FFT spectrum. The SLSE broadening example keeps the RF carrier
fixed at the central transition while summing detuned EFG variants; its
spectrum panel is the FFT of the averaged echo over a finite acquisition window
`T_acq`, avoiding the nonphysical echo-train FFT artifact. Use `--acq-us`,
`--noise-snr`, and `--deconvolve` to explore receiver-window broadening,
additive time-domain noise, and regularized deconvolution. The weak-B0 example
is a static transition-spectrum example, not a pulsed simulation, and supports
both spin-1 and spin-3/2 sites.
Use `--n-chi` and `--b1-b0-angle` to control the correlated weak-field powder
average between the static field and RF field.

The full density-matrix model adds spin-3/2 (chlorine-style) examples:
`plot_nqr_full_powder_nutation.py` overlays the spin-1 and spin-3/2 powder
nutation curves, and `plot_nqr_spin32_slse.py` runs the `35Cl` powder SLSE echo
train and shows how a weak Zeeman field (`--b0-mt`) reshapes the decay.
The polarization-enhancement example uses a finite Halbach fringe field, a
Glickstein-style adiabatic crossing model, and by default estimates the
`1H-14N` coupling from the melamine CIF in `QuadrupolarDFT/structures/Melamine`.
The database-driven variant demonstrates the same transport model using
transition frequencies pulled from `NQRDatabase/data/exports/nqr.sqlite`;
by default it loads glycine NQR lines, estimates the `1H-14N` coupling from the
bundled glycine CIF, and reports the estimated signal gain for each transition.

```powershell
python examples\plot_nqr_powder_nutation.py --output results\nqr_powder_nutation.png
python examples\plot_nqr_population_transfer.py --output results\nqr_population_transfer.png
python examples\plot_nqr_slse_offset.py --output results\nqr_slse_offset.png
python examples\plot_nqr_slse_spacing.py --output results\nqr_slse_spacing.png
python examples\plot_nqr_efg_broadening.py --output results\nqr_efg_broadening.png
python examples\plot_nqr_temperature_broadening.py --output results\nqr_temperature_broadening.png
python examples\plot_nqr_slse_efg_broadening.py --output results\nqr_slse_efg_broadening.png
python examples\plot_nqr_weak_b0_spectrum.py --output results\nqr_weak_b0_spectrum.png
python examples\plot_nqr_full_powder_nutation.py --output results\nqr_full_powder_nutation.png
python examples\plot_nqr_spin32_slse.py --output results\nqr_spin32_slse.png
python examples\plot_nqr_polarization_enhancement.py --output results\nqr_polarization_enhancement.png
python examples\plot_nqr_database_prepolarization.py --output results\nqr_database_prepolarization.png
```

## Radiation Damping

These examples couple deterministic radiation-damping back-action to tuned or
matched probe parameters. The FID workflow also reports the analytic
Section 10.2.5 envelope for direct comparison.

```powershell
python examples\radiation_damping_fid.py --probe matched --points 401
python examples\radiation_damping_cpmg_train.py --probe tuned --numpts 21 --num-echoes 4
python examples\plot_radiation_damping.py --output results\radiation_damping.png
python examples\plot_radiation_damping_detuning.py --output results\rd_detuning.png
python examples\plot_radiation_damping_cpmg_train.py --output results\rd_cpmg.png
python examples\nmr_maser.py
python examples\plot_nmr_maser.py --output results\nmr_maser.png
```

Use `--model circuit`, `--detuning`, and `--phase` on the FID example to inspect
the finite-ringdown probe model. The maser examples use an inverted longitudinal
pump and show the threshold where radiation damping becomes gain; the plotting
example defaults include a strong `16x`-threshold pump so saturation and
inversion depletion are visible.

## Plot Inverse Laplace Examples

This example requires SciPy and Matplotlib. It generates synthetic T1, T2,
T1-T2, and D-T2 data, adds Gaussian noise at several requested SNR levels, and
plots the regularized non-negative inverse Laplace recoveries.

```powershell
python examples\plot_inverse_laplace.py --output results\inverse_laplace.png
```

Use `--snr-levels`, `--regularization`, `--regularization-order`, `--cases`,
and `--t1-mode` to compare conditioning and T1 preparation choices. Add
`--auto-regularization` to select a separate regularization strength for each
panel from its SNR estimate.

## Plot Probe Parameter Sweep

This example requires Matplotlib.

```powershell
python examples\plot_probe_parameter_sweep.py --probe tuned --sweep q --output results\tuned_q_sweep.png
```

## Plot Probe CPMG Comparison

This example requires Matplotlib.

```powershell
python examples\plot_probe_cpmg.py --numpts 101 --output results\probe_cpmg.png
```

The asymptotic magnetization panel shows magnitude by default because tuned and
untuned probe responses can rotate most of the signal into the imaginary
component. Use `--masy-component real`, `imag`, or `phase` to inspect a specific
component.

## Plot Optimization Workflows

This example requires Matplotlib. It runs compact tuned-probe refocusing and
target-excitation optimization diagnostics, then probes the current
inverse-excitation objective. The inverse stage starts from the phase-flipped
target pulse, then uses a MATLAB-style multi-start strategy that perturbs the
best inverse found so far. Treat the inverse panel as a parity diagnostic, not
as a validated inverse pulse-design recipe yet.

```powershell
python examples\plot_optimization_workflows.py --numpts 11 --segments 2 --starts 2 --inverse-starts 4 --output results\optimization_workflows.png
```

The defaults are intentionally small. Increase `--numpts`, `--segments`,
`--starts`, `--excitation-segments`, `--inverse-starts`, and the optimizer
settings when using it as the start of a real pulse-design study.

## Plot Optimization Pipeline

This example requires Matplotlib. It runs a compact ideal-v0crit refocusing
multi-start, converts that result to MATLAB-style cells, reconstructs the
selected refocusing axis, and then calls the plotting-free tuned
excitation/inverse-excitation pipeline helper. The figure shows the selected
refocusing pulse, objective histories, received spectra, and echo magnitudes.

```powershell
python examples\plot_optimization_pipeline.py --numpts 11 --refocusing-segments 2 --refocusing-starts 2 --excitation-segments 2 --excitation-starts 2 --inverse-starts 3 --output results\optimization_pipeline.png
```

Use this example when checking the end-to-end optimization handoff. The inverse
stage still reports both objective-best and residual-best inverse candidates
because strong inverse-cancellation parity remains under validation.

## Diagnose Optimization Backends

This non-plotting diagnostic compares the NumPy pattern-search fallback, the
optional SciPy backend, and a random inverse-candidate baseline. It prints
objective values and residual/target area ratios so optimizer behavior can be
debugged without relying on plots.

```powershell
python examples\diagnose_optimization_backends.py --backend all --numpts 21 --segments 3
```

On WSL2, create a virtual environment in the Linux filesystem and install the
optional SciPy backend before running the comparison. On Windows/OneDrive
checkouts, prefer an external unsynced environment such as
`C:\Users\smandal\codex-envs\python-spin-dynamics`.

```bash
python3 -m venv ~/venvs/python-spin-dynamics
source ~/venvs/python-spin-dynamics/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[opt]"
python examples/diagnose_optimization_backends.py --backend all --numpts 21 --segments 3
```

On Windows with Conda, keep the environment outside the OneDrive checkout:

```powershell
conda create -p "C:\Users\smandal\codex-envs\python-spin-dynamics" python=3.11 numpy scipy matplotlib -y
conda run -p "C:\Users\smandal\codex-envs\python-spin-dynamics" python -m pip install -e .
```

## Plot Finite Train Workflows

This example requires Matplotlib. It compares finite CPMG echo trains across
ideal, tuned, untuned, and matched probe models.

```powershell
python examples\plot_finite_train_workflows.py --numpts 65 --num-echoes 4 --output results\finite_trains.png
```

This plot defaults to automatic grid refinement and reports the effective
number of offsets used. Disable refinement only when intentionally testing the
rephasing guard.

## Plot Diffusion Sweep

This example requires Matplotlib. It plots a compact matched-probe diffusion
CPMG Q sweep, including echo-integral decay and a Q-by-echo heatmap.

```powershell
python examples\plot_diffusion_sweep.py --numpts 65 --num-echoes 3 --output results\diffusion_sweep.png
```

Use `--dz-um` to set the physical slice thickness that determines the
normalized offset span; the default is intentionally compact for a stable
teaching plot.

## Plot Time-Varying Field Sweep

This example requires Matplotlib. It visualizes the ideal time-varying-field
CPMG amplitude sweep, including the B0 waveform and final echoes.

```powershell
python examples\plot_time_varying_sweep.py --numpts 51 --num-echoes 12 --output results\time_varying_sweep.png
```

## Plot Moving Isochromats

These examples require Matplotlib and exercise the `spin_dynamics.motion`
helpers. The first moves a transverse spin packet linearly through static B0
and inside-out-style receive-B1 maps.

```powershell
python examples\plot_motion_linear.py --output results\motion_linear.png
```

The pipe-flow example uses the same moving-isochromat CPMG engine with a
Poiseuille velocity callback. A cylindrical pipe is polarized upstream, then the
flowing packet is detected by downstream transmit/receive coils. Sweeping the
mean velocity shows incomplete prepolarization and motion-induced echo
dephasing.

```powershell
python examples\plot_cpmg_pipe_flow.py --output results\cpmg_pipe_flow.png
```

The diffusion example runs a simple idealized CPMG loop while Brownian walkers
diffuse in a static gradient.

```powershell
python examples\plot_motion_diffusion_cpmg.py --output results\motion_diffusion_cpmg.png
```

The UDD counterpart compares CPMG and Uhrig timing with the same walker
diffusion, static gradient, and `T1`/`T2` relaxation model. By default it also
adds a slow sinusoidal gradient fluctuation, creating the correlated
low-frequency dephasing regime where UDD can outperform evenly spaced CPMG.

```powershell
python examples\plot_motion_diffusion_udd.py --output results\motion_diffusion_udd.png
```
