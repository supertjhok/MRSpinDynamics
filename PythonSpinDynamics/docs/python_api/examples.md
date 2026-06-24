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

```powershell
python examples\plot_nqr_powder_nutation.py --output results\nqr_powder_nutation.png
python examples\plot_nqr_population_transfer.py --output results\nqr_population_transfer.png
python examples\plot_nqr_slse_offset.py --output results\nqr_slse_offset.png
python examples\plot_nqr_slse_spacing.py --output results\nqr_slse_spacing.png
python examples\plot_nqr_efg_broadening.py --output results\nqr_efg_broadening.png
python examples\plot_nqr_temperature_broadening.py --output results\nqr_temperature_broadening.png
python examples\plot_nqr_slse_efg_broadening.py --output results\nqr_slse_efg_broadening.png
python examples\plot_nqr_weak_b0_spectrum.py --output results\nqr_weak_b0_spectrum.png
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

The second runs a simple idealized CPMG loop while Brownian walkers diffuse in
a static gradient.

```powershell
python examples\plot_motion_diffusion_cpmg.py --output results\motion_diffusion_cpmg.png
```
