# Analysis

Analysis helpers live under `spin_dynamics.analysis`. They are intended for
post-processing simulated or measured signals rather than for reproducing a
specific MATLAB workflow line by line.

## Inverse Laplace Transforms

The inverse Laplace utilities provide 1D and separable 2D non-negative
Tikhonov-regularized solves for common NMR kernels:

- T2 decay: `exp(-te / T2)`;
- T1 saturation recovery: `1 - exp(-tau / T1)`;
- T1 inversion recovery: `1 - 2 exp(-tau / T1)` (signed);
- diffusion attenuation: `exp(-b D)`;
- T1-T2 and D-T2 separable products.

The T2, saturation-recovery, and diffusion kernels are non-negative, so
magnitude or phase-corrected real data are both fine. The **inversion-recovery**
kernel is signed (negative for `tau < T1 ln 2`), so it requires *signed*,
phase-corrected real data: magnitude data folds the negative lobe upward and
cannot be fit by the signed kernel. `invert_laplace_1d`/`invert_laplace_2d`
emit a warning if they detect an entirely non-negative signal paired with a
signed kernel.

The distribution axes are user supplied, so choose log-spaced T1/T2 grids for
broad relaxation spectra and a physically plausible linear or log diffusion grid
for D-T2 work. The recovered `distribution` holds the *per-grid-point amplitude*
`x` in `signal = K @ x`, not a probability density: on a uniform log grid the
summed amplitude of a peak approximates that component's signal fraction, but
the amplitudes are unnormalized and are not divided by bin width. Divide by the
log-spacing if a true density `f(log T2)` is required.

```python
import numpy as np

from spin_dynamics.analysis import invert_t2, t2_kernel

echo_times = np.linspace(0.5e-3, 90e-3, 40)
t2_axis = np.logspace(-4, -1, 60)

distribution = np.zeros_like(t2_axis)
distribution[25] = 1.0
signal = t2_kernel(echo_times, t2_axis) @ distribution

result = invert_t2(
    signal,
    echo_times,
    t2_axis,
    regularization=5e-4,
    regularization_order=2,
)
```

For 2D data, the forward model is:

```text
data = K1 @ distribution @ K2.T
```

Use `invert_t1_t2(...)` for recovery/inversion-by-echo data and
`invert_d_t2(...)` for b-value-by-echo data. Use `invert_t2_t2(...)` for
relaxation-exchange (REXSY) encode-by-detect data, where both axes share the T2
decay kernel and off-diagonal peaks signal site exchange; pair it with
`spin_dynamics.exchange.simulate_relaxation_exchange_2d` and see
[Chemical / Site Exchange](exchange.md). The generic `invert_laplace_1d(...)`
and `invert_laplace_2d(...)` functions also accept precomputed kernel matrices.

The regularization penalty can be a scalar or a `Regularization` object.
`regularization_order=0` damps amplitudes, `1` damps slopes, and `2` damps
curvature. Two-dimensional solves also accept per-axis regularization tuples,
for example `regularization=(1e-4, 5e-4)`.

## SNR-Based Regularization Selection

`spin_dynamics.analysis` also provides an SNR-informed selector based on the
discrepancy principle. It estimates the expected noise RMS from the observed
data and an RMS SNR value, scans a logarithmic strength grid, and chooses the
strongest regularization whose residual norm remains within the expected noise
norm. If every candidate exceeds the target residual, it chooses the closest
candidate and returns the full candidate trace for inspection. This is a
discrepancy-principle selector and therefore needs a known or estimated SNR; it
is **not** the Butler-Reeds-Dawson (BRD) auto-`alpha`, an L-curve, or a GCV
selector.

The SNR convention is `clean_signal_rms / noise_rms`. For measured data the
clean RMS is usually unavailable, so the helper estimates the noise RMS from
the observed RMS using `observed_rms^2 ~= clean_rms^2 + noise_rms^2`.

```python
import numpy as np

from spin_dynamics.analysis import select_regularization_1d

selection = select_regularization_1d(
    noisy_signal,
    echo_times,
    t2_axis,
    snr=40.0,
    kernel="t2",
    strengths=np.logspace(-8, 1, 37),
    regularization_order=2,
)

result = selection.result
lambda_selected = selection.selected_strength
candidate_table = selection.candidates
```

For separable 2D maps, use `select_regularization_2d(...)`. It selects a
shared strength scale and can apply per-axis ratios:

```python
selection = select_regularization_2d(
    noisy_t1_t2_data,
    recovery_times,
    echo_times,
    t1_axis,
    t2_axis,
    snr=25.0,
    kernel1="t1_ir",
    kernel2="t2",
    axis_strength_ratio=(1.0, 2.0),
)
```

The synthetic plotting example can use this mode directly:

```powershell
python examples\plot_inverse_laplace.py --auto-regularization --output results\inverse_laplace_auto.png
```

Non-negative solves use SciPy's `nnls`, so install the optional optimization
extra or include SciPy in your Conda environment. Unconstrained least-squares
solves can be run with `nonnegative=False` using only NumPy.

The kernels are real-valued distribution models. Pass phase-corrected real
signals for complex acquisitions; magnitudes are acceptable only for the
non-negative kernels (T2, saturation recovery, diffusion), never for the signed
inversion-recovery kernel.
