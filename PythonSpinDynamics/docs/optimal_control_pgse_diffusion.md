# PGSE Optimal Control: q-vector + detected-SNR objectives (Milestone 5)

Design/progress note for the diffusion (PGSE) capstone of
`spin_dynamics.optimal_control`: co-optimize the RF pulses **and** the gradient
waveform of a pulsed-gradient spin echo so that, on realistic hardware (a tuned
RF probe on transmit *and* receive, a gradient driver with `L/R` slew + eddy
currents, and an inhomogeneous `B0`/`B1` field), the sequence both delivers the
intended diffusion encoding (**correct q-vector**) and maximizes the
**detected echo SNR**. This is the end-to-end exerciser of the M4 hardware-
response layer; see `optimal_control_hardware_response.md`.

Builds on M1 (RF-robust), M2 (gradient channel), M3 (NQR), M4 (probe/driver
response). Reuses the existing PGSE machinery's conventions (below) rather than
inventing new ones.

## Reused conventions (from `workflows/pgse.py`, `fields/positions.py`, `motion.py`)

- **Gyromagnetic ratio** `gamma = 2.675e8` rad/(s*T) (proton default).
- **Gradient coupling** is Lagrangian: a spin at position `r` (meters) sees an
  offset `g . r`, the `positions @ gradient` rule (`fields.positions.gradient_offset`).
  The GRAPE gradient channel already matches this: `g(t) * position_k * h_grad`
  with `h_grad = 2*pi*Iz`.
- **q-vector** `q(t) = integral gamma*g(tau) dtau` (rad/m); a stationary-spin
  echo requires the net `q` at the echo to return to zero. `gradient_moment_b_value`
  (piecewise-constant segments) computes `b = integral q(t)^2 dt` and already
  **warns when the residual `q` != 0** -- exactly the "correct q-vector" check.
- **Stejskal-Tanner** rectangular b-value `b = (gamma*G*delta)^2*(Delta - delta/3)`
  is the closed-form reference the moment integral must reproduce.
- **Detection** is a coherent sum `signal = sum_k w_k*(M-_k + M+_k)` over the
  ensemble (`motion.receive_signal`); in the Hilbert-space GRAPE picture the
  detected quantity is `sum_k w_k <psi_k|I+|psi_k>`.

## What the two objectives mean

Let `g_cmd(t)` be the commanded gradient and `g(t) = L_grad(g_cmd)` the delivered
gradient (after the RL+eddy driver). The RF envelope is likewise delivered
through the tuned probe `L_rf`.

1. **Correct q-vector** -- a functional of the *delivered* gradient:
   - encode moment: first-lobe `q_enc = integral_lobe gamma*g dt` hits a target
     `q*` (the intended diffusion wavevector); and
   - refocus: net `q` at the echo returns to ~0 so static spins rephase.
   Eddy droop makes the delivered lobes asymmetric, so `q_enc` drifts off `q*`
   and the residual moment spoils refocusing -- the optimizer pre-emphasizes
   `g_cmd` (bounded by `gradient_max`) to correct both.

2. **Maximize detected SNR** in inhomogeneous `(B0, B1)` -- the 90/180 RF
   (distorted by `L_rf`) must produce a large, refocused transverse magnetization
   across the `(B0, B1)` distribution; the echo is acquired over a short window,
   passed through the **Rx** tuned probe (same probe, by reciprocity), matched-
   filtered, and its peak taken as the detected amplitude. With a fixed noise
   floor (the chosen model), maximizing detected SNR = maximizing the matched-
   filtered echo peak of the coherently-summed, Rx-filtered signal.

The example optimizes a weighted objective
`maximize  detected_SNR  -  lambda_q*(q_enc - q*)^2  -  lambda_0*q_echo^2`,
co-optimizing RF phase (90 + 180) and the gradient waveform.

## Ensemble: (position x B0 x B1) -- the propagator gap

Each ensemble member `k` is fully specified by four operators with the *shared*
`(amplitude, phase, gradient)` controls:

- `h_drift_k` -- its B0 offset (via `coupled_spin_system([offset_k], ...)`);
- `h_grad_k = r_k * (2*pi*Iz)` -- its position (`position_gradient_batch`);
- `(h_x_k, h_y_k) = beta_k*(h_x, h_y)` -- its B1 scale factor.

The current `propagate_batched_grad` shares `h_x`/`h_y` across the batch, so it
cannot also vary B1 per member. **New primitive** `propagate_batched_grad_controls`
vmaps over all four (`h_drift`, `h_x`, `h_y`, `h_grad`) with shared controls --
a mechanical extension mirroring `propagate_batched_controls` (which does the
per-case `(h_x,h_y)` for NQR powder but without a gradient channel). The
objective's current "control_operator_batch and the gradient channel cannot be
combined" guard is lifted for this combined path.

## Library pieces to add (promoted, tested)

1. **`_jax_propagation.propagate_batched_grad_controls`** (+ NumPy reference) --
   the (B0 x B1 x position) gradient propagator. Diffusion is a *static position
   ensemble* only (per the scoping decision): the gradient dephases/rephases the
   position grid; no Bloch-Torrey/random-walk. (An explicit b-value attenuation
   can be layered on later via `gradient_moment_b_value`; the existing walker
   backends in `workflows/pgse.py` remain the route to true restricted diffusion.)

2. **Gradient-moment / q-vector objective term** (`objectives`) -- a
   differentiable functional of the *delivered* gradient: `q(t)` by cumulative
   trapezoid, `q_enc` over an encode window, and `q_echo` (residual). Validated
   against `workflows.pgse.gradient_moment_b_value` on the delivered waveform.

3. **Detected-echo SNR objective** (`objectives`) -- the deferred M4 item:
   coherently sum `<I+>` over the ensemble, evolve it over a short acquisition
   window as an isochromat FID (`sum_k w_k <I+>_k * exp(i*2*pi*dnu_k*t)`), filter
   through a `ReceiverResponse` (Rx tuned probe), matched-filter, and return the
   peak over a fixed noise floor. Reduces to the coherent echo amplitude when no
   receiver is supplied.

## The example

`examples/plot_grape_pgse_diffusion.py` (plot-prefixed): a proton PGSE, tuned
probe on Tx+Rx, gradient driver with eddy terms, `(B0, B1)` inhomogeneity and a
position grid. Co-optimizes RF phase (90 + 180) and the gradient waveform against
the weighted detected-SNR + q-accuracy objective, versus a naive
(rect-RF + ideal-gradient-assumed) baseline. Panels:

- pulse sequence (RF + commanded vs delivered gradient -- eddy pre-emphasis);
- delivered q(t) with target `q*` and the residual at the echo;
- detected echo amplitude heatmap over `(B0, B1)`;
- q accuracy and detected-SNR: naive baseline vs optimized.

## Milestones

- [x] Add this design note.
- [x] `propagate_batched_grad_controls` (+ NumPy ref) + parity test (jax vs numpy
      ~1e-15). Added an `expm` (degeneracy-safe) path -- on-resonance spins have
      `H_seg == 0` in RF/gradient gaps, whose `eigh` gradient is NaN.
- [x] Combined per-case `(h_drift, h_x, h_y, h_grad)` propagation with shared
      controls (B0 x B1 x position), consumed by `make_pgse_objective`.
- [x] Differentiable q-vector / moment term (`gradient_moments`), with the
      coherence-sign convention, validated vs `workflows.pgse.gradient_moment_b_value`.
- [x] Detected-echo SNR objective (`detected_echo_signal` / `detected_echo_snr`),
      `ReceiverResponse` on the acquired echo. (NaN fix: `|s|^2` as `Re(s*conj s)`,
      not `abs(s)**2`, which has a NaN gradient at echo zeros.)
- [x] `plot_grape_pgse_diffusion.py` example + registration. In a **grossly
      inhomogeneous** field (B0 spread ~2x nutation) the hard 180 refocuses only
      within ~+-nutation; phase-only GRAPE finds a broadband phase-modulated
      refocusing pulse (RP2 / symmetric-phase-alternating family) that recovers
      the echo across the whole band -- ~5x detected-SNR gain -- while the
      gradient pre-emphasis fixes the eddy-drooped q-vector (naive ~17% low +
      imperfect refocus -> ~0% error + refocused). **Multistart is essential**: a
      rectangular (phase=0) seed on a symmetric offset ensemble sits at an exact
      saddle, so a single start never moves the phase.
- [x] Manual section + API reference regen + PDF.

Also hardened `build_control_delivery`'s polar conversion (gradient-safe `|u|` /
`angle(u)` via floor + double-where) so a gapped RF template rung down through
the probe cannot NaN the gradient; a no-op (~1e-15) for constant-amplitude pulses.

## Open questions

- Whether the encode/refocus windows are fixed by the sequence timing or
  co-optimized (start simple: fixed PGSE timing, optimize shapes within lobes).
- Whether to expose the weighted PGSE objective as a library factory or keep the
  weighting in the example (leaning: keep the *pieces* in the library, the
  weighting in the example).
