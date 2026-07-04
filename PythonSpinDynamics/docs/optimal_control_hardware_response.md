# Optimal-Control Hardware Response (GRAPE Milestone 4)

This note is the design/progress document for folding **probe and driver
dynamics** into the `spin_dynamics.optimal_control` GRAPE engine. It covers the
RF probe (tuned / untuned / matched), a *new* gradient-driver model
(RL time constant plus eddy-current terms), and a receiver transfer function
for detected-signal objectives.

Prior milestones: M1 RF-robust GRAPE, M2 gradient-waveform channel,
M3 NQR / quadrupolar targets. This is M4.

## Problem

GRAPE currently optimizes an *ideal* piecewise-constant control and assumes the
spins see exactly that waveform. Real hardware does not deliver it:

- A **tuned or matched RF probe** has finite bandwidth (loaded `Q`). It filters
  the pulse envelope — magnitude roll-off, phase / group delay, and a
  post-pulse **ring-down** tail during which spins keep evolving.
- A **gradient driver** cannot switch instantly: the coil is inductive
  (`tau = L/R` slew) and surrounding conductors carry **eddy currents** that
  droop the delivered gradient with one or more extra time constants.
- The **receiver** chain band-limits the *detected* signal, which matters when
  the objective is the acquired echo/FID (e.g. CPMG SNR-per-time) rather than a
  magnetization trajectory.

Optimizing against the ideal waveform therefore over-states achievable fidelity
and produces pulses that degrade on the real probe.

## Core principle: an LTI stage in the optimization loop

Each piece of hardware above is, to excellent approximation, a **linear
time-invariant (LTI) filter**. The clean way to account for it is to insert that
filter as one differentiable stage between the *commanded* control and the
Hamiltonian, and leave the GRAPE core (propagator, fidelity, ensemble/powder
batching) untouched.

```
command u_cmd = amp·e^{iφ}  ──►  L_tx (probe)  ──►  delivered B1(t)  ──►  H_seg  ──►  propagate ─┐
        ▲ box constraints stay on the COMMAND                                                    │
                                                                                          Mxy(t) │
gradient g_cmd  ──►  L_grad (driver)  ──►  delivered G(t)  ──►  H_seg ────────────────────────────┤
        ▲ gradient_max stays on the COMMAND                                                       │
                                                                                                  ▼
                                       objective ◄── H_rx (receiver, output side) ◄── detected v(t)
```

Two properties make this drop into the existing code cleanly:

1. **Autodiff supplies the adjoint for free.** Every stage is JAX, so
   `jax.grad` differentiates through the filter — no hand-derived circuit
   gradient. (For reference the adjoint of a convolution is correlation; of a
   frequency-domain multiply `H`, it is multiply by `conj(H)`; of a state-space
   filter, the reverse-time adjoint recursion. We never write these by hand.)

2. **Constraints already live on the command.** `ControlBounds` bounds the
   control vector — i.e. what the amplifier can produce. Placing the filter
   *after* the bounds makes the physics correct automatically: the optimizer
   pre-emphasizes up to the real hardware ceiling and, where the bandwidth
   genuinely cannot deliver, returns the best *achievable* pulse instead of an
   unphysical pre-emphasized spike. This is strictly stronger than classic
   post-hoc pre-emphasis (invert `L`, pre-distort the command), which routinely
   demands voltages the amplifier does not have.

### Transmit: filter once, then batch-propagate

The coil produces **one** physical `B1(t)`. Every isochromat sees the same
delivered waveform and differs only by its own detuning, which is already in
`h_drift_batch`. So the transmit filter is applied **once**, in time, to the
complex envelope, yielding one effective `(amp(t), phase(t))` that feeds the
existing batched propagator. There is **no** per-offset transmit filtering — a
common mistake that conflates transmit with the (genuinely per-frequency)
receiver response below.

## Rotating-frame envelope model

GRAPE works with the complex envelope `u = amp·e^{iφ}` in the rotating frame. A
probe that is a bandpass around the carrier `ω0` becomes a **low-order lowpass
acting on the envelope** — this *is* the "limited bandwidth" of the probe. Tuned
and matched probes have **distinct dynamics** and are modelled separately.

### Tuned probe — 2nd-order, single resonance

Series/parallel RLC, one resonance at `ω0 = 1/√(LC)`, power bandwidth
`Δω = R/L`, `Q = ω0/Δω`. The baseband-equivalent of this bandpass is a **single
complex pole** — a one-pole lowpass on the envelope with the ring-down time
constant

```
τ_probe = 2L/R = 2Q/ω0 = Q/(π·f0)
```

Magnitude roll-off *and* phase/group delay are both captured by the complex
pole. **Mistuning** moves the pole off the carrier → asymmetric magnitude plus
linear phase across the band; still one pole, now with an imaginary offset.

### Matched probe — 3rd-order network, its own dynamics

The matched probe is a tuned coil plus a two-element matching network
(`C1`, `C2`); `matched.py` integrates a **3rd-order** ODE and already exposes the
network transfer functions `tf1 = s/(s³+c3 s²+c2 s+c1)` and
`tf2 = -i s³/(s³+c3 s²+c2 s+c1)`. This is **not** reducible to the tuned
one-pole model: the matching network adds a pole, changes the loaded `Q`, and
alters the phase and wideband roll-off. We keep a separate `MatchedProbeResponse`
that uses the full rational `H_env` derived from `(L, R, C1, C2)` (or reuses the
existing `find_coil_current` transfer functions). The one-pole form is only a
near-resonance approximation and is not the default here.

### Untuned probe — near-flat

Low `Q`, very wide bandwidth: `H(s) = 1/(R + sL)`. In the envelope picture this
is nearly flat magnitude with a mild phase — the model correctly predicts
minimal distortion. Included for completeness and A/B comparison.

### Source of `H_env`

Expose two paths per probe: `from_probe_params(sp, pp)` reusing the existing
`tuned_probe_rx_tf` / `untuned_probe_rx_tf` / matched `tf1,tf2` circuit code
(reciprocity: the Tx and Rx coil response share the same frequency shape), and a
closed-form shortcut (`TunedProbeResponse.single_pole(tau, detuning=0)`), for
quick studies and tests.

## Gradient-driver model (new)

Same abstraction, on the `h_grad` channel. The delivered gradient is the command
through **one RL slew pole in series with a few eddy-current shelf terms**:

```
             1              ┌                         α_k        ┐
H_grad(s) = ─────── ·  Π_k  │ (1 - α_k)  +  ───────────────────  │
           1 + sτ_RL        └                     1 + s·τ_k       ┘
             slew                        eddy term (DC gain 1)
```

- `τ_RL = L/R` of the gradient coil/amplifier — the finite slew pole.
- Eddy currents follow the classic MRI model: a step command yields
  `G(t) = G_cmd·[1 − Σ_k α_k e^{−t/τ_k}]`. Each term is a first-order shelf; the
  product has DC gain 1 (steady-state gradient is correct) and transient droop
  set by `(α_k, τ_k)`.

This is implemented as a **state-space filter** (`x' = Ax + Bu`, `lax.scan`),
which is strictly causal and represents the post-pulse eddy tail directly — the
right form when the gradient is part of a slice-select or diffusion-weighting
pulse whose fidelity depends on the delivered gradient after the command ends.

## Receiver transfer function (detected-signal objective)

When the objective is the **detected signal** — not the commanded trajectory or
the final magnetization — the receiver chain must be included as an output-side
LTI stage:

```
Mxy(t)  ──►  H_rx(f) [coil + preamp + matching]  ──►  matched filter (mf_type)  ──►  detected v(t)
```

The existing `*_probe_rx_tf` and `matched_probe_rx` already compute `H_rx` and
the matched filter on the `del_w` grid; `ReceiverResponse` wraps them so an
objective such as detected-echo SNR or matched-filter peak can be optimized.
This is the output-side mirror of the transmit story and connects directly to
the CPMG SNR-per-time metric. **Unlike transmit, the receiver response is
genuinely per-frequency** (each offset sits at a different point on `H_rx`), so
it is applied to the acquired spectrum/echo, not to a single time-domain
envelope.

## Numerics

- **RF probes → FFT multiply.** `ifft( fft(u) · H_env(Δ) )` with `H_env` sampled
  on the FFT grid. `O(N log N)`, exact for the sampled response, trivially
  differentiable. Ring-down is captured by zero-padding the envelope window so
  the tail is resolved (see below).
- **Gradient driver → state-space / `lax.scan`.** Strictly causal; explicit
  eddy tail.
- **Oversampling.** GRAPE segments are a coarse piecewise-constant grid and the
  probe/eddy time constants can be `≲ dt`. Each response upsamples the command
  onto a finer grid (`oversample` factor), filters there, and the propagator
  runs on the fine grid. (Distortion + interpolation, per Motzoi/Gambetta/
  Wilhelm.)
- **Ring-down tail counts.** Propagation continues through the delivered
  waveform *including* the post-command ring-down / eddy tail; the spins really
  do rotate during it. This means appending extra (zero-command) propagation
  segments whose length covers a few `τ`.
- **Backward compatible.** `rf_response=None` / `gradient_response=None` /
  `receiver_response=None` reproduce the current ideal behavior bit-for-bit.

## Proposed package layout

```text
spin_dynamics/optimal_control/
  control_response.py     # NEW
    ControlResponse            # protocol: apply(u, dt) -> u_delivered (JAX)
    TunedProbeResponse         # 2nd-order → 1-pole envelope; FFT multiply
    MatchedProbeResponse       # 3rd-order network H_env; FFT multiply
    UntunedProbeResponse       # near-flat; FFT multiply
    GradientDriverResponse     # RL pole + eddy shelves; state-space/lax.scan
    ReceiverResponse           # output-side H_rx + matched filter (per-offset)
```

Thread optional `rf_response=`, `gradient_response=`, `receiver_response=`
through `make_grape_objective` and `grape_optimize`. The transmit/gradient
filters sit between control unpacking and `_segment_propagator` on the
oversampled grid; the receiver filter sits on the observable before the fidelity
reduction.

## Correctness invariants and tests

- **Identity:** `L = identity` reproduces current results bit-for-bit.
- **Adjoint check:** finite-difference gradient vs autodiff through each filter.
- **Fidelity recovery:** a deliberately band-limited tuned probe degrades an
  uncompensated rect/composite pulse; GRAPE-with-`L` recovers fidelity and the
  recovered command shows the expected pre-emphasis, staying within
  `amplitude_max_hz`.
- **Matched ≠ tuned:** the same coil with vs without the matching network gives
  measurably different `H_env` and different optimized pulses.
- **Gradient droop:** a slice-select pulse degrades under RL+eddy droop and is
  corrected by GRAPE; delivered gradient (not command) matches target.
- **Ring-down:** magnetization computed with vs without the appended tail
  differs for high-`Q` / slow-eddy cases, confirming the tail is propagated.
- **Receiver:** detected-SNR objective with `ReceiverResponse` differs from a
  magnetization objective and tracks the CPMG SNR-per-time reference.

## Robustness extension (reuses M1 machinery)

Probe-robust pulses fall out of the existing ensemble path
(`hamiltonian_batch` / `ensemble_reduction`): make the *filter parameters*
(tuning offset, `Q`, dominant eddy `τ_k`) the ensemble axis and optimize a pulse
robust to probe detuning / eddy variation — same batching, now over `L`.

## Milestones

- [x] Add this design/progress document.
- [x] Add the `EnvelopeResponse` FFT-multiply base and adjoint (finite-diff) test.
- [x] Add `TunedProbeResponse` (1-pole; `from_quality_factor` / `from_circuit`).
- [x] Add `MatchedProbeResponse` (two-pole network response; `from_loaded_q`).
- [x] Add `UntunedProbeResponse`.
- [x] Add `GradientDriverResponse` (RL + eddy, state-space) + eddy-fit helper.
- [x] Wire `rf_response` / `gradient_response` into objective + solver with
      oversampling and ring-down tail (identity fallback, bit-for-bit).
- [x] Add `ReceiverResponse` (LTI filter + resample-to-grid primitive).
- [ ] Detected-signal GRAPE objective wiring (acquisition + matched filter +
      SNR) using `ReceiverResponse` -- the remaining M4 item; needs the echo/FID
      acquisition model, so deferred as its own increment.
- [x] Fidelity-recovery + matched-vs-tuned + gradient-droop examples/tests
      (`examples/grape_probe_response_pulse.py`, `tests/test_optimal_control_response.py`).
- [ ] Dedicated probe-robust (detuning/`Q` ensemble) example (mechanism already
      supported via the ensemble path; example not yet written).
- [x] Manual section (§ Hardware response) + API reference regen + PDF.

## Resolved design decisions

- **Receiver frequency grid:** the receiver stage **resamples `H_rx` onto the
  optimizer's frequency grid** (rather than reusing the probe `del_w` verbatim),
  so the detected-signal objective and the propagation ensemble share one grid.
- **Eddy terms:** expose a **helper that fits `(α_k, τ_k)` from a measured (or
  simulated) step response**, in addition to accepting them directly.
  `GradientDriverResponse` takes explicit terms; the helper produces them.
- **Ring-down / eddy tail length:** support **both** policies via a switch —
  a fixed multiple of the dominant `τ` (default) *or* an explicit
  acquisition/dead-time window supplied by the user.
