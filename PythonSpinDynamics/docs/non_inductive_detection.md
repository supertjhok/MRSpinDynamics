# Non-Inductive Detection: SQUID and OPM Magnetometers

This note is the design/progress document for adding **non-inductive detectors**
— SQUIDs (Superconducting QUantum Interference Devices) and OPMs (Optically
Pumped / atomic Magnetometers) — to the `spin_dynamics` signal-detection chain.
Every receiver model in the package today assumes a conventional **inductive
(Faraday) coil**. SQUIDs and OPMs are the enabling detectors for ultra-low-field
(ULF) NMR/MRI and for zero-/low-field NQR, and they obey a *different* detection
physics that the current pipeline cannot express.

Primary reference: Clarke, Hatridge & Mößle, "SQUID-Detected MRI in Microtesla
Fields," *Annu. Rev. Biomed. Eng.* **9**, 389 (2007) — in `References/`. NQR
anchors: sub-fT/√Hz RF atomic magnetometer detecting ¹⁴N NQR (US 7,521,928;
arXiv physics/0611058) and SQUID-detected ¹⁴N NQR (IEEE Trans. Appl. Supercond.;
*Rev. Sci. Instrum.*).

## Problem

The detected-signal chain models Faraday induction end-to-end:

- `interference/coils.py::_induced_voltage` computes `V = gain · dΦ/dt` — an
  explicit time derivative of the pickup flux (`spectral_derivative`).
- `noise.py` provides `tuned_/untuned_/matched_probe_output_noise_density`,
  i.e. **coil Johnson noise** `4kTR|H|²` plus amplifier terms, referred to the
  probe output (volts²/Hz), on the probe's `del_w` frequency grid.
- `optimal_control/control_response.py::ReceiverResponse` and
  `optimal_control/diffusion.py::detected_echo_snr` band-limit and score the
  detected echo through that same tuned/matched/untuned probe transfer.

All three encode the same two assumptions:

1. **The transducer differentiates** — the observable is `dΦ/dt`, so the signal
   voltage scales as `ω₀·M`. With a thermally polarized sample (`M ∝ B₀`) the
   detected signal scales as **`ω₀² ∝ B₀²`**. This is the quadratic collapse that
   makes ULF Faraday NMR/NQR hopeless and is the central incentive to raise `B₀`.
2. **The noise floor is the coil's Johnson noise**, which is frequency dependent
   and, referred to *field*, degrades toward low frequency.

SQUIDs and OPMs break **both** assumptions, so they cannot be modelled by tuning
a probe `Q` or swapping a `ReceiverResponse`. They need a first-class detector
abstraction.

## The physics the model must encode

| | Inductive (Faraday) | SQUID | OPM (atomic) |
|---|---|---|---|
| Observable | `dΦ/dt` (EMF) | flux `Φ` (field `B`) | field `B` (spin precession in the cell) |
| Signal vs field | `∝ ω₀·M ⇒ ∝ B₀²` (thermal) | `∝ M` (`∝ ω₀`, or **B₀-independent if prepolarized**) | `∝ M`, same as SQUID |
| Noise floor (field) | Johnson noise, **frequency dependent** | **flat** ~1 fT/√Hz (→0.2–0.4), + 1/f knee | **flat** 0.16–1 fT/√Hz, rolls off past atomic bandwidth |
| Bandwidth | Q-limited bandpass at `ω₀` | broadband (untuned flux transformer) | SERF ≲ kHz; RF/Mₓ tunable, ~10²–10³ Hz around a MHz-scale carrier |
| Cryogenics | none | liquid He (4.2 K) | none (room-temp vapour cell) |

Two consequences the reference exploits, which the model should reproduce:

- **Prepolarization decouples amplitude from `B₀`.** Polarize the spins in a
  strong `Bₚ`, then precess/detect in a weak `B₀`. The moment is set by `Bₚ`,
  independent of `B₀`. The package already has a `prepolarization` module and the
  `nonresonant` field-switching engine, so the preparation side exists.
- **Inhomogeneous linewidth scales as `B₀`.** `Δν' = (γ/2π)(ΔB₀/B₀)·B₀`, so at
  microtesla fields grossly inhomogeneous magnets still give ~1 Hz lines. Cheap
  magnets suffice, and SNR concentrates into narrow lines.

## Core principle: refer signal and noise to *field*

The clean unification is to stop working in volts and work in **magnetic field at
the sensor (T, T/√Hz)**. Then a detector is fully described by two frequency
functions:

- a **transfer** `H(f)` (dimensionless response shape), and
- a **field-referred noise PSD** `N²(f)` (T²/Hz) — the detector's own noise
  referred back through `H` to an equivalent field at the pickup.

and the matched-filter detected SNR of a field-at-sensor spectrum `S(f)` is

```
        ⌠  |S(f)|²
SNR² =  │  ─────── df                    (field-referred, detector-agnostic)
        ⌡   N²(f)
```

Why this is the right abstraction:

- `S(f)` — the field the precessing sample produces at the pickup — is the **same
  for every detector**. It depends on the moment (hence `Bₚ`) and geometry, *not*
  on the readout. All detector differences live in `N²(f)`.
- Faraday's `ω²` collapse is **not a special case** here: a coil's *field*
  sensitivity `N_coil(f) = √(4kTR)/|H_coil(f)|` rises toward low `f` (the tuned
  transfer `H` shrinks away from resonance), so `∫|S|²/N²` reproduces the `B₀²`
  scaling automatically. SQUID/OPM just have a **flat** `N²` instead.
- SNR is invariant to the reference: dividing both `S` and `N` by the same `|H|`
  (volts→field) leaves `SNR²` unchanged. So the inductive path can be re-expressed
  in field units and reproduce today's probe SNR **bit-for-bit** — the migration
  is provably lossless.
- The **tuned-coil-vs-magnetometer crossover** falls out: `N_coil(f)` (rising at
  low `f`) crosses the flat magnetometer floor at a few MHz. Below it,
  SQUID/OPM win; above it, tuned coils win. One plot, no special-casing.

`H(f)` is still needed to *derive* `N²(f)` for the inductive and OPM cases (both
have real transfer shape) and for any time-domain filtering of the detected
signal; it drops out of the pure SNR only because signal and detector noise share
it.

## Proposed package layout

A new subpackage `spin_dynamics/detection/`, sibling to `probes/` and `noise.py`,
following the compact instrument-facing style of `nqr/piezo_detection.py` (the
existing non-inductive-detector precedent):

```text
spin_dynamics/detection/
  base.py         # Detector protocol + field-referred matched-filter SNR core
    Detector          # protocol: transfer(freqs) -> H; field_noise_psd(freqs) -> N²
    DetectedFieldSNR  # result dataclass (snr, matched_filter, noise_rms, ...)
    detected_field_snr(field_spectrum, freqs, detector, ...)  # shared SNR
    field_referred_from_output(output_psd, transfer)          # N²_out/|H|²
  inductive.py    # InductiveCoilDetector — adapter over the existing probe
                  # noise densities; reproduces current numbers (regression guard)
  squid.py        # SQUIDMagnetometer — flat N² + 1/f knee, optional gradiometer
  opm.py          # OPMMagnetometer — SERF & RF/Mx modes, atomic-bandwidth rolloff
  gradiometer.py  # Gradiometer pickup geometry — spatial sensitivity via fields/
```

### Gradiometer pickup geometry

The detectors above describe the *frequency* response (`H`, `N²`); a **pickup
geometry** describes the *spatial* response — which sources couple in, and how
ambient noise is rejected. Clarke's SQUID system uses a second-derivative axial
gradiometer: a stack of coaxial loops wound so a spatially uniform field and its
first gradient cancel, leaving a response to `∂²B_z/∂z²`. By **reciprocity**, a
pickup's receive sensitivity to a source at `r` equals the field the winding
would produce at `r` per unit current (its B1 map) — so the gradiometer's
sensitive region is exactly the region it would "excite" as a transmitter.

`Gradiometer` holds coaxial loops (radii, axial positions, winding weights) and
computes the net sensitivity map `Σᵢ wᵢ B_loop,ᵢ(r)` by reusing `fields/coils.py`
loop fields. The physics to reproduce (Clarke §Experimental): the field, its
first, and its second derivative fall off from a dipole as `1/r³`, `1/r⁴`,
`1/r⁵`, so an order-`n` gradiometer's distant-source sensitivity falls as
`1/r^{3+n}` and it rejects uniform (and, at 2nd order, first-gradient) ambient
noise, while a nearby sample sits close to the bottom loop and couples strongly.
Presets: `magnetometer` (1 loop), `first_order_axial`, `second_order_axial`.

Then thread an **optional, backward-compatible** `detector=` through the existing
scorers (default `None` → today's inductive behavior, untouched):

- `optimal_control/diffusion.py::detected_echo_snr` / `make_pgse_objective`
- `noise.py::add_received_noise` / `estimate_matched_filter_snr`
- a detection-mode flag so flux detectors skip the `dΦ/dt` ω-scaling and let
  prepolarization set the amplitude.

## Detector models

### InductiveCoilDetector (adapter)

Wraps an existing output-referred probe density (`*_probe_output_noise_density`)
plus its Rx transfer `H`, and exposes them as `field_noise_psd = output/|H|²` and
`transfer = H`. Purpose: prove the field-referred core reproduces
`estimate_matched_filter_snr(...).predicted_snr` exactly, so nothing regresses.

### SQUIDMagnetometer

- Flat field-noise floor (default 1 fT/√Hz; presets to 0.4 and 0.2 per the
  reference's projected upgrades), plus an optional `1/f` knee.
- Broadband (untuned flux transformer) → `H ≈ 1` over the band; no resonant gain.
- Optional **gradiometer** spatial transfer (`1/rⁿ` rejection of distant
  sources), reusing `fields/` coil geometry for the pickup/baseline.
- Config-only attributes for context (bath temperature, slew/dynamic-range,
  flux-transformer turns) — not simulated dynamically in the first cut.

### OPMMagnetometer

- `SERF` mode: sub-fT/√Hz flat floor with a **low-pass atomic-response rolloff**
  (bandwidth ≲ kHz). Zero-field regime.
- `RF`/`Mₓ` mode: tunable Lorentzian centered on a carrier (the NQR/Larmor line)
  with ~10²–10³ Hz bandwidth; sub-fT/√Hz on resonance. **The rolloff is
  mandatory** — without it OPM SNR at MHz NQR is overstated.

## Phased plan (one PR each)

Scope decided with the user: **SQUID first**, validated against **Clarke's ¹H
ULF-MRI** numbers.

- **PR-1 — `detection/` core + inductive adapter.** `base.py` (protocol +
  field-referred SNR) and `inductive.py` reproducing current probe SNR. Pure
  additive layer; no existing code path changed. *(this PR)*
- **PR-2 — `SQUIDMagnetometer` + Clarke validation.** The headline PR: flat floor
  + 1/f knee + optional gradiometer, detection-mode flag dropping the Faraday
  ω-scaling, and a validation example reproducing (a) inductive `B₀²` vs SQUID
  flat-with-prepolarization → crossover figure, (b) `Δν ∝ B₀` narrowing
  (1.8 mT → 1.8 µT, Fig. 4), (c) ~1 fT/√Hz @ 5.6 kHz / 132 µT. Manual section +
  this doc updated.
- **PR-3 — `OPMMagnetometer`** (SERF + RF/Mₓ with atomic-bandwidth rolloff) and a
  ULF ¹⁴N NQR detection example. Designed-for now so the base protocol is stable.
- [x] **PR-3.5 — gradiometer pickup geometry.** `Gradiometer` (coaxial loops +
  winding weights) computing the spatial sensitivity / reciprocal excitation
  region via `fields.magnetostatics` (`circular_loop` + `biot_savart`); presets
  `magnetometer` / `first_order_axial` / `second_order_axial`;
  `examples/plot_gradiometer_sensitivity.py` mapping the excited-region
  localization + 1/r^{3+n} falloff; `tests/test_detection_gradiometer.py`
  (uniform-field nulling, falloff exponents 3/4/5, near-sample coupling).
- **PR-4 — detector-aware GRAPE objectives.** Optimize pulses / prepolarization
  for a flux readout instead of a tuned coil, over the existing ensemble path.

## Correctness invariants and tests

- **Reference invariance:** scaling `(S, N)` by a common `|H|` leaves `SNR`
  unchanged (volts ↔ field).
- **Inductive equivalence (bit-for-bit):** `detected_field_snr` with an
  `InductiveCoilDetector` equals `estimate_matched_filter_snr(...).predicted_snr`
  for matched inputs, to floating-point tolerance.
- **Flat-floor sanity:** a `SQUIDMagnetometer` with flat `N²` gives
  `SNR² = ∫|S|²/N² df` and is `B₀`-independent under prepolarization.
- **Crossover:** inductive field-noise crosses the flat magnetometer floor at the
  expected few-MHz scale.
- **OPM rolloff:** SNR degrades past the atomic bandwidth; SERF ≲ kHz vs RF
  tunable carrier behave distinctly.
- **Gradiometer:** a balanced order-`n` gradiometer nulls a uniform field (and,
  at 2nd order, a uniform gradient); its distant-source sensitivity falls as
  `1/r^{3+n}`; and its near-loop sensitivity localizes the excited region.
- **Backward compatible:** `detector=None` reproduces current results.

## Risks / subtleties

- **OPM bandwidth** must roll off or MHz-NQR SNR is overstated (SERF ≲ kHz; RF
  Lorentzian).
- **Reciprocity breaks** for magnetometers: Tx (coil) and Rx (SQUID/OPM) are
  independent, so the `rx = ReceiverResponse(tau=rf.tau)` idiom must not carry
  over — the receiver is not the transmit probe.
- **Sample/body Johnson noise** can dominate the detector floor at these
  sensitivities; expose it so SNR isn't unphysically optimistic.
- **1/f and environmental noise** (urban/geomagnetic) set the real ULF floor;
  gradiometry and shielding are what recover it (Clarke §Experimental).
- Keep everything an **opt-in additive layer** — zero change to default inductive
  behavior.

## Milestones

- [x] Add this design/progress document.
- [ ] **PR-1:** `detection/base.py` (Detector protocol, `detected_field_snr`,
      `field_referred_from_output`) + `detection/inductive.py`
      (`InductiveCoilDetector`) + `tests/test_detection.py`
      (invariance + inductive equivalence) + api-reference regen.
- [x] **PR-2:** `SQUIDMagnetometer` (flat + 1/f knee, presets) and
      `IdealFaradayCoil` (analytic 1/f baseline); `Detector.field_noise_amplitude`
      convenience; `examples/plot_squid_ulf_crossover.py` reproducing the
      noise-floor crossover, prepolarized SNR∝B₀^∓½ scalings, and linewidth∝B₀
      narrowing; `tests/test_detection_squid.py`; manual §Non-Inductive Detection.
      Note: no ω-scaling *flag* was needed — the field-referred core keeps the
      signal detector-independent and puts the ω dependence in the coil's 1/f
      field-noise, so thermal/prepolarized is just a choice of `S(f)`.
- [x] **PR-3:** `OPMMagnetometer` (SERF low-pass at DC + RF/Mₓ band-pass at a
      tuned carrier; field noise `N² = S0²[1 + ((f−f_c)/Δf)²]` — the mandatory
      atomic-bandwidth roll-off); `examples/plot_opm_nqr_detection.py` on a ¹⁴N
      line (RF-OPM vs SERF vs SQUID vs coil, + bandwidth-matching penalty);
      `tests/test_detection_opm.py`; manual §Non-Inductive Detection extended.
      Anchored to 0.24 fT/√Hz @ 423 kHz (Savukov / US 7,521,928); SERF rolled off
      ~f/Δf (≈2115×) at 423 kHz.
- [ ] **PR-4:** detector-aware GRAPE objective wiring over the ensemble path.

## Resolved design decisions

- **Field-referred, not volts.** The SNR core operates on a field-at-sensor
  spectrum with a field-referred noise PSD, making inductive detection one point
  on the same axis as SQUID/OPM rather than a separate code path.
- **Additive, opt-in.** New `detection/` subpackage; existing scorers gain an
  optional `detector=` argument defaulting to current behavior.
- **First detector / validation:** SQUID first, anchored to Clarke 2007 ¹H
  microtesla MRI/NMR (132 µT, 5.6 kHz, 1 fT/√Hz).
