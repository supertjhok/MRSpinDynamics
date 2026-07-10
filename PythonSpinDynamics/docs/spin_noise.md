# Spin Noise

This page records the physics analysis behind the spin-noise models, the
scrutiny of the previous ad-hoc approach, and the implementation plan. The
primary sources are three papers kept local-only in the repository
`References/` folder (gitignored):

- D. I. Hoult and N. S. Ginsberg, *The Quantum Origins of the Free Induction
  Decay Signal and Spin Noise*, J. Magn. Reson. 148, 182-199 (2001).
- T. R. Field and A. D. Bain, *Origins of Spin Noise*, Appl. Magn. Reson. 38,
  167-178 (2010).
- R. Annabestani, D. G. Cory and J. Emerson, *Quantum Model of Spin Noise*,
  J. Magn. Reson. 252, 94-102 (2015).

## Physics summary

The three papers converge on one observable model:

1. **Origin.** Spin noise is near-field Faraday induction from stochastic
   transverse magnetization fluctuations of the whole spin ensemble. It is not
   spontaneous emission and does not require a high-Q resonator (Hoult observed
   it at Q of order 1).
2. **Amplitude.** The RMS spin-noise EMF is `xi_rms = w0 * B1_hat * m * sqrt(N)`
   with `m = gamma * hbar / 2` and `B1_hat` the coil field per unit current
   (Hoult Eq. 21). It scales as `sqrt(N)` of **all** spins, not the Boltzmann
   excess, so its relative importance grows for small samples and low
   polarization.
3. **Spectrum.** The fluctuations are a stationary, zero-mean Gaussian process
   with exponential autocorrelation at time constant `T2` (Field Eq. 7;
   Annabestani Eq. 15, `R(k) = (N/4) exp(-t/T)`), hence a Lorentzian PSD of
   FWHM `1/(pi*T2)` Hz centered on the Larmor frequency. `T1` plays no role at
   equilibrium.
4. **Fluctuation-dissipation.** The same coupling makes the sample present an
   impedance in series with the coil (Hoult Eq. 17):

   ```text
   Z_n(dw) = R_n0 * (1 - j*dw*T2) / (1 + (dw*T2)^2)
   R_n0    = 0.5 * w0 * gamma * B1_hat^2 * T2 * M0 * V_s
   ```

   The spin-noise EMF is exactly the Nyquist noise of `R_n` at the **sample**
   temperature, `S(w) = 4*k*T_s*R_n(dw)`, while the coil resistance contributes
   `4*k*T_c*r_c` at the **coil** temperature. Bump-versus-dip phenomenology and
   radiation-damping broadening of the noise line follow from inserting `Z_n`
   into the receiver circuit: the sample both emits its own noise and absorbs
   circuit noise, so the sign of the on-resonance feature tracks the sample
   temperature relative to the effective circuit temperature.
5. **Radiation-damping identity.** With the package convention
   `1/Trd = 0.5 * mu0 * gamma * M0 * Q * eta` and the solenoid relation
   `B1_hat^2 = mu0 * L / (2 * V_c)`, the on-resonance sample resistance obeys

   ```text
   R_n0 = mu0 * eta * gamma * M0 * T2 * w0 * L / 4 = R_coil * T2 / (2 * Trd)
   ```

   so the frequency-domain circuit model (Option B) and the time-domain
   stochastic source (Option C) share one coupling constant with the existing
   deterministic radiation-damping module. The circuit-loaded spin-noise
   linewidth is `1/T2 + 1/Trd` for a tuned probe on resonance.
6. **Quantum versus semiclassical.** Annabestani et al. derive the same
   covariance from repeated collective quantum measurement with no stochastic
   field at all; for the observable spectrum of a weakly polarized ensemble the
   semiclassical Ornstein-Uhlenbeck description is exact. Only
   measurement-back-action studies (weak versus strong readout) need the full
   quantum measurement chain, which stays out of package scope.

## Scrutiny of the previous approach

Before this work the package had no spin-noise source at all.
`spin_dynamics.noise` is a received-signal layer: white Gaussian noise with a
free `sigma`/`target_snr`, or Johnson-plus-amplifier noise colored by the
tuned/untuned/matched circuit. Measured against the papers this fails on four
counts:

1. **Wrong spectrum** - spin noise is a Lorentzian of width `1/(pi*T2)` riding
   on the flat coil/amplifier baseline; white or probe-colored noise is smooth
   across the line.
2. **No physical scale** - the amplitude should be computed from
   `w0 * B1_hat * (gamma*hbar/2) * sqrt(N)`, not chosen by hand; the package
   already carries every needed ingredient (`M0` helpers, fill factor, `Q`,
   coil geometry).
3. **No fluctuation-dissipation link** - the sample resistance that emits the
   noise also loads the circuit and absorbs circuit noise; additive noise
   applied after the transfer function can never produce the observed
   on-resonance dips.
4. **No back-action** - output-only noise never enters the spin dynamics, so
   e.g. `simulate_nmr_maser` had to be seeded with an arbitrary deterministic
   tip instead of by spin noise.

## Modeling options considered

- **A. Physically scaled colored additive noise.** A Lorentzian output PSD with
  the correct amplitude pushed through the existing probe transfer functions.
  Cheap, but no back-action, no dips, no radiation-damping broadening.
  Superseded by B, which costs little more.
- **B. Sample impedance in the circuit model (implemented).** Insert Hoult's
  `Z_n(dw)` in series with the coil inside the tuned/untuned/matched
  output-noise densities, with two Nyquist sources at their own temperatures
  (`4*k*T_c*r_c` for the coil, `4*k*T_s*R_n` for the sample). Bump/dip,
  frequency pulling from `X_n`, and radiation-damping linewidth broadening all
  emerge from the circuit algebra. Stationary and linear; ideal for CW
  spin-noise spectra and SNR budgets.
- **C. Stochastic source-level model (implemented).** A complex
  Ornstein-Uhlenbeck Langevin term on the transverse magnetization,

  ```text
  dm+ = (j*dw - 1/T2) m+ dt - [RD back-action] dt + sigma_s dxi_s + sigma_c dxi_c
  sigma_s^2 = (2/T2)  * m_rms^2            (spin bath at T_s)
  sigma_c^2 = (2/Trd) * m_rms^2 * (T_c/T_s) (coil channel at T_c)
  ```

  with normalized stationary RMS `m_rms = 1/(sqrt(N)*p)` (`p` the thermal
  polarization; `M0*V_s = N*(gamma*hbar/2)*p` makes this Hoult's
  `sqrt(N)*(gamma*hbar/2)` in absolute units), coupled into the existing
  radiation-damping feedback state. The coil-channel realization also appears
  (correlated) in the receiver output, which is what produces the equilibrium
  bump/dip physics in the time domain. Gives back-action, RD-broadened lines,
  spin-noise-seeded maser start-up, and pulsed spin-noise experiments.
- **D. Quantum measurement chain (not implemented).** Annabestani's iterated
  weak/strong collective measurement. Only needed for readout-back-action
  studies; out of package scope per the user manual boundary.

## Implementation plan (status)

1. **Docs page** - this file.
2. **Sample properties layer** (`spin_dynamics.sample`) - geometry/dimensions,
   spin density, temperature, nucleus, relaxation times; derived spin count,
   polarization, Curie-law magnetization density, and fluctuation scales.
   Cleanly separated from detector/coil properties, which may sit at a
   different temperature (`sp.T` remains the coil/circuit temperature; the
   sample temperature lives on the sample object).
3. **Option B** (`spin_dynamics.spin_noise.SampleCoupling` plus `sample=`
   keyword on the `*_probe_output_noise_density` functions in
   `spin_dynamics.noise`) - validated against the `R_n0 = R*T2/(2*Trd)`
   identity, Lorentzian shape, bump/dip temperature crossover, and linewidth
   broadening `1/T2 + 1/Trd`.
4. **Option C** (`spin_dynamics.spin_noise.simulate_spin_noise` and helpers) -
   validated against the stationary variance, the analytic autocorrelation
   `R(t) = m_rms^2 * exp(-t/T2)`, the Lorentzian PSD, RD broadening, and
   cross-checked against Option B in the linear regime.

## Boundaries and caveats

- The matched-probe Option B variant treats `Z_n` as a series perturbation of
  the already-matched source (voltage-divider factor `2*Rc/(2*Rc + Z_n)`); it
  does not re-derive the matching network, so it is quantitative for
  `R_n0 << Rc` and indicative beyond that.
- Option C is semiclassical and linear in the noise terms; it reproduces the
  quantum covariance for weakly polarized ensembles (Annabestani Sec. 3.1) but
  does not model measurement-strength effects.
- The deterministic radiation-damping path documented in
  [radiation_damping.md](radiation_damping.md) is unchanged; the stochastic
  model is opt-in and reduces to it when the noise amplitudes are zero.
