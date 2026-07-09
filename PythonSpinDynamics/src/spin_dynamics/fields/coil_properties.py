"""Lumped RF properties of single-layer helical round-wire solenoids.

Turns a solenoid's geometry into the properties a probe designer needs -- series
inductance, AC (skin + proximity) resistance, stray capacitance, unloaded quality
factor and self-resonant frequency -- so a coil can be handed straight to the probe
noise/SNR model (:func:`spin_dynamics.noise.tuned_probe_output_noise_density`, which
consumes ``L``, ``R``, ``C``) and the matching-network designer
(:mod:`spin_dynamics.probes.matched`).

The model is the ``n = 0`` sheath-helix waveguide method of Corum, with the NIST
round-wire and field-non-uniformity correction factors (Knight/Rosa/Lundin) and the
Medhurst proximity data. Unlike a purely quasistatic inductance it stays accurate for
electrically long coils, where transmission-line effects raise the effective inductance
toward the first (quarter-wave) self-resonance.

This is a faithful port of Serge Y. Stroobandt's QOIL calculator
(https://hamwaves.com/qoil/, GPL-3.0), vendored at
``References/hamwaves_inductance_calculator/``. The scalar per-frequency ``math``
pipeline is re-expressed with :mod:`scipy.special` Bessel functions and
:func:`scipy.optimize.brentq` root finds. :func:`solenoid_field_inductance` provides an
independent field-based inductance (Biot-Savart / Neumann, via
:func:`spin_dynamics.fields.quasistatic.coil_inductance`) that agrees with the
current-sheet inductance at low frequency.

**Scope / staging.** This covers single-layer air-core solenoids -- the geometry with a
validated semi-analytical reference. Field-based extraction of proximity-effect
resistance (a 2-D wire-cross-section eddy solve) and turn-to-turn capacitance (an
electrostatic solve) for arbitrary coil geometries is a planned follow-on that will slot
in alongside :func:`solenoid_properties`.

References
----------
* J. F. Corum & K. L. Corum, "RF Coils, Helical Resonators and Voltage Magnification by
  Coherent Spatial Modes," Microwave Review (IEEE), 2001.
* D. W. Knight, G3YNH, "Solenoid inductance calculation with emphasis on radio-frequency
  applications," https://g3ynh.info/ (round-wire ``k_s``/``k_m`` corrections).
* R. Lundin, "A handbook formula for the inductance of a single-layer circular coil,"
  Proc. IEEE 73(9), 1985 (non-uniformity ``k_L``).
* R. G. Medhurst, "H.F. Resistance and Self-Capacitance of Single-Layer Solenoids,"
  Wireless Engineer, 1947 (proximity factor ``Phi``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import i0, i1, k0, k1

from spin_dynamics.fields.magnetostatics import MU0

C_0 = 299792458.0  # speed of light (m/s)

__all__ = [
    "ConductorMaterial",
    "ANNEALED_COPPER",
    "HARD_DRAWN_COPPER",
    "SILVER",
    "ALUMINIUM",
    "CoilProperties",
    "medhurst_proximity_factor",
    "sheath_helix_dispersion",
    "solenoid_properties",
    "solenoid_field_inductance",
]


@dataclass(frozen=True)
class ConductorMaterial:
    """A conductor's RF-relevant material constants, with a temperature model.

    ``resistivity`` is the DC electrical resistivity (ohm*m) at ``reference_temperature``
    (K) and ``mu_r`` the relative permeability (near 1 for the diamagnetic/weakly-
    paramagnetic plating metals). The reference values follow the QOIL plating table
    (293.15 K).

    ``temp_coefficient`` is the linear temperature coefficient of resistivity
    ``alpha`` (1/K) about the reference temperature. ``residual_resistivity_ratio``
    (RRR ``= rho(293 K) / rho(~0 K)``) is optional and, when set, switches
    :meth:`resistivity_at` to a Matthiessen model with a residual floor -- required for
    cryogenic use, where the linear model would go negative.
    """

    name: str
    resistivity: float
    mu_r: float = 1.0
    temp_coefficient: float = 0.0
    reference_temperature: float = 293.15
    residual_resistivity_ratio: float | None = None

    def resistivity_at(self, temperature: float | None = None) -> float:
        """Resistivity (ohm*m) at ``temperature`` (K); reference value if ``None``.

        With no ``residual_resistivity_ratio`` a linear coefficient
        ``rho(T) = rho_ref [1 + alpha (T - T_ref)]`` is used -- accurate within roughly
        +/-100 K of the reference but unphysical (it would go negative) at cryogenic
        temperatures. With an RRR, Matthiessen's rule adds a residual floor
        ``rho_res = rho_ref / RRR`` to a linear (high-temperature Bloch-Grueneisen limit)
        phonon term, valid from cryogenic up through room temperature; the phonon term
        over-predicts the ideal resistivity well below ~Debye/3 (e.g. below ~110 K for
        copper), so it is a conservative (upper-bound) estimate of the cryogenic loss.
        """

        if temperature is None:
            return float(self.resistivity)
        t = float(temperature)
        if t < 0.0:
            raise ValueError("temperature must be non-negative (kelvin)")
        if self.residual_resistivity_ratio is None:
            rho = self.resistivity * (1.0 + self.temp_coefficient * (t - self.reference_temperature))
        else:
            rho_res = self.resistivity / self.residual_resistivity_ratio
            rho_phonon_ref = self.resistivity - rho_res
            rho = rho_res + rho_phonon_ref * (t / self.reference_temperature)
        return float(max(rho, 1e-13))


# Plating presets (resistivity ohm*m at 293.15 K, mu_r, linear temp coefficient 1/K) --
# QOIL `plating` table extended with textbook temperature coefficients.
ANNEALED_COPPER = ConductorMaterial("annealed copper", 17.241e-9, 0.99999044, temp_coefficient=3.93e-3)
HARD_DRAWN_COPPER = ConductorMaterial("hard-drawn copper", 17.71e-9, 0.99999044, temp_coefficient=3.93e-3)
SILVER = ConductorMaterial("silver", 15.9e-9, 0.9999738, temp_coefficient=3.8e-3)
ALUMINIUM = ConductorMaterial("aluminium", 28.24e-9, 1.00002212, temp_coefficient=4.03e-3)


# --- Medhurst proximity-effect data ----------------------------------------------------
# Rows indexed by l/D, columns by p/d (winding pitch / wire diameter). The final
# 1e31 sentinel row/column extend the table to the asymptotic (infinitely long / widely
# spaced) limit for clamped bilinear interpolation.
_MEDHURST_L_OVER_D = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1e31])
_MEDHURST_P_OVER_D = np.array([1.0, 1.111, 1.25, 1.429, 1.667, 2.0, 2.5, 3.333, 5.0, 10.0, 1e31])
_MEDHURST = np.array([
    [5.31, 3.73, 2.74, 2.12, 1.74, 1.44, 1.20, 1.16, 1.07, 1.02, 1.00],
    [5.45, 3.84, 2.83, 2.20, 1.77, 1.48, 1.29, 1.19, 1.08, 1.02, 1.00],
    [5.65, 3.99, 2.97, 2.28, 1.83, 1.54, 1.33, 1.21, 1.08, 1.03, 1.00],
    [5.80, 4.11, 3.10, 2.38, 1.89, 1.60, 1.38, 1.22, 1.10, 1.03, 1.00],
    [5.80, 4.17, 3.20, 2.44, 1.92, 1.64, 1.42, 1.23, 1.10, 1.03, 1.00],
    [5.55, 4.10, 3.17, 2.47, 1.94, 1.67, 1.45, 1.24, 1.10, 1.03, 1.00],
    [4.10, 3.36, 2.74, 2.32, 1.98, 1.74, 1.50, 1.28, 1.13, 1.04, 1.00],
    [3.54, 3.05, 2.60, 2.27, 2.01, 1.78, 1.54, 1.32, 1.15, 1.04, 1.00],
    [3.31, 2.92, 2.60, 2.29, 2.03, 1.80, 1.56, 1.34, 1.16, 1.04, 1.00],
    [3.20, 2.90, 2.62, 2.34, 2.08, 1.81, 1.57, 1.34, 1.165, 1.04, 1.00],
    [3.23, 2.93, 2.65, 2.27, 2.10, 1.83, 1.58, 1.35, 1.17, 1.04, 1.00],
    [3.41, 3.11, 2.815, 2.51, 2.22, 1.93, 1.65, 1.395, 1.19, 1.05, 1.00],
])


def _lower_index(header: np.ndarray, value: float) -> int:
    """Lower bracketing index of ``value`` in the ascending ``header`` (QOIL logic)."""

    index2 = 0
    n = header.size
    while value >= header[index2] and index2 < n - 1:
        index2 += 1
    return index2 - 1 if index2 > 0 else 0


def medhurst_proximity_factor(l_over_D: float, p_over_d: float) -> float:
    """Medhurst proximity factor ``Phi`` from the ``(l/D, p/d)`` table.

    ``Phi`` is the ratio of AC-to-uniform-current resistance caused by the proximity
    effect between adjacent turns; it multiplies the skin-effect wire resistance.
    ``l/D`` is the coil length/diameter ratio and ``p/d`` the pitch/wire-diameter ratio.
    Values are clamped to the tabulated range and bilinearly interpolated, matching the
    QOIL implementation.
    """

    i = _lower_index(_MEDHURST_L_OVER_D, float(l_over_D))
    j = _lower_index(_MEDHURST_P_OVER_D, float(p_over_d))
    x1, x2 = _MEDHURST_L_OVER_D[i], _MEDHURST_L_OVER_D[i + 1]
    y1, y2 = _MEDHURST_P_OVER_D[j], _MEDHURST_P_OVER_D[j + 1]
    # Interpolate in l/D first (down the two flanking p/d columns), then in p/d.
    t = (float(l_over_D) - x1) / (x2 - x1)
    phi_j1 = _MEDHURST[i, j] + t * (_MEDHURST[i + 1, j] - _MEDHURST[i, j])
    phi_j2 = _MEDHURST[i, j + 1] + t * (_MEDHURST[i + 1, j + 1] - _MEDHURST[i, j + 1])
    u = (float(p_over_d) - y1) / (y2 - y1)
    return float(phi_j1 + u * (phi_j2 - phi_j1))


def _dispersion_residual(tau: float, a: float, k_0: float, tan_psi: float) -> float:
    """Sheath-helix dispersion residual ``F(tau)`` (zero at the guided mode)."""

    ta = tau * a
    return k1(ta) * i1(ta) / (k0(ta) * i0(ta)) - (tau / k_0 * tan_psi) ** 2


def sheath_helix_dispersion(
    frequency: float, a: float, psi: float
) -> tuple[float, float, float]:
    """Solve the ``n = 0`` sheath-helix dispersion at ``frequency``.

    Returns ``(beta, tau, Z_c)``: the axial propagation factor ``beta`` (rad/m), the
    radial decay constant ``tau`` (rad/m) and the characteristic impedance ``Z_c`` (ohm)
    of the guided helical-waveguide mode. ``a`` is the effective coil radius (m) and
    ``psi`` the pitch angle (rad). Raises ``RuntimeError`` if the transcendental root is
    not bracketed (which happens as the coil approaches a resonance).
    """

    omega = 2.0 * np.pi * float(frequency)
    k_0 = omega / C_0
    tan_psi = np.tan(psi)
    cot2 = 1.0 / np.tan(psi) ** 2
    # QOIL brackets tau between k_0 (smallest) and k_0 * cot(psi)^2 (largest).
    lo = k_0 * (1.0 + 1e-9)
    hi = k_0 * cot2
    f_lo = _dispersion_residual(lo, a, k_0, tan_psi)
    f_hi = _dispersion_residual(hi, a, k_0, tan_psi)
    if f_lo == 0.0:
        tau = lo
    elif f_hi == 0.0:
        tau = hi
    elif f_lo * f_hi > 0.0:
        raise RuntimeError("sheath-helix dispersion root is not bracketed")
    else:
        tau = brentq(_dispersion_residual, lo, hi, args=(a, k_0, tan_psi), xtol=1e-12, rtol=1e-12)
    beta = np.sqrt(k_0**2 + tau**2)
    z_c = 60.0 * beta / k_0 * i0(tau * a) * k0(tau * a)
    return float(beta), float(tau), float(z_c)


def _beta_l(frequency: float, a: float, psi: float, length: float) -> float:
    beta, _, _ = sheath_helix_dispersion(frequency, a, psi)
    return beta * length


def _self_resonant_frequency(
    a: float, psi: float, length: float, wire_length_eff: float
) -> float:
    """First (quarter-wave) self-resonance, where ``beta(f) * l = pi/2``.

    ``beta`` increases monotonically with frequency, so ``g(f) = beta*l - pi/2`` has a
    single root; it is bracketed starting from QOIL's ``c_0 / l_w_eff`` scale and solved
    with :func:`scipy.optimize.brentq`. Returns ``nan`` if a bracket cannot be found.
    """

    target = np.pi / 2.0

    def g(f: float) -> float:
        return _beta_l(f, a, psi, length) - target

    f_lo = C_0 / wire_length_eff / 40.0
    f_hi = f_lo * 100.0
    try:
        g_lo = g(f_lo)
        # Expand upward until the sign changes (or give up).
        g_hi = g(f_hi)
        tries = 0
        while g_lo * g_hi > 0.0 and tries < 60:
            f_lo, g_lo = f_hi, g_hi
            f_hi *= 1.5
            g_hi = g(f_hi)
            tries += 1
        if g_lo * g_hi > 0.0:
            return float("nan")
        return float(brentq(g, f_lo, f_hi, xtol=1.0, rtol=1e-10))
    except (RuntimeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class CoilProperties:
    """Lumped RF properties of a single-layer solenoid at a design frequency.

    All quantities are SI. The ``inductance_currentsheet`` (``L_s``) is the
    frequency-independent geometric inductance; ``inductance_effective`` (``L_eff``) is
    the sheath-helix value at ``frequency`` (equal to ``L_s`` when electrically short,
    rising toward ``self_resonant_frequency``). Effective-circuit fields are ``nan`` when
    the dispersion solve does not converge near resonance.
    """

    # Inputs (echoed)
    diameter: float
    length: float
    turns: int
    wire_diameter: float
    frequency: float
    material: ConductorMaterial
    temperature: float
    resistivity: float
    # Geometry / intermediates
    pitch: float
    proximity_phi: float
    effective_diameter: float
    pitch_angle: float
    k_L: float
    k_s: float
    k_m: float
    skin_depth: float
    wire_length_physical: float
    wire_length_effective: float
    beta: float
    characteristic_impedance: float
    # Results
    inductance_currentsheet: float
    inductance_effective: float
    reactance_effective: float
    ac_resistance: float
    q_factor: float
    stray_capacitance: float
    self_resonant_frequency: float

    @property
    def inductance(self) -> float:
        """Best single-value inductance: ``L_eff`` if available, else ``L_s``."""

        return self.inductance_effective if np.isfinite(self.inductance_effective) else self.inductance_currentsheet

    def tuning_capacitance(self, frequency: float | None = None) -> float:
        """Capacitance that resonates the coil at ``frequency`` (default: design ``f``).

        ``C = 1 / (omega^2 L)`` with ``L`` the effective inductance -- the tank
        capacitance for a lumped LC match, ignoring the coil's own stray ``C``.
        """

        f = self.frequency if frequency is None else float(frequency)
        omega = 2.0 * np.pi * f
        return float(1.0 / (omega**2 * self.inductance))

    def to_probe_params(self) -> dict[str, float]:
        """Return ``{"L", "R", "C"}`` for the probe noise model.

        ``L``/``R`` are the effective series inductance and AC resistance; ``C`` is the
        tuning capacitance at the design frequency. These seed the ``sp`` mapping consumed
        by :func:`spin_dynamics.noise.tuned_probe_output_noise_density` (which needs the
        remaining amplifier/receiver fields supplied separately).
        """

        return {
            "L": self.inductance,
            "R": self.ac_resistance,
            "C": self.tuning_capacitance(),
        }


def solenoid_properties(
    *,
    diameter: float,
    length: float,
    turns: int,
    wire_diameter: float,
    frequency: float,
    material: ConductorMaterial = ANNEALED_COPPER,
    temperature: float | None = None,
) -> CoilProperties:
    """Extract the lumped RF properties of a single-layer round-wire solenoid.

    ``diameter`` is the mean coil diameter (conductor centre to centre, m), ``length``
    the coil length (m), ``turns`` the number of turns, ``wire_diameter`` the wire/tubing
    diameter (m), ``frequency`` the design frequency (Hz) and ``material`` the conductor.
    ``temperature`` (K) sets the operating temperature for the conductor resistivity via
    :meth:`ConductorMaterial.resistivity_at`; ``None`` uses the material's reference
    (room) temperature. Only the AC resistance (hence ``Q``) depends on temperature --
    the inductance, capacitance and self-resonance are geometric.

    Implements the Corum sheath-helix model with the Knight/Rosa/Lundin round-wire and
    non-uniformity corrections and Medhurst proximity data (see module docstring). The
    effective-circuit results (``L_eff``, ``Q``, stray ``C``) are ``nan`` when the coil is
    at/near a self-resonance where the dispersion solve degenerates; ``L_s`` and the AC
    resistance are always returned.
    """

    D = float(diameter)
    length = float(length)
    N = int(turns)
    d = float(wire_diameter)
    f = float(frequency)
    if D <= 0.0 or length <= 0.0 or d <= 0.0 or f <= 0.0:
        raise ValueError("diameter, length, wire_diameter and frequency must be positive")
    if N < 1:
        raise ValueError("turns must be at least 1")
    if d >= D:
        raise ValueError("wire_diameter must be smaller than the coil diameter")

    p = length / N
    if d > p:
        raise ValueError("wire_diameter cannot exceed the winding pitch (turns would overlap)")

    # Proximity factor and effective diameter.
    phi = medhurst_proximity_factor(length / D, p / d)
    d_eff = D - d * (1.0 - 1.0 / np.sqrt(phi))

    # Correction factors: Lundin non-uniformity (short/long branch), Rosa self-L, Knight
    # mutual-L (series in 1/N).
    if length <= d_eff:
        r = length / d_eff
        k_l = (1.0 + 0.383901 * r**2 + 0.017108 * r**4) / (1.0 + 0.258952 * r**2)
        k_l *= np.log(4.0 * d_eff / length) - 0.5
        k_l += 0.093842 * r**2 + 0.002029 * r**4 - 0.000801 * r**6
        k_l *= 2.0 / np.pi * r
    else:
        r = d_eff / length
        k_l = (1.0 + 0.383901 * r**2 + 0.017108 * r**4) / (1.0 + 0.258952 * r**2)
        k_l -= 4.0 / 3.0 / np.pi * r

    k_s = 5.0 / 4.0 - np.log(2.0 * p / d)
    c_9 = -np.log(2.0 * np.pi) + 3.0 / 2.0 + 0.33084236 + 1.0 / 120.0 - 1.0 / 504.0 + 0.0011925
    nn = float(N)
    k_m = (
        np.log(2.0 * np.pi) - 3.0 / 2.0 - np.log(nn) / 6.0 / nn - 0.33084236 / nn
        - 1.0 / (120.0 * nn**3) + 1.0 / (504.0 * nn**5) - 0.0011925 / nn**7 + c_9 / nn**9
    )

    # Wire lengths and skin depth (resistivity at the operating temperature).
    rho = material.resistivity_at(temperature)
    l_w_phys = np.sqrt((nn * np.pi * D) ** 2 + length**2)
    l_w_eff = np.sqrt((nn * np.pi * d_eff) ** 2 + length**2)
    delta = np.sqrt(rho / (np.pi * f * MU0 * material.mu_r))

    # Effective series AC resistance (skin annulus * proximity, current-sharing (N-1)/N).
    r_ac = rho * l_w_eff / (np.pi * (d * delta - delta**2)) * phi
    if N > 1:
        r_ac *= (nn - 1.0) / nn

    # Current-sheet geometric inductance (frequency independent).
    l_s = MU0 * (np.pi * (d_eff * nn) ** 2 / (4.0 * length) * k_l - d_eff * nn * (k_s + k_m) / 2.0)

    psi = np.arctan(p / (np.pi * d_eff))
    a = d_eff / 2.0
    omega = 2.0 * np.pi * f

    # Sheath-helix effective circuit (degrades gracefully near resonance).
    try:
        beta, _, z_c = sheath_helix_dispersion(f, a, psi)
        l_eff = z_c / omega * np.tan(beta * length) * k_l - MU0 * d_eff * nn * (k_s + k_m) / 2.0
        x_eff = omega * l_eff
        q_eff = x_eff / r_ac
        c_p = _stray_capacitance(l_s, r_ac, x_eff, q_eff, omega)
    except (RuntimeError, ValueError):
        beta = float("nan")
        z_c = float("nan")
        l_eff = float("nan")
        x_eff = float("nan")
        q_eff = float("nan")
        c_p = float("nan")

    f_res = _self_resonant_frequency(a, psi, length, l_w_eff)

    return CoilProperties(
        diameter=D,
        length=length,
        turns=N,
        wire_diameter=d,
        frequency=f,
        material=material,
        temperature=float(material.reference_temperature if temperature is None else temperature),
        resistivity=float(rho),
        pitch=float(p),
        proximity_phi=float(phi),
        effective_diameter=float(d_eff),
        pitch_angle=float(psi),
        k_L=float(k_l),
        k_s=float(k_s),
        k_m=float(k_m),
        skin_depth=float(delta),
        wire_length_physical=float(l_w_phys),
        wire_length_effective=float(l_w_eff),
        beta=float(beta),
        characteristic_impedance=float(z_c),
        inductance_currentsheet=float(l_s),
        inductance_effective=float(l_eff),
        reactance_effective=float(x_eff),
        ac_resistance=float(r_ac),
        q_factor=float(q_eff),
        stray_capacitance=float(c_p),
        self_resonant_frequency=float(f_res),
    )


def _stray_capacitance(
    l_s: float, r_ac: float, x_eff: float, q_eff: float, omega: float
) -> float:
    """Parallel stray capacitance ``C_p`` of the lumped equivalent circuit (QOIL algebra).

    Matches the effective and lumped equivalent circuits at the design frequency by
    solving for the parallel capacitance that reconciles the frequency-independent
    ``L_s`` with the frequency-dependent ``L_eff``.
    """

    r_p = (q_eff**2 + 1.0) * r_ac
    x_l_s = omega * l_s
    pp = r_p / (2.0 * x_l_s)
    q_l = pp + np.sqrt(pp**2 - 1.0)
    x_eff_p = (q_eff**2 + 1.0) / q_eff**2 * x_eff
    x_l_p = (q_l**2 + 1.0) / q_l**2 * x_l_s
    x_c_p = x_eff_p * x_l_p / (x_l_p - x_eff_p)
    return float(-1.0 / (omega * x_c_p))


def solenoid_field_inductance(
    *,
    diameter: float,
    length: float,
    turns: int,
    wire_diameter: float,
    n_segments: int = 120,
) -> float:
    """Independent field-based inductance of the solenoid (Biot-Savart / Neumann).

    Builds the ``turns`` coaxial loops and sums the self/mutual partial inductances via
    :func:`spin_dynamics.fields.quasistatic.coil_inductance`. This is a purely
    quasistatic (DC) value and should agree with
    :attr:`CoilProperties.inductance_currentsheet` at low frequency, serving as a
    solver-independent cross-check of the semi-analytical model.
    """

    from spin_dynamics.fields.quasistatic import coil_inductance

    N = int(turns)
    if N < 1:
        raise ValueError("turns must be at least 1")
    radius = float(diameter) / 2.0
    wire_radius = float(wire_diameter) / 2.0
    if N == 1:
        centers = np.zeros((1, 3))
    else:
        offsets = np.linspace(-float(length) / 2.0, float(length) / 2.0, N)
        centers = np.zeros((N, 3))
        centers[:, 2] = offsets
    radii = np.full(N, radius)
    return float(
        coil_inductance(radii, centers, wire_radius=wire_radius, axis="z", n_segments=n_segments)
    )
