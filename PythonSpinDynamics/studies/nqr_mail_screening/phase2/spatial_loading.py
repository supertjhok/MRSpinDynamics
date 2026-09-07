"""Spatial loading bounds and a tuned circuit using built-in coil/eddy solvers.

The dielectric envelope is a variational bound, not a solved heterogeneous E map.
Complex dielectric contrast is bounded; magnetic reaction remains Born order.
High-loss/metal parcels outside the declared domain are rejected.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from spin_dynamics.fields.coil_peec import _potential_coefficient_matrix
from spin_dynamics.fields.quasistatic import vector_potential

EPS0 = 8.8541878128e-12
KB = 1.380649e-23


class ElectrostaticReference:
    """PEEC winding charge distribution for a 1 V single-ended winding ramp."""

    def __init__(self, conductor):
        self.conductor = conductor
        p, lengths = _potential_coefficient_matrix(conductor)
        potential = (np.cumsum(lengths) - lengths / 2) / sum(lengths)
        self.charge = np.linalg.solve(p, potential)
        self.capacitance = float(self.charge @ potential)
        self.mids = (conductor.path_points[:-1] + conductor.path_points[1:]) / 2

    def field(self, points):
        # Same midpoint charge kernel as the PEEC off-diagonal potential matrix.
        flat = np.asarray(points).reshape(-1, 3)
        result = np.empty_like(flat)
        for start in range(0, len(flat), 2048):
            displacement = flat[start : start + 2048, None, :] - self.mids[None, :, :]
            distance = np.linalg.norm(displacement, axis=-1)
            if np.min(distance) < 2 * self.conductor.wire_radius:
                raise ValueError(
                    "dielectric voxel too close to a wire for the charge model"
                )
            result[start : start + 2048] = np.einsum(
                "njk,j,nj->nk", displacement, self.charge, distance**-3
            ) / (4 * np.pi * EPS0)
        return result.reshape(np.asarray(points).shape)


def dielectric_bounds(energy_integral, epsilon_r, conductivity, frequency):
    """Positive spectral-measure bounds for a uniform complex contrast.

    At depolarization eigenvalue t in [0,1], the complex response is
    chi/(1+t*chi), chi=eps_r-1-i*sigma/(omega*eps0). Its extremal real
    part bounds delta C; sigma/|1+t*chi|^2 bounds conductance. The measure
    integrates to integral_support |E_vac|^2 for the fixed winding drive.
    """
    values = (energy_integral, epsilon_r, conductivity, frequency)
    if (
        not np.all(np.isfinite(values))
        or epsilon_r < 1
        or conductivity < 0
        or energy_integral < 0
        or frequency <= 0
    ):
        raise ValueError("invalid passive material")
    a = epsilon_r - 1
    b = conductivity / (2 * np.pi * frequency * EPS0)
    # The declared dry-mail study domain; more conductive contents need a new gate.
    if b > 1:
        raise ValueError("conductive parcel exceeds the declared dielectric domain")
    points = [0.0, 1.0]
    if a * a + b * b > 0:
        critical = (b - a) / (a * a + b * b)
        if 0 < critical < 1:
            points.append(critical)
    response = [((a - 1j * b) / (1 + t * (a - 1j * b))).real for t in points]
    return {
        "delta_capacitance_f": [
            EPS0 * energy_integral * min(response),
            EPS0 * energy_integral * max(response),
        ],
        "shunt_conductance_s": [
            conductivity * energy_integral / (epsilon_r**2 + b * b),
            conductivity * energy_integral,
        ],
        "loss_parameter_sigma_over_omega_eps0": b,
    }


def total_loss_bounds(electric_conductance, eddy_resistance, wire_impedance):
    """Cauchy bounds include interference between capacitive and inductive E."""
    lo, hi = np.sqrt(electric_conductance)
    eddy = np.sqrt(eddy_resistance) / abs(wire_impedance)
    lower = max(0.0, lo - eddy, eddy - hi) ** 2
    return [lower, (hi + eddy) ** 2]


def region_integrals(
    reference,
    region,
    shape,
    translation=(0.0, 0.0, 0.0),
    rotation=None,
    frequency=3.3043e6,
):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    points, _ = region.voxels(shape, translation, rotation)
    electric = reference.field(points)
    dv = np.prod(region.size_m) / len(points)
    integral = float(np.sum(electric**2) * dv)
    path = reference.conductor.path_points
    potential = vector_potential(points, list(zip(path[:-1], path[1:])), current=1.0)
    return integral, {
        "a_squared": float(np.sum(potential**2) * dv),
        "e_dot_a": float(np.sum(electric * potential) * dv),
    }


def combined_incident_integral(
    e_squared, a_squared, e_dot_a, frequency, wire_impedance
):
    coefficient = 2j * np.pi * frequency / wire_impedance
    return float(
        e_squared + abs(coefficient) ** 2 * a_squared + 2 * coefficient.real * e_dot_a
    )


@dataclass(frozen=True)
class TunedCircuit:
    inductance_h: float
    wire_resistance_ohm: float
    self_capacitance_f: float
    frequency_hz: float
    source_resistance_ohm: float = 1.0

    @property
    def tuning_capacitance_f(self):
        w = 2 * np.pi * self.frequency_hz
        z = 1 / (
            1 / (self.wire_resistance_ohm + 1j * w * self.inductance_h)
            + 1j * w * self.self_capacitance_f
        )
        if z.imag <= 0:
            raise ValueError("coil cannot be series tuned below this self resonance")
        return 1 / (w * z.imag)

    def evaluate(
        self,
        delta_c=0.0,
        conductance=0.0,
        eddy_r=0.0,
        l_scale=1.0,
        r_scale=1.0,
        c_scale=1.0,
        wire_temperature=293.15,
        parcel_temperature=293.15,
    ):
        inductance = self.inductance_h * l_scale
        rw = (
            self.wire_resistance_ohm
            * r_scale
            * (1 + 0.00393 * (wire_temperature - 293.15))
        )
        r = rw + eddy_r
        cp = self.self_capacitance_f + delta_c
        cs = self.tuning_capacitance_f * c_scale

        def terms(f):
            w = 2 * np.pi * f
            zw = r + 1j * w * inductance
            y = 1 / zw + conductance + 1j * w * cp
            z = 1 / y
            return zw, y, z, z + 1 / (1j * w * cs) + self.source_resistance_ohm

        # Find the first series resonance below the coil parallel resonance.
        parallel = 1 / (2 * np.pi * np.sqrt(inductance * cp)) if cp > 0 else np.inf
        low, high = 0.2 * self.frequency_hz, min(2 * self.frequency_hz, 0.98 * parallel)
        resonance = brentq(lambda f: terms(f)[3].imag, low, high)
        zw, y, z, total = terms(self.frequency_hz)
        coil_current = abs(z / (total * zw))  # A per 1 V source, including mismatch
        step = resonance * 1e-5
        slope = (terms(resonance + step)[3].imag - terms(resonance - step)[3].imag) / (
            2 * step
        )
        q = resonance * slope / (2 * terms(resonance)[3].real)
        # Receiver voltage at coil port after drive source is disconnected.
        noise_current = (
            4
            * KB
            * (
                (rw * wire_temperature + eddy_r * parcel_temperature) / abs(zw) ** 2
                + parcel_temperature * conductance
            )
        )
        return {
            "resonance_hz": resonance,
            "loaded_q_during_drive": q,
            "drive_ringdown_s": q / (np.pi * resonance),
            "coil_current_a_per_source_v": coil_current,
            "receiver_port_johnson_psd_v2_hz": noise_current / abs(y) ** 2,
            "series_tuning_capacitance_f": cs,
            "coil_port_voltage_v_per_source_v": abs(z / total),
        }
