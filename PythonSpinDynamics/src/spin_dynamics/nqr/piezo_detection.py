"""Piezoelectric/acoustic detection estimates for quadrupolar NQR lines.

The model in this module is intentionally compact and instrument-facing.  It
turns a voltage on a piezoelectric crystal into a resonant strain estimate,
then into a quadrupolar Rabi rate through an explicit gradient-elastic
sensitivity parameter.  That sensitivity is the part to replace with
DFT-derived strain-to-EFG tensors when they are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

import numpy as np

from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.systems import QuadrupolarSite

PLANCK = 6.62607015e-34
BOLTZMANN = 1.380649e-23
AVOGADRO = 6.02214076e23
EPSILON_0 = 8.8541878128e-12


@dataclass(frozen=True)
class PiezoelectricCrystal:
    """Rectangular piezoelectric sample between full-area electrodes."""

    name: str = "piezoelectric crystal"
    thickness_m: float = 0.5e-3
    electrode_area_m2: float = 25.0e-6
    density_kg_m3: float = 1576.0
    molar_mass_kg_mol: float = 75.07e-3
    quadrupolar_nuclei_per_molecule: float = 1.0
    d_eff_m_per_v: float = 6.1e-12
    relative_permittivity: float = 10.0
    sound_velocity_m_s: float = 2500.0
    mechanical_q: float = 100.0
    mode_shape_factor: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "thickness_m",
            "electrode_area_m2",
            "density_kg_m3",
            "molar_mass_kg_mol",
            "quadrupolar_nuclei_per_molecule",
            "d_eff_m_per_v",
            "relative_permittivity",
            "sound_velocity_m_s",
            "mechanical_q",
            "mode_shape_factor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "name", str(self.name))

    @property
    def volume_m3(self) -> float:
        """Active crystal volume under the electrodes."""

        return self.electrode_area_m2 * self.thickness_m

    @property
    def capacitance_f(self) -> float:
        """Parallel-plate capacitance of the electroded crystal."""

        return (
            EPSILON_0
            * self.relative_permittivity
            * self.electrode_area_m2
            / self.thickness_m
        )

    @property
    def quadrupolar_spin_count(self) -> float:
        """Number of quadrupolar nuclei in the active volume."""

        moles = self.density_kg_m3 * self.volume_m3 / self.molar_mass_kg_mol
        return moles * AVOGADRO * self.quadrupolar_nuclei_per_molecule


@dataclass(frozen=True)
class PiezoNQRLine:
    """One quadrupolar transition used by the detection model."""

    label: str
    frequency_hz: float
    transition_label: str = "x"
    qcc_hz: float | None = None
    eta: float | None = None
    t1_seconds: float = 50.0
    t2_seconds: float = 15.0
    linewidth_hz: float | None = None

    def __post_init__(self) -> None:
        frequency_hz = float(self.frequency_hz)
        if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive and finite")
        for name in ("t1_seconds", "t2_seconds"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.qcc_hz is not None:
            object.__setattr__(self, "qcc_hz", float(self.qcc_hz))
        if self.eta is not None:
            eta = float(self.eta)
            if not np.isfinite(eta) or eta < 0.0 or eta > 1.0:
                raise ValueError("eta must be in the range [0, 1]")
            object.__setattr__(self, "eta", eta)
        if self.linewidth_hz is not None:
            linewidth = float(self.linewidth_hz)
            if not np.isfinite(linewidth) or linewidth < 0.0:
                raise ValueError("linewidth_hz must be non-negative and finite")
            object.__setattr__(self, "linewidth_hz", linewidth)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "transition_label", str(self.transition_label))


@dataclass(frozen=True)
class PiezoNQRCoupling:
    """Strain-to-transition coupling and drive geometry."""

    drive_hz_per_strain: float = 1.0e6
    transition_matrix_element: float = 1.0
    resonant_mode_overlap: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "drive_hz_per_strain",
            "transition_matrix_element",
            "resonant_mode_overlap",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PiezoDetectionResult:
    """Detection estimate for one piezoelectrically driven NQR line."""

    line: PiezoNQRLine
    voltage_rms: float
    electric_field_rms_v_m: float
    strain_peak: float
    rabi_hz: float
    saturation_parameter: float
    spin_absorbed_power_w: float
    mechanical_energy_j: float
    mechanical_drive_power_w: float
    fractional_q_shift: float
    reflected_power_change_w: float
    power_snr: float
    fractional_snr: float
    capacitance_f: float
    capacitive_reactance_ohm: float


def glycine_crystal_from_cif(
    path: str | Path,
    *,
    thickness_m: float = 0.5e-3,
    electrode_area_m2: float = 25.0e-6,
    d_eff_m_per_v: float = 6.1e-12,
    relative_permittivity: float = 10.0,
    sound_velocity_m_s: float = 2500.0,
    mechanical_q: float = 100.0,
    mode_shape_factor: float = 0.5,
) -> PiezoelectricCrystal:
    """Build glycine crystal parameters from CIF density and formula data."""

    text = Path(path).read_text(encoding="utf-8")
    density = _optional_cif_float(text, "_exptl_crystal_density_diffrn", 1.576)
    molar_mass = _optional_cif_float(text, "_chemical_formula_weight", 75.07)
    return PiezoelectricCrystal(
        name="glycine",
        thickness_m=thickness_m,
        electrode_area_m2=electrode_area_m2,
        density_kg_m3=density * 1000.0,
        molar_mass_kg_mol=molar_mass * 1.0e-3,
        quadrupolar_nuclei_per_molecule=1.0,
        d_eff_m_per_v=d_eff_m_per_v,
        relative_permittivity=relative_permittivity,
        sound_velocity_m_s=sound_velocity_m_s,
        mechanical_q=mechanical_q,
        mode_shape_factor=mode_shape_factor,
    )


def glycine_site_from_qcc(
    qcc_hz: float = 1.193e6,
    eta: float = 0.528,
) -> QuadrupolarSite:
    """Return the spin-1 ``14N`` glycine site using the NQRDatabase convention."""

    return QuadrupolarSite(
        spin=1.0,
        isotope="14N",
        label="glycine amine-14N",
        quadrupole_frequency_hz=0.75 * float(qcc_hz),
        eta=float(eta),
    )


def default_glycine_nqr_lines() -> tuple[PiezoNQRLine, ...]:
    """Return glycine ``14N`` lines from the bundled NQRDatabase values."""

    site = glycine_site_from_qcc()
    by_frequency = sorted(
        diagonalize_site(site).transitions,
        key=lambda item: item.frequency_hz,
    )
    database = {
        "y": (737.0e3, 50.0, 17.1, 0.8e3),
        "x": (1052.0e3, 43.4, 12.5, 2.8e3),
    }
    lines: list[PiezoNQRLine] = []
    for transition in by_frequency:
        if transition.label not in database:
            continue
        frequency, t1, t2, width = database[transition.label]
        lines.append(
            PiezoNQRLine(
                label=f"glycine {transition.label}",
                frequency_hz=frequency,
                transition_label=transition.label,
                qcc_hz=1.193e6,
                eta=0.528,
                t1_seconds=t1,
                t2_seconds=t2,
                linewidth_hz=width,
            )
        )
    return tuple(lines)


def load_glycine_nqr_lines_from_sqlite(path: str | Path) -> tuple[PiezoNQRLine, ...]:
    """Load deduplicated glycine ``14N`` lines from an NQRDatabase SQLite export."""

    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"NQR database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                l.frequency_khz,
                l.fwhm_khz,
                l.t1_s,
                l.t2_s,
                st.qcc_khz,
                st.eta,
                st.site_label
            FROM compounds c
            JOIN samples s ON s.compound_id = c.id
            JOIN sites st ON st.sample_id = s.id
            JOIN lines l ON l.site_id = st.id
            WHERE lower(c.canonical_name) = 'glycine'
              AND st.isotope = '14N'
              AND l.frequency_khz IS NOT NULL
            ORDER BY l.frequency_khz, l.source_id
            """
        ).fetchall()
    if not rows:
        return default_glycine_nqr_lines()

    grouped: dict[float, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(round(float(row["frequency_khz"]), 6), []).append(row)

    qcc = next(
        (float(row["qcc_khz"]) * 1e3 for row in rows if row["qcc_khz"] is not None),
        1.193e6,
    )
    eta = next(
        (float(row["eta"]) for row in rows if row["eta"] is not None),
        0.528,
    )
    transition_labels = _labels_by_frequency(qcc, eta)
    lines: list[PiezoNQRLine] = []
    for frequency_khz, group in sorted(grouped.items()):
        t1 = _mean_optional(row["t1_s"] for row in group) or 50.0
        t2 = _mean_optional(row["t2_s"] for row in group) or 15.0
        linewidth = _mean_optional(row["fwhm_khz"] for row in group)
        frequency_hz = float(frequency_khz) * 1e3
        transition_label = min(
            transition_labels,
            key=lambda item: abs(transition_labels[item] - frequency_hz),
        )
        lines.append(
            PiezoNQRLine(
                label=f"glycine {transition_label}",
                frequency_hz=frequency_hz,
                transition_label=transition_label,
                qcc_hz=qcc,
                eta=eta,
                t1_seconds=t1,
                t2_seconds=t2,
                linewidth_hz=(None if linewidth is None else linewidth * 1e3),
            )
        )
    return tuple(lines)


def resonant_strain_peak(
    crystal: PiezoelectricCrystal,
    voltage_rms: float,
) -> float:
    """Return peak strain at a resonantly enhanced acoustic antinode."""

    voltage_rms = _positive_finite(voltage_rms, "voltage_rms")
    electric_field_peak = np.sqrt(2.0) * voltage_rms / crystal.thickness_m
    return (
        crystal.d_eff_m_per_v
        * electric_field_peak
        * crystal.mechanical_q
        * crystal.mode_shape_factor
    )


def simulate_piezoelectric_nqr_detection(
    crystal: PiezoelectricCrystal,
    line: PiezoNQRLine,
    coupling: PiezoNQRCoupling | None = None,
    *,
    voltage_rms: float = 10.0,
    detuning_hz: float = 0.0,
    temperature_k: float = 300.0,
    spin_temperature_enhancement: float = 1.0,
    readout_efficiency: float = 1.0,
    power_noise_density_w_per_sqrt_hz: float = 1.0e-15,
    fractional_noise_density_per_sqrt_hz: float = 1.0e-6,
    integration_time_seconds: float = 1.0,
) -> PiezoDetectionResult:
    """Estimate electrical/acoustic detectability for one NQR transition."""

    coupling = PiezoNQRCoupling() if coupling is None else coupling
    voltage_rms = _positive_finite(voltage_rms, "voltage_rms")
    temperature_k = _positive_finite(temperature_k, "temperature_k")
    spin_temperature_enhancement = _positive_finite(
        spin_temperature_enhancement,
        "spin_temperature_enhancement",
    )
    readout_efficiency = _positive_finite(readout_efficiency, "readout_efficiency")
    integration_time_seconds = _positive_finite(
        integration_time_seconds,
        "integration_time_seconds",
    )
    power_noise = _positive_finite(
        power_noise_density_w_per_sqrt_hz,
        "power_noise_density_w_per_sqrt_hz",
    )
    fractional_noise = _positive_finite(
        fractional_noise_density_per_sqrt_hz,
        "fractional_noise_density_per_sqrt_hz",
    )

    strain = resonant_strain_peak(crystal, voltage_rms)
    rabi_hz = (
        coupling.drive_hz_per_strain
        * coupling.transition_matrix_element
        * coupling.resonant_mode_overlap
        * strain
    )
    rabi_rad_s = 2.0 * np.pi * rabi_hz
    detuning_rad_s = 2.0 * np.pi * float(detuning_hz)
    saturation = (
        (rabi_rad_s * rabi_rad_s) * line.t1_seconds * line.t2_seconds
        / (1.0 + (detuning_rad_s * line.t2_seconds) ** 2)
    )
    absorbed = nuclear_absorbed_power(
        crystal.quadrupolar_spin_count,
        line.frequency_hz,
        line.t1_seconds,
        temperature_k=temperature_k,
        spin_temperature_enhancement=spin_temperature_enhancement,
        saturation_parameter=saturation,
        spin_dimension=3,
    )
    energy = acoustic_strain_energy(crystal, strain)
    drive_power = 2.0 * np.pi * line.frequency_hz * energy / crystal.mechanical_q
    fractional_q = 0.0 if drive_power == 0.0 else absorbed / drive_power
    reflected_change = readout_efficiency * absorbed
    power_snr = reflected_change * np.sqrt(integration_time_seconds) / power_noise
    fractional_snr = (
        abs(fractional_q)
        * np.sqrt(integration_time_seconds)
        / fractional_noise
    )
    reactance = 1.0 / (2.0 * np.pi * line.frequency_hz * crystal.capacitance_f)

    return PiezoDetectionResult(
        line=line,
        voltage_rms=voltage_rms,
        electric_field_rms_v_m=voltage_rms / crystal.thickness_m,
        strain_peak=strain,
        rabi_hz=rabi_hz,
        saturation_parameter=saturation,
        spin_absorbed_power_w=absorbed,
        mechanical_energy_j=energy,
        mechanical_drive_power_w=drive_power,
        fractional_q_shift=fractional_q,
        reflected_power_change_w=reflected_change,
        power_snr=power_snr,
        fractional_snr=fractional_snr,
        capacitance_f=crystal.capacitance_f,
        capacitive_reactance_ohm=reactance,
    )


def acoustic_strain_energy(
    crystal: PiezoelectricCrystal,
    strain_peak: float,
) -> float:
    """Return peak standing-wave strain energy for the active volume."""

    strain_peak = _positive_finite(strain_peak, "strain_peak")
    modulus = crystal.density_kg_m3 * crystal.sound_velocity_m_s**2
    return 0.5 * modulus * strain_peak**2 * crystal.volume_m3


def nuclear_absorbed_power(
    spin_count: float,
    frequency_hz: float,
    t1_seconds: float,
    *,
    temperature_k: float = 300.0,
    spin_temperature_enhancement: float = 1.0,
    saturation_parameter: float = 1.0,
    spin_dimension: int = 3,
) -> float:
    """Return steady resonant power absorbed by a high-temperature transition."""

    spin_count = _positive_finite(spin_count, "spin_count")
    frequency_hz = _positive_finite(frequency_hz, "frequency_hz")
    t1_seconds = _positive_finite(t1_seconds, "t1_seconds")
    temperature_k = _positive_finite(temperature_k, "temperature_k")
    spin_temperature_enhancement = _positive_finite(
        spin_temperature_enhancement,
        "spin_temperature_enhancement",
    )
    saturation = max(float(saturation_parameter), 0.0)
    if int(spin_dimension) < 2:
        raise ValueError("spin_dimension must be at least 2")
    thermal_difference = (
        spin_temperature_enhancement
        * PLANCK
        * frequency_hz
        / (int(spin_dimension) * BOLTZMANN * temperature_k)
    )
    saturated_fraction = (
        1.0 if np.isinf(saturation) else saturation / (1.0 + saturation)
    )
    return (
        0.5
        * spin_count
        * PLANCK
        * frequency_hz
        * thermal_difference
        * saturated_fraction
        / t1_seconds
    )


def _labels_by_frequency(qcc_hz: float, eta: float) -> dict[str, float]:
    transitions = diagonalize_site(glycine_site_from_qcc(qcc_hz, eta)).transitions
    return {transition.label: transition.frequency_hz for transition in transitions}


def _mean_optional(values) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _optional_cif_float(text: str, tag: str, default: float) -> float:
    pattern = rf"^{re.escape(tag)}\s+([^\s]+)"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        return float(default)
    value = match.group(1).strip().strip("'\"")
    if value in ("?", "."):
        return float(default)
    if "(" in value:
        value = value[: value.index("(")]
    return float(value)


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value
