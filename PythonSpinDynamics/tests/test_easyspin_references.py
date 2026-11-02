"""Comparisons against EasySpin (MATLAB), from committed reference data.

``validation/easyspin/generate_easyspin_references.m`` runs EasySpin 6 and
writes ``validation/external/easyspin/``; this test needs neither MATLAB nor
EasySpin. Regenerate the data after changing the cases (see that script).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from spin_dynamics.esr.hyperfine import (
    NuclearSite,
    diagonalize_hyperfine_system,
    electron_nuclear_system,
    zeeman_hamiltonian,
)
from spin_dynamics.nqr.isotopes import quadrupolar_isotope

REFERENCES = Path(__file__).resolve().parents[1] / "validation" / "external" / "easyspin"
METADATA = json.loads((REFERENCES / "metadata.json").read_text(encoding="utf-8"))
TAU = 2.0 * np.pi

_NUCLEI = {
    "1H": NuclearSite("H", "1H"),
    "14N": NuclearSite("N", "14N", spin=1.0, gamma_hz_per_t=quadrupolar_isotope("14N").gamma_hz_per_t),
}


def _system(case: dict) -> object:
    names = case["nucs"].split(",")
    hyperfine_mhz = np.atleast_1d(np.asarray(case["A_MHz"], dtype=float))
    return electron_nuclear_system(
        hyperfine_mhz * 1e6, nuclei=[_NUCLEI[name] for name in names], g_tensor=case["g"]
    )


def _resonances(system, frequency_hz: float, fields_tesla: np.ndarray):
    """Resonance fields of every allowed transition, with the field-swept weight.

    The weight is EasySpin's: transition probability |<l|Sx|u>|^2 (B0 along z,
    B1 along x) times the generalized 1/g factor 1/|d(nu)/dB|.
    """

    def gaps(field: float) -> np.ndarray:
        levels = diagonalize_hyperfine_system(system, (0.0, 0.0, field)).levels_hz
        return levels[None, :] - levels[:, None]

    samples = np.array([gaps(field) for field in fields_tesla]) - frequency_hz
    zeeman = zeeman_hamiltonian(system, (0.0, 0.0, 1.0)) / TAU
    sx = 0.5 * np.kron(np.array([[0, 1], [1, 0]]), np.eye(system.dimension // 2))
    found = []
    n = system.dimension
    for lower in range(n):
        for upper in range(n):
            if upper == lower:
                continue
            crossing = np.nonzero(np.diff(np.sign(samples[:, lower, upper])))[0]
            for index in crossing:
                field = brentq(
                    lambda b: gaps(b)[lower, upper] - frequency_hz,
                    fields_tesla[index],
                    fields_tesla[index + 1],
                    xtol=1e-14,
                )
                eig = diagonalize_hyperfine_system(system, (0.0, 0.0, field))
                vl, vu = eig.eigenvectors[:, lower], eig.eigenvectors[:, upper]
                probability = abs(vl.conj() @ sx @ vu) ** 2
                slope = abs(np.real(vu.conj() @ zeeman @ vu - vl.conj() @ zeeman @ vl))
                found.append((field, probability / slope))
    return np.array(sorted(found))


@pytest.mark.parametrize("case", METADATA["hyperfine"]["cases"], ids=lambda case: case["name"])
def test_hyperfine_resonance_fields_and_intensities_match_easyspin(case: dict) -> None:
    reference = np.loadtxt(REFERENCES / f"resfields_{case['name']}.csv", delimiter=",", ndmin=2)
    frequency_hz = METADATA["hyperfine"]["mwFreq_GHz"] * 1e9
    low, high = (value * 1e-3 for value in METADATA["hyperfine"]["range_mT"])
    ours = _resonances(_system(case), frequency_hz, np.linspace(low, high, 2001))
    # The references list every level pair (Threshold 0), including exactly
    # forbidden ones with zero amplitude; compare the lines either code can see.
    ours = ours[ours[:, 1] > 1e-8 * ours[:, 1].max()]
    reference = reference[reference[:, 1] > 1e-8 * reference[:, 1].max()]
    assert len(ours) == len(reference)
    # Fields: differences in the nuclear constants (14N gamma 3.0766 vs 3.0777
    # MHz/T) and CODATA versions stay below 1e-4 mT.
    np.testing.assert_allclose(ours[:, 0] * 1e3, reference[:, 0], rtol=0, atol=1e-4)
    # Relative intensities, including forbidden lines four orders weaker.
    np.testing.assert_allclose(
        ours[:, 1] / ours[:, 1].max(), reference[:, 1] / reference[:, 1].max(), rtol=2e-3, atol=1e-7
    )


def test_rhombic_powder_spectrum_matches_easyspin_pepper() -> None:
    from spin_dynamics.esr import ESRSpinSystem
    from spin_dynamics.esr.spectra import simulate_field_sweep

    powder = METADATA["powder"]
    default = np.loadtxt(REFERENCES / "powder_rhombic_freq2field1.csv", delimiter=",")
    no_one_over_g = np.loadtxt(REFERENCES / "powder_rhombic_freq2field0.csv", delimiter=",")
    fields = default[:, 0] * 1e-3
    sigma = powder["lw_gaussian_fwhm_mT"] * 1e-3 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # The default "powder" grid is sized from the anisotropy (L-64); it was
    # 16 x 32 x 8 and missed this comparison by 46%.
    ours = simulate_field_sweep(
        ESRSpinSystem(g_tensor=powder["g"]),
        powder["mwFreq_GHz"] * 1e9,
        orientations="powder",
        broadening_tesla=sigma,
        fields_tesla=fields,
    ).spectrum

    def normalized(values):
        return values / np.trapezoid(values, fields)

    def deviation(reference):
        expected = normalized(reference[:, 1])
        return np.max(np.abs(normalized(ours) - expected)) / expected.max()

    # EasySpin's default includes the field-swept 1/g (Aasa-Vanngard) factor;
    # without it the shape differs by about 2%, which this check resolves.
    assert deviation(default) < 0.008
    assert deviation(no_one_over_g) > 0.015
