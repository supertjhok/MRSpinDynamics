"""Comparisons against established external codes.

Agreement with our own MATLAB port is weaker evidence than agreement with an
independent code, so these tests rebuild the physics with another package's
operators and solvers:

- nmrsim (second-order NMR line lists for coupled spin-1/2 systems);
- QuTiP (quadrupolar plus Zeeman eigenproblems from its own spin matrices,
  Lindblad propagation with ``mesolve`` and ``steadystate``).

They need the ``crossval`` extra (``pip install -e ".[crossval]"``) and run in
their own CI job, which sets ``SPIN_DYNAMICS_REQUIRE_CROSSVAL=1`` so a missing
package fails instead of skipping. Each result is recorded in
``validation/evidence.json``.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

from spin_dynamics.coupling import coupled_spin_system
from spin_dynamics.coupling.evolution import evolve_density
from spin_dynamics.coupling.hamiltonians import isotropic_j_hamiltonian, zeeman_hamiltonian
from spin_dynamics.coupling.operators import total_operator
from spin_dynamics.dnp import (
    CoupledElectronNuclearModel,
    ElectronNuclearTransitionRates,
    MicrowaveDrive,
)
from spin_dynamics.nqr import diagonalize_site, quadrupolar_site

TAU = 2.0 * np.pi


def _reference(module: str):
    """Import a reference package, skipping unless the CI job requires it."""

    if os.environ.get("SPIN_DYNAMICS_REQUIRE_CROSSVAL") == "1":
        return importlib.import_module(module)
    return pytest.importorskip(module)

# --------------------------------------------------------------------------
# nmrsim: strongly coupled spin-1/2 spectra


# (offsets in Hz, couplings in Hz): strong coupling (|delta nu| ~ J) in each.
# nmrsim reports positive frequencies only (spectrometer convention), so every
# line of these systems lies above zero.
_COUPLED_SYSTEMS = {
    "AB": ([100.0, 115.0], [[0, 12.0], [12.0, 0]]),
    "AB2": ([40.0, 52.0, 52.0], [[0, 9.0, 9.0], [9.0, 0, 0], [9.0, 0, 0]]),
    "ABX": (
        [100.0, 108.0, 400.0],
        [[0, 15.0, 4.0], [15.0, 0, 9.0], [4.0, 9.0, 0]],
    ),
    "AA'BB'": (
        [80.0, 80.0, 110.0, 110.0],
        [[0, 2.0, 7.0, 0.5], [2.0, 0, 0.5, 7.0], [7.0, 0.5, 0, 1.5], [0.5, 7.0, 1.5, 0]],
    ),
    "ABCDE": (
        [200.0, 211.0, 219.0, 233.0, 237.0],
        [
            [0, 8.0, -3.0, 1.0, 0.0],
            [8.0, 0, 6.5, -2.0, 1.2],
            [-3.0, 6.5, 0, 9.0, 0.4],
            [1.0, -2.0, 9.0, 0, 13.0],
            [0.0, 1.2, 0.4, 13.0, 0],
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(_COUPLED_SYSTEMS))
def test_coupled_spin_fid_matches_nmrsim_line_list(name: str) -> None:
    qm = _reference("nmrsim.qm")
    offsets, couplings = _COUPLED_SYSTEMS[name]
    # nmrsim's sparse path depends on an incompatible `sparse` release; the
    # dense path is the same quantum-mechanical calculation. cutoff=0 keeps the
    # weak and combination lines it otherwise drops below intensity 0.001.
    lines = np.asarray(
        qm.qm_spinsystem(offsets, couplings, sparse=False, cutoff=0.0), dtype=float
    )

    system = coupled_spin_system(offsets, couplings)
    hamiltonian = zeeman_hamiltonian(system) + isotropic_j_hamiltonian(system)
    nspin = system.nspin
    density = total_operator(nspin, "x")  # after an ideal 90-degree pulse
    detector = total_operator(nspin, "x") + 1j * total_operator(nspin, "y")

    times = np.random.default_rng(len(name)).uniform(0.0, 0.5, 48)
    # The trace runs over all 2**n states; per spin that is 2**(n - 1) copies
    # of the single-spin Tr(Ix Ix) = 1/2.
    ours = np.array(
        [np.trace(detector @ evolve_density(density, hamiltonian, t)) for t in times]
    ) / 2 ** (nspin - 1)
    # nmrsim normalizes the summed intensity to the number of spins; one
    # isolated spin gives Tr(F+ rho) = exp(2 pi i nu t) / 2.
    reference = 0.5 * np.exp(1j * TAU * np.outer(times, lines[:, 0])) @ lines[:, 1]
    # 1e-8: ABX also has a combination line at -192 Hz (intensity ~1e-9) that
    # nmrsim's positive-frequency list omits; the other systems agree to 1e-9.
    np.testing.assert_allclose(ours, reference, rtol=0, atol=1e-8 * nspin)


# --------------------------------------------------------------------------
# QuTiP: quadrupolar + Zeeman eigenproblem from its own spin matrices


def _qutip_quadrupolar_hamiltonian(qutip, spin, cq_hz, eta, gamma_hz_per_t, b0):
    """Textbook H/h in Hz: C_Q / (4I(2I-1)) [3Iz^2 - I^2 + eta (Ix^2 - Iy^2)] - gamma B.I."""

    jx, jy, jz = (qutip.jmat(spin, axis) for axis in "xyz")
    identity = qutip.qeye(int(round(2 * spin + 1)))
    prefactor = cq_hz / (4.0 * spin * (2.0 * spin - 1.0))
    quadrupole = prefactor * (
        3.0 * jz * jz - spin * (spin + 1.0) * identity + eta * (jx * jx - jy * jy)
    )
    zeeman = -gamma_hz_per_t * (b0[0] * jx + b0[1] * jy + b0[2] * jz)
    return quadrupole + zeeman, (jx, jy, jz)


def _line_strengths(levels_hz, vectors, components, floor_hz):
    """Distinct transition frequencies with their summed dipole strength squared.

    Summing over degenerate partners makes the result independent of the basis
    chosen inside a degenerate level (the zero-field Kramers doublets).
    """

    lines: list[list[float]] = []
    for lower in range(len(levels_hz)):
        for upper in range(lower + 1, len(levels_hz)):
            frequency = levels_hz[upper] - levels_hz[lower]
            if frequency <= floor_hz:
                continue
            weight = sum(
                abs(vectors[:, lower].conj() @ component @ vectors[:, upper]) ** 2
                for component in components
            )
            if weight <= 1e-20:
                continue
            match = next((line for line in lines if abs(line[0] - frequency) <= floor_hz), None)
            if match is None:
                lines.append([frequency, weight])
            else:
                match[1] += weight
    return sorted((round(f, 3), w) for f, w in lines)


_NQR_CASES = [
    # isotope, spin, C_Q (Hz), eta, B0 (T) in the EFG principal-axis frame
    ("14N", 1.0, 3.2e6, 0.3, (0.0, 0.0, 0.0)),
    ("14N", 1.0, 3.2e6, 0.3, (0.12, -0.07, 0.31)),
    ("35Cl", 1.5, 68.0e6, 0.1, (0.0, 0.0, 0.0)),
    ("35Cl", 1.5, 68.0e6, 0.1, (0.9, 0.4, -1.3)),
    ("27Al", 2.5, 4.1e6, 0.45, (0.0, 0.0, 0.0)),
    ("27Al", 2.5, 4.1e6, 0.45, (0.05, 0.11, -0.08)),
    ("51V", 3.5, 3.6e6, 0.2, (0.02, -0.03, 0.04)),
]


@pytest.mark.parametrize("isotope, spin, cq_hz, eta, b0", _NQR_CASES)
def test_quadrupolar_zeeman_lines_match_qutip(isotope, spin, cq_hz, eta, b0) -> None:
    qutip = _reference("qutip")
    site = quadrupolar_site(isotope, cq_hz=cq_hz, eta=eta, spin=spin)
    reference, components = _qutip_quadrupolar_hamiltonian(
        qutip, spin, cq_hz, eta, site.gamma_hz_per_t, b0
    )
    ref_levels, ref_states = reference.eigenstates()
    ref_vectors = np.column_stack([state.full().ravel() for state in ref_states])
    ref_components = [component.full() for component in components]

    eigensystem = diagonalize_site(site, b0)
    np.testing.assert_allclose(
        eigensystem.levels_hz - eigensystem.levels_hz.mean(),
        ref_levels - ref_levels.mean(),
        rtol=0,
        atol=1e-9 * abs(cq_hz),
    )
    floor = 1e-6 * abs(cq_hz)
    ours = _line_strengths(
        eigensystem.levels_hz,
        eigensystem.eigenvectors,
        ref_components,
        floor,
    )
    expected = _line_strengths(ref_levels, ref_vectors, ref_components, floor)
    assert [f for f, _ in ours] == [f for f, _ in expected]
    np.testing.assert_allclose([w for _, w in ours], [w for _, w in expected], rtol=1e-9)
    # Our transition list carries the same lines and strengths.
    listed = sorted(round(t.frequency_hz, 3) for t in eigensystem.transitions)
    assert sorted({f for f, _ in expected}) == sorted(set(listed))


def test_qutip_reference_reproduces_the_spin1_textbook_lines() -> None:
    qutip = _reference("qutip")
    cq, eta = 3.2e6, 0.3
    hamiltonian, _ = _qutip_quadrupolar_hamiltonian(qutip, 1.0, cq, eta, 0.0, (0.0, 0.0, 0.0))
    levels = np.sort(hamiltonian.eigenenergies())
    gaps = sorted({levels[j] - levels[i] for i in range(3) for j in range(i + 1, 3)})
    expected = sorted([0.75 * cq * (1 + eta / 3), 0.75 * cq * (1 - eta / 3), 0.5 * cq * eta])
    np.testing.assert_allclose(gaps, expected, rtol=1e-12)


# --------------------------------------------------------------------------
# QuTiP: Lindblad propagation of the coupled electron-nuclear model


def _overhauser_model(temperature_kelvin: float = 10.0) -> CoupledElectronNuclearModel:
    return CoupledElectronNuclearModel(
        0.35,
        temperature_kelvin=temperature_kelvin,
        electron_t1_seconds=2e-6,
        nuclear_t1_seconds=0.5,
        electron_dephasing_per_second=3e5,
        secular_hyperfine_hz=2e5,
        transitions=ElectronNuclearTransitionRates(0.05, 0.3, 0.1, 0.02),
    )


def _qutip_rotating_hamiltonian(qutip, model, drive):
    """Rotating-frame Hamiltonian built from QuTiP spin-1/2 operators (S first)."""

    sx, sy, sz = (qutip.tensor(qutip.jmat(0.5, axis), qutip.qeye(2)) for axis in "xyz")
    iz = qutip.tensor(qutip.qeye(2), qutip.jmat(0.5, "z"))
    return (
        drive.detuning_rad_per_s * sz
        + TAU * model.secular_hyperfine_hz * sz * iz
        + drive.nutation_rad_per_s * (np.cos(drive.phase_rad) * sx + np.sin(drive.phase_rad) * sy)
    )


def _as_qobj(qutip, matrix):
    return qutip.Qobj(np.asarray(matrix), dims=[[2, 2], [2, 2]])


def test_overhauser_dynamics_match_qutip_mesolve() -> None:
    qutip = _reference("qutip")
    model = _overhauser_model()
    drive = MicrowaveDrive(nutation_rad_per_s=4e5, detuning_rad_per_s=TAU * 1e4, phase_rad=0.3)
    hamiltonian = _qutip_rotating_hamiltonian(qutip, model, drive)
    jumps = [_as_qobj(qutip, jump) for jump in model.jump_operators()]
    times = np.linspace(0.0, 2e-5, 41)

    ours = model.simulate(times, drive=drive)
    result = qutip.mesolve(
        hamiltonian,
        _as_qobj(qutip, model.thermal_density()),
        times,
        c_ops=jumps,
        options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000},
    )
    reference = np.array([state.full() for state in result.states])
    np.testing.assert_allclose(ours.density_matrices, reference, rtol=0, atol=1e-8)


def test_overhauser_steady_state_matches_qutip() -> None:
    qutip = _reference("qutip")
    model = _overhauser_model()
    jumps = [_as_qobj(qutip, jump) for jump in model.jump_operators()]
    drive = MicrowaveDrive(nutation_rad_per_s=4e5, detuning_rad_per_s=TAU * 1e4)

    driven = qutip.steadystate(_qutip_rotating_hamiltonian(qutip, model, drive), jumps)
    np.testing.assert_allclose(model.steady_state(drive).density, driven.full(), atol=1e-10)

    # Without a drive QuTiP's steady state is the Gibbs state of a laboratory
    # Hamiltonian it builds itself.
    idle = qutip.steadystate(_qutip_rotating_hamiltonian(qutip, model, MicrowaveDrive()), jumps)
    sz = qutip.tensor(qutip.jmat(0.5, "z"), qutip.qeye(2))
    iz = qutip.tensor(qutip.qeye(2), qutip.jmat(0.5, "z"))
    lab = TAU * (
        model.field_tesla * (model.gamma_e_hz_per_t * sz - model.gamma_n_hz_per_t * iz)
        + model.secular_hyperfine_hz * sz * iz
    )
    hbar_over_kt = 1.054571817e-34 / (1.380649e-23 * model.temperature_kelvin)
    gibbs = (-hbar_over_kt * lab).expm()
    gibbs = gibbs / gibbs.tr()
    np.testing.assert_allclose(idle.full(), gibbs.full(), atol=1e-10)
