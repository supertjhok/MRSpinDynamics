"""Tests for the PEEC (partial-element) arbitrary-geometry coil solver.

Validates the physics against independent references: the isolated-wire AC resistance
against the exact Kelvin-function skin-effect ratio; the partial-inductance kernels against
the Neumann mutual and the analytic straight-filament self-inductance; the DC inductance
against the existing Biot-Savart/Neumann `coil_inductance`; and a helical solenoid's L, C
and self-resonance against the QOIL semi-analytical model (`solenoid_properties`). The
resistance in the deep-skin (high a/delta) regime is under-resolved at modest cell counts
by construction -- tested only for the correct qualitative trend there.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy.special import bei, beip, ber, berp  # noqa: E402

from spin_dynamics.fields.coil_peec import (  # noqa: E402
    Conductor,
    PEECCoilProperties,
    _mutualfil_matrix,
    capacitance_to_ground,
    coil_properties_peec,
    extract_impedance,
    helical_solenoid,
    self_capacitance,
    self_partial_inductance,
)
from spin_dynamics.fields.coil_properties import ConductorMaterial, solenoid_properties  # noqa: E402
from spin_dynamics.fields.magnetostatics import MU0  # noqa: E402
from spin_dynamics.fields.quasistatic import coil_inductance  # noqa: E402

COPPER = ConductorMaterial("copper", 1.7241e-8, 1.0)


def _kelvin_rac_over_rdc(a: float, delta: float) -> float:
    q = np.sqrt(2.0) * a / delta
    return (q / 2.0) * (ber(q) * beip(q) - bei(q) * berp(q)) / (berp(q) ** 2 + beip(q) ** 2)


def _straight_wire(a: float, length: float, n_radial: int, n_angular: int) -> Conductor:
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]])
    return Conductor(pts, wire_radius=a, material=COPPER, n_radial=n_radial, n_angular=n_angular)


class KernelTests(unittest.TestCase):
    def test_mutualfil_matches_exact_parallel_filaments(self) -> None:
        # The exact filament mutual reproduces the closed form for two parallel filaments,
        # mu0*l/2pi*(asinh(l/d) - sqrt(1+(d/l)^2) + d/l), to machine precision.
        length, d = 1.0, 1e-3
        s1 = np.array([[0.0, 0.0, 0.0]])
        e1 = np.array([[0.0, 0.0, length]])
        s2 = np.array([[d, 0.0, 0.0]])
        e2 = np.array([[d, 0.0, length]])
        exact = (MU0 * length / (2 * np.pi)) * (np.arcsinh(length / d) - np.sqrt(1 + (d / length) ** 2) + d / length)
        m = _mutualfil_matrix(s1, e1, s2, e2)[0, 0]
        self.assertLess(abs(m - exact) / exact, 1e-12)

    def test_mutualfil_collinear_split_is_additive(self) -> None:
        # Splitting a straight filament into collinear pieces must reconstruct the same
        # self-inductance (self_partial + exact mutuals) -- the property a numerical
        # quadrature violates and that inflates coil inductance.
        gmd = np.sqrt(1e-6 / np.pi) * np.exp(-0.25)
        length = 0.1

        def self_l(n):
            z = np.linspace(0, length, n + 1)
            s = np.column_stack([np.zeros(n), np.zeros(n), z[:-1]])
            e = np.column_stack([np.zeros(n), np.zeros(n), z[1:]])
            lens = np.linalg.norm(e - s, axis=1)
            cross = _mutualfil_matrix(s, e, s, e)
            np.fill_diagonal(cross, 0.0)
            return sum(self_partial_inductance(x, gmd) for x in lens) + cross.sum()

        # 1-segment vs 8-segment reconstruction agree to ~1.5%.
        self.assertLess(abs(self_l(8) - self_l(1)) / self_l(1), 0.015)

    def test_self_partial_inductance_asymptote(self) -> None:
        length, r = 1.0, 1e-3
        approx = (MU0 * length / (2 * np.pi)) * (np.log(2 * length / r) - 1)
        self.assertLess(abs(self_partial_inductance(length, r) - approx) / approx, 1e-3)


class SkinEffectTests(unittest.TestCase):
    def test_dc_resistance_exact(self) -> None:
        a, length = 1e-3, 1.0
        c = _straight_wire(a, length, 8, 12)
        imp = extract_impedance(c, [1.0])
        self.assertAlmostEqual(imp.dc_resistance, COPPER.resistivity * length / (np.pi * a**2), places=9)

    def test_rac_converges_to_kelvin(self) -> None:
        # Moderate skin depth (a/delta ~ 3) where a few hundred filaments resolve it.
        a, length, freq = 1e-3, 1.0, 4.0e4
        delta = np.sqrt(2 * COPPER.resistivity / (2 * np.pi * freq * MU0))
        exact = _kelvin_rac_over_rdc(a, delta)
        c = _straight_wire(a, length, 16, 20)
        imp = extract_impedance(c, [freq])
        ratio = imp.resistance[0] / imp.dc_resistance
        self.assertGreater(ratio, 1.0)  # AC resistance exceeds DC
        self.assertLess(abs(ratio - exact) / exact, 0.15)

    def test_rac_converges_toward_kelvin_with_cells(self) -> None:
        # Deep skin (a/delta ~ 15): a coarse cross-section mesh under-resolves the AC
        # resistance; refining the cell count moves it monotonically toward the exact
        # Kelvin value. This is why PEEC and FastHenry (both partial-element solvers) differ
        # at high frequency on a coarse mesh but agree as the filament count rises.
        a, length, freq = 1e-3, 1.0, 1e6
        delta = np.sqrt(2 * COPPER.resistivity / (2 * np.pi * freq * MU0))
        exact = _kelvin_rac_over_rdc(a, delta)
        errs = []
        for n in (6, 12, 20):
            c = _straight_wire(a, length, n, n)
            imp = extract_impedance(c, [freq])
            errs.append(abs(imp.resistance[0] / imp.dc_resistance - exact) / exact)
        # error shrinks monotonically with refinement and is small at the finest mesh
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[2], errs[1])
        self.assertLess(errs[2], 0.15)

    def test_surface_impedance_matches_kelvin_deep_skin(self) -> None:
        # SIBC: deep skin (a/delta ~ 48), 48 perimeter cells, within a few % of Kelvin --
        # where the volume solver would need thousands of cells.
        from spin_dynamics.fields.coil_peec import extract_impedance_surface

        a, length, freq = 1e-3, 1.0, 1e7
        delta = np.sqrt(2 * COPPER.resistivity / (2 * np.pi * freq * MU0))
        exact = _kelvin_rac_over_rdc(a, delta) * COPPER.resistivity * length / (np.pi * a**2)
        c = Conductor(np.array([[0, 0, 0], [0, 0, length]]), wire_radius=a, material=COPPER)
        r = extract_impedance_surface(c, [freq], n_perimeter=48).resistance[0]
        self.assertLess(abs(r - exact) / exact, 0.03)

    def test_graded_mesh_more_accurate_than_uniform_at_fixed_cells(self) -> None:
        # At a fixed (coarse) cell count in deep skin, surface-concentrated grading gives a
        # markedly more accurate AC resistance than a uniform mesh.
        side, length, freq = 1e-3, 0.1, 1e6
        a_eq = np.sqrt(side * side / np.pi)
        delta = np.sqrt(2 * COPPER.resistivity / (2 * np.pi * freq * MU0))
        exact = _kelvin_rac_over_rdc(a_eq, delta)
        pts = np.array([[0, 0, 0], [0, 0, length]])
        kw = dict(material=COPPER, cross_section="rect", width=side, height=side, n_width=8, n_height=8)
        r_unif = extract_impedance(Conductor(pts, grading=1.0, **kw), [freq])
        r_grad = extract_impedance(Conductor(pts, grading=2.5, **kw), [freq])
        e_unif = abs(r_unif.resistance[0] / r_unif.dc_resistance - exact) / exact
        e_grad = abs(r_grad.resistance[0] / r_grad.dc_resistance - exact) / exact
        self.assertLess(e_grad, e_unif)

    def test_resistance_rises_with_frequency(self) -> None:
        a = 1e-3
        c = _straight_wire(a, 1.0, 12, 16)
        imp = extract_impedance(c, [1e4, 1e5, 1e6])
        self.assertLess(imp.resistance[0], imp.resistance[1])
        self.assertLess(imp.resistance[1], imp.resistance[2])


class InductanceTests(unittest.TestCase):
    def test_dc_inductance_matches_references(self) -> None:
        # Helical PEEC solenoid DC inductance vs QOIL's current-sheet L_s (the apt
        # reference) and, more loosely, the stacked-loop Neumann coil_inductance.
        D, length, turns, d = 20e-3, 30e-3, 6, 1e-3
        c = helical_solenoid(diameter=D, length=length, turns=turns, wire_radius=d / 2,
                             material=COPPER, n_per_turn=16, n_radial=2, n_angular=6)
        imp = extract_impedance(c, [1.0])
        l_qoil = solenoid_properties(diameter=D, length=length, turns=turns,
                                     wire_diameter=d, frequency=1e6).inductance_currentsheet
        self.assertLess(abs(imp.dc_inductance - l_qoil) / l_qoil, 0.10)
        radii = np.full(turns, D / 2)
        centers = np.zeros((turns, 3))
        centers[:, 2] = np.linspace(-length / 2, length / 2, turns)
        l_ref = coil_inductance(radii, centers, wire_radius=d / 2)
        self.assertLess(abs(imp.dc_inductance - l_ref) / l_ref, 0.25)

    def test_solenoid_inductance_stable_with_path_resolution(self) -> None:
        # The exact filament mutual makes the inductance path-resolution independent (a
        # numerical quadrature over-couples vertex-sharing segments and drifts up without
        # bound). L must be stable as n_per_turn increases.
        vals = []
        for n_per in (8, 12, 20):
            c = helical_solenoid(diameter=20e-3, length=30e-3, turns=6, wire_radius=0.5e-3,
                                 material=COPPER, n_per_turn=n_per, n_radial=3, n_angular=6)
            vals.append(extract_impedance(c, [1.0]).dc_inductance)
        # Spread across 8->20 segments/turn stays under 8% (a quadrature drifts ~2x).
        self.assertLess((max(vals) - min(vals)) / min(vals), 0.08)


class SolenoidVsQOILTests(unittest.TestCase):
    def setUp(self) -> None:
        self.D, self.l, self.N, self.d = 20e-3, 30e-3, 10, 1e-3
        self.f = 5e6
        self.c = helical_solenoid(diameter=self.D, length=self.l, turns=self.N,
                                  wire_radius=self.d / 2, material=COPPER,
                                  n_per_turn=12, n_radial=4, n_angular=8)
        self.q = solenoid_properties(diameter=self.D, length=self.l, turns=self.N,
                                     wire_diameter=self.d, frequency=self.f)

    def test_inductance_matches_qoil(self) -> None:
        imp = extract_impedance(self.c, [self.f])
        self.assertLess(abs(imp.inductance[0] - self.q.inductance_effective) / self.q.inductance_effective, 0.08)

    def test_self_resonance_matches_qoil(self) -> None:
        # PEEC self-capacitance -> f_res matches QOIL's f_res (the physical self-resonance),
        # even though PEEC's C differs from QOIL's lumped-equivalent C_p fitting capacitance.
        p = coil_properties_peec(self.c, self.f)
        self.assertLess(
            abs(p.self_resonant_frequency - self.q.self_resonant_frequency) / self.q.self_resonant_frequency,
            0.10,
        )

    def test_capacitance_to_ground_matches_analytic_wire(self) -> None:
        # Isolated-wire self-capacitance (the FastCap/FasterCap single-conductor result)
        # vs the analytic thin-wire formula 2*pi*eps0*L/(ln(L/a)-1).
        eps0 = 8.8541878128e-12
        length, a = 1.0, 1e-3
        npts = 300
        pts = np.column_stack([np.zeros(npts), np.zeros(npts), np.linspace(0, length, npts)])
        c = Conductor(pts, wire_radius=a, material=COPPER)
        c_peec = capacitance_to_ground(c)
        c_analytic = 2 * np.pi * eps0 * length / (np.log(length / a) - 1)
        self.assertLess(abs(c_peec - c_analytic) / c_analytic, 0.15)

    def test_grounded_box_raises_capacitance(self) -> None:
        # A grounded shield raises the coil's self-capacitance; a tighter box raises it more,
        # and a box receding to infinity recovers the free-space value.
        from spin_dynamics.fields.coil_peec import GroundedBox, helical_solenoid

        coil = helical_solenoid(diameter=40e-3, length=60e-3, turns=12, wire_radius=0.8e-3,
                                material=COPPER, n_per_turn=10, axis="z")
        c_free = self_capacitance(coil)
        c_near = self_capacitance(coil, shield=GroundedBox((0, 0, 0), (0.05, 0.05, 0.06)))
        c_far = self_capacitance(coil, shield=GroundedBox((0, 0, 0), (2.0, 2.0, 2.0)))
        self.assertGreater(c_near, c_free)               # shield adds capacitance
        self.assertLess(abs(c_far - c_free) / c_free, 0.02)  # distant box -> free space

    def test_dielectric_former_scales_capacitance(self) -> None:
        from spin_dynamics.fields.coil_peec import helical_solenoid

        coil = helical_solenoid(diameter=40e-3, length=60e-3, turns=12, wire_radius=0.8e-3,
                                material=COPPER, n_per_turn=10)
        self.assertAlmostEqual(
            self_capacitance(coil, relative_permittivity=1.8) / self_capacitance(coil), 1.8, places=6
        )

    def test_capacitance_matches_medhurst(self) -> None:
        # Medhurst empirical self-capacitance (pF) for a single-layer solenoid.
        d_cm, l_over_d = self.D * 100, self.l / self.D
        c_medhurst = d_cm * (0.1126 * l_over_d + 0.08 + 0.27 / np.sqrt(l_over_d)) * 1e-12
        c_peec = self_capacitance(self.c)
        self.assertLess(abs(c_peec - c_medhurst) / c_medhurst, 0.30)

    def test_resistance_has_skin_proximity_loss(self) -> None:
        # Deep skin at 5 MHz is under-resolved (documented), so test only the trend:
        # AC resistance exceeds DC and is positive.
        imp = extract_impedance(self.c, [self.f])
        self.assertGreater(imp.resistance[0], imp.dc_resistance)


class FullFormulationTests(unittest.TestCase):
    """The FastHenry-style per-segment ("full") formulation vs the reduced chain one.

    The chain reduction constrains each cross-section sub-filament to a path-constant
    current, so it captures skin effect but only part of the turn-to-turn proximity effect.
    The full formulation keeps one branch per (segment, sub-filament) -- FastHenry's actual
    system (its ``fillM.c`` adds K-1 mini-meshes per segment) -- and resolves proximity from
    first principles.
    """

    def test_full_equals_chain_for_straight_wire(self) -> None:
        # A single-segment straight bar has no per-segment freedom to exploit: the full
        # system reduces exactly to the chain system (M = 1 makes 1^T A^{-1} 1 == 1/(1^T Y 1)).
        bar = Conductor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]), material=COPPER,
                        cross_section="rect", width=1e-3, height=1e-3, n_width=6, n_height=6)
        freqs = [1e4, 1e6]
        chain = extract_impedance(bar, freqs)
        full = extract_impedance(bar, freqs, formulation="full")
        np.testing.assert_allclose(full.resistance, chain.resistance, rtol=1e-10)
        np.testing.assert_allclose(full.inductance, chain.inductance, rtol=1e-10)

    def test_full_reveals_proximity_loss_chain_misses(self) -> None:
        # A closely-wound solenoid: the full solve must show MORE AC resistance than the
        # chain solve (per-segment crowding toward neighbouring turns adds loss), while the
        # inductance stays within ~2%.
        c = helical_solenoid(diameter=20e-3, length=15e-3, turns=6, wire_radius=0.5e-3,
                             material=COPPER, n_per_turn=10, n_radial=3, n_angular=6)
        f = [2e5]
        chain = extract_impedance(c, f)
        full = extract_impedance(c, f, formulation="full")
        self.assertGreater(full.resistance[0], 1.05 * chain.resistance[0])
        self.assertLess(abs(full.inductance[0] - chain.inductance[0]) / chain.inductance[0], 0.02)

    def test_hairpin_proximity_factor_matches_analytic(self) -> None:
        # Two antiparallel round wires, spacing s, HF limit: the proximity factor is exactly
        # 1/sqrt(1 - (d/s)^2) (conformal-map result). A long hairpin at s/d = 1.5 -> 1.3416.
        # The SIBC-full solve approaches it from below as a/delta rises (validated to ~3%
        # at 10 MHz); the chain SIBC solve misses it almost entirely.
        from spin_dynamics.fields.coil_peec import extract_impedance_surface

        d, s, leg = 1e-3, 1.5e-3, 0.5
        nseg = 16
        za = np.linspace(0.0, leg, nseg + 1)
        leg1 = np.column_stack([np.zeros(nseg + 1), np.zeros(nseg + 1), za])
        leg2 = np.column_stack([np.full(nseg + 1, s), np.zeros(nseg + 1), za[::-1]])
        hairpin = Conductor(np.vstack([leg1, leg2]), wire_radius=d / 2, material=COPPER)
        wire = Conductor(np.column_stack([np.zeros(nseg + 1), np.zeros(nseg + 1),
                                          np.linspace(0.0, 2 * leg + s, nseg + 1)]),
                         wire_radius=d / 2, material=COPPER)
        exact = 1.0 / np.sqrt(1.0 - (d / s) ** 2)
        f = [1e7]
        r_pair = extract_impedance_surface(hairpin, f, n_perimeter=32, formulation="full").resistance[0]
        r_wire = extract_impedance_surface(wire, f, n_perimeter=32).resistance[0]
        factor = r_pair / r_wire
        self.assertLess(abs(factor - exact) / exact, 0.05)

    def test_full_current_distribution_shows_per_segment_crowding(self) -> None:
        # In the hairpin, the full per-segment distribution at a mid-leg segment must be
        # asymmetric (crowded toward the return leg); the chain distribution cannot be.
        d, s, leg = 1e-3, 1.5e-3, 0.2
        nseg = 8
        za = np.linspace(0.0, leg, nseg + 1)
        leg1 = np.column_stack([np.zeros(nseg + 1), np.zeros(nseg + 1), za])
        leg2 = np.column_stack([np.full(nseg + 1, s), np.zeros(nseg + 1), za[::-1]])
        hairpin = Conductor(np.vstack([leg1, leg2]), wire_radius=d / 2, material=COPPER,
                            n_radial=4, n_angular=8)
        from spin_dynamics.fields.coil_peec import current_distribution

        offs, cur = current_distribution(hairpin, 1e6, formulation="full", segment=nseg // 2)
        # centre-of-current shifts toward the return leg (positive x side)
        shift = float(np.sum(offs[:, 0] * cur))
        self.assertGreater(abs(shift), 0.02 * d / 2)

    def test_full_size_guardrail(self) -> None:
        c = helical_solenoid(diameter=20e-3, length=30e-3, turns=30, wire_radius=0.5e-3,
                             material=COPPER, n_per_turn=20, n_radial=6, n_angular=8)
        with self.assertRaises(ValueError):
            extract_impedance(c, [1e6], formulation="full")


class RadiationResistanceTests(unittest.TestCase):
    """First-order (magnetic-dipole) radiation resistance."""

    @staticmethod
    def _loop(radius: float, n: int = 128, turns: int = 1, pitch: float = 0.0) -> Conductor:
        th = np.linspace(0.0, 2.0 * np.pi * turns, turns * n + 1)
        z = np.linspace(0.0, pitch * turns, th.size)
        pts = np.column_stack([radius * np.cos(th), radius * np.sin(th), z])
        return Conductor(pts, wire_radius=1e-3, material=COPPER)

    def test_matches_small_loop_formula(self) -> None:
        # Single circular loop: R_rad = 20 pi^2 (C/lambda)^4 (electrically small).
        from spin_dynamics.fields.coil_peec import radiation_resistance

        a, f = 0.1, 30e6
        lam = 299792458.0 / f
        exact = 20.0 * np.pi**2 * (2.0 * np.pi * a / lam) ** 4
        r = radiation_resistance(self._loop(a), f)
        self.assertLess(abs(r - exact) / exact, 0.01)  # polygon-area discretization only

    def test_n_squared_scaling(self) -> None:
        from spin_dynamics.fields.coil_peec import radiation_resistance

        f = 10e6
        r1 = radiation_resistance(self._loop(0.05, turns=1, pitch=1e-3), f)
        r6 = radiation_resistance(self._loop(0.05, turns=6, pitch=1e-3), f)
        self.assertLess(abs(r6 / r1 - 36.0) / 36.0, 0.05)

    def test_opposed_loops_do_not_dipole_radiate(self) -> None:
        # A Maxwell pair has zero net vector area -> no magnetic-dipole radiation.
        from spin_dynamics.fields.coil_peec import radiation_resistance

        a, sep, n, f = 0.05, 0.05, 96, 10e6
        th = np.linspace(0.0, 2.0 * np.pi, n + 1)
        top = np.column_stack([a * np.cos(th), a * np.sin(th), np.full(th.size, sep / 2)])
        bot = np.column_stack([a * np.cos(th[::-1]), a * np.sin(th[::-1]),
                               np.full(th.size, -sep / 2)])
        pair = Conductor(np.vstack([top, bot[1:]]), wire_radius=1e-3, material=COPPER)
        r_pair = radiation_resistance(pair, f)
        r_single = radiation_resistance(self._loop(a, n=n), f)
        self.assertLess(r_pair, 1e-3 * r_single)

    def test_warns_when_not_electrically_small(self) -> None:
        from spin_dynamics.fields.coil_peec import radiation_resistance

        with self.assertWarns(UserWarning):
            radiation_resistance(self._loop(0.5), 100e6)  # 1 m loop at 100 MHz

    def test_shield_suppresses_radiation_below_cutoff(self) -> None:
        # A grounded box is a cavity: below its lowest resonance the closed conductor lets
        # no field radiate, so R_rad -> 0. The same 5 cm loop that radiates 24 mOhm at
        # 100 MHz in free space radiates nothing inside a 20 cm box (cutoff ~1 GHz).
        from spin_dynamics.fields.coil_peec import GroundedBox, radiation_resistance

        c = self._loop(0.05, n=64)
        box = GroundedBox((0, 0, 0), (0.1, 0.1, 0.1))
        self.assertGreater(radiation_resistance(c, 100e6), 1e-3)          # free space: real
        self.assertEqual(radiation_resistance(c, 100e6, shield=box), 0.0)  # shielded: ~0

    def test_shield_fundamental_resonance_matches_cavity_formula(self) -> None:
        from spin_dynamics.fields.coil_peec import GroundedBox

        # interior 70 x 70 x 100 mm -> TE101 = (c/2) sqrt(1/0.07^2 + 1/0.1^2) ~ 2.62 GHz
        box = GroundedBox((0, 0, 0), (0.035, 0.035, 0.05))
        expected = 0.5 * 299792458.0 * np.sqrt(1 / 0.07**2 + 1 / 0.1**2)
        self.assertLess(abs(box.fundamental_resonance() - expected) / expected, 1e-9)

    def test_shield_warns_near_cavity_resonance(self) -> None:
        from spin_dynamics.fields.coil_peec import GroundedBox, radiation_resistance

        box = GroundedBox((0, 0, 0), (0.1, 0.1, 0.1))
        f_cav = box.fundamental_resonance()
        with self.assertWarns(UserWarning):
            radiation_resistance(self._loop(0.05, n=64), 0.9 * f_cav, shield=box)

    def test_bundle_shield_suppresses_radiation_and_shifts_srf(self) -> None:
        # coil_properties_peec threads the shield to both the capacitance (raises C, lowers
        # SRF) and the radiation (suppressed -> higher Q than the unshielded free-space case).
        from spin_dynamics.fields.coil_peec import GroundedBox

        c = self._loop(0.05, n=24)
        box = GroundedBox((0, 0, 0), (0.1, 0.1, 0.1))
        free = coil_properties_peec(c, 50e6, formulation="chain")
        shielded = coil_properties_peec(c, 50e6, formulation="chain", shield=box)
        self.assertGreater(free.radiation_resistance, 0.0)
        self.assertEqual(shielded.radiation_resistance, 0.0)
        self.assertGreater(shielded.q_factor, free.q_factor)          # radiation removed
        self.assertGreater(shielded.self_capacitance, free.self_capacitance)  # box raises C

    def test_bundle_includes_radiation_in_q(self) -> None:
        # At VHF for a large-ish loop, radiation dominates the ohmic loss and must lower Q;
        # ac_resistance stays ohmic while to_probe_params carries the total.
        c = self._loop(0.05, n=24)
        f = 50e6
        with_rad = coil_properties_peec(c, f, formulation="chain")
        without = coil_properties_peec(c, f, formulation="chain", include_radiation=False)
        self.assertGreater(with_rad.radiation_resistance, 0.0)
        self.assertLess(with_rad.q_factor, without.q_factor)
        self.assertAlmostEqual(with_rad.ac_resistance, without.ac_resistance)
        self.assertAlmostEqual(
            with_rad.to_probe_params()["R"],
            with_rad.ac_resistance + with_rad.radiation_resistance,
        )


class BundleTests(unittest.TestCase):
    def test_coil_properties_peec_fields_and_helpers(self) -> None:
        c = helical_solenoid(diameter=20e-3, length=30e-3, turns=6, wire_radius=0.5e-3,
                             material=COPPER, n_per_turn=12, n_radial=3, n_angular=6)
        p = coil_properties_peec(c, 5e6)
        self.assertIsInstance(p, PEECCoilProperties)
        self.assertTrue(np.isfinite(p.inductance) and p.inductance > 0)
        self.assertTrue(np.isfinite(p.q_factor) and p.q_factor > 0)
        # tuning capacitance resonates the inductance at the solve frequency
        omega = 2 * np.pi * p.frequency
        self.assertAlmostEqual(omega**2 * p.inductance * p.tuning_capacitance(), 1.0, places=9)
        self.assertEqual(set(p.to_probe_params()), {"L", "R", "C"})

    def test_temperature_lowers_resistance(self) -> None:
        cold = ConductorMaterial("copper-cryo", 1.7241e-8, 1.0, temp_coefficient=3.93e-3,
                                 residual_resistivity_ratio=100.0)
        warm = helical_solenoid(diameter=20e-3, length=30e-3, turns=6, wire_radius=0.5e-3,
                                material=cold, n_per_turn=12, n_radial=3, n_angular=6, temperature=293.15)
        cryo = helical_solenoid(diameter=20e-3, length=30e-3, turns=6, wire_radius=0.5e-3,
                                material=cold, n_per_turn=12, n_radial=3, n_angular=6, temperature=77.0)
        # Low frequency (uniform current, R = rho*L/A) so temperature scaling is clean;
        # at high frequency the cryo wire's tiny skin depth is under-resolved.
        r_warm = extract_impedance(warm, [1e4]).resistance[0]
        r_cryo = extract_impedance(cryo, [1e4]).resistance[0]
        self.assertLess(r_cryo, r_warm)


if __name__ == "__main__":
    unittest.main()
