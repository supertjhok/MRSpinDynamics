"""Regression tests for core review items M-45, L-01 to L-06 and L-49."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.core import _jax_kernels as jk  # noqa: E402
from spin_dynamics.core import _numba_kernels as nk  # noqa: E402
from spin_dynamics.core.echo import calc_time_domain_echo  # noqa: E402
from spin_dynamics.core.kernels import (  # noqa: E402
    _stack_matrix_elements,
    arb10_pulse_relaxation,
    get_arb10_pulse_relaxation,
    sim_spin_dynamics_arb10,
    sim_spin_dynamics_arb10_chunked,
    sim_spin_dynamics_arb10_diffusion,
)
from spin_dynamics.core.rotations import rf_matrix_elements  # noqa: E402
from spin_dynamics.motion import make_circular_reflector, make_elliptical_reflector  # noqa: E402
from spin_dynamics.relaxation import matrix_exponential  # noqa: E402
from spin_dynamics.sequences import (  # noqa: E402
    ADCEvent,
    RFPulse,
    SequenceBlock,
    SequenceIR,
    compile_sequence,
)
from spin_dynamics.sequences.motion import _detuning_values  # noqa: E402


def _cpmg_params(del_w, t1, t2, mth, num_echoes=3, tau=3.0):
    """90x - [tau - 180y - tau - acquire] x N with w1 = 1 rectangular pulses."""

    n = del_w.size
    tp, pul, amp, acq = [np.pi / 2], [1], [1.0], [0]
    for _ in range(num_echoes):
        tp += [tau, np.pi, tau]
        pul += [0, 2, 0]
        amp += [0.0, 1.0, 0.0]
        acq += [0, 0, 1]
    return {
        "tp": tp,
        "pul": pul,
        "amp": amp,
        "acq": acq,
        "grad": np.zeros(len(tp)),
        "Rtot": [
            rf_matrix_elements(del_w, 1.0, np.pi / 2, 0.0),
            rf_matrix_elements(del_w, 1.0, np.pi, np.pi / 2),
        ],
        "del_w": del_w,
        "del_wg": np.zeros(n),
        "T1n": np.full(n, t1),
        "T2n": np.full(n, t2),
        "m0": np.ones(n),
        "mth": np.full(n, mth),
    }


def _bloch_reference(params, dt=2e-3):
    """Independent RK4 of the Bloch equations with relaxation at all times."""

    del_w = np.asarray(params["del_w"], dtype=np.float64)
    n = del_w.size
    r1 = 1.0 / np.asarray(params["T1n"], dtype=np.float64)
    r2 = 1.0 / np.asarray(params["T2n"], dtype=np.float64)
    mth = np.real(np.asarray(params["mth"]))
    phases = {1: 0.0, 2: np.pi / 2}
    m = np.zeros((3, n))
    m[2] = 1.0

    def rhs(state, w1, phi):
        omega = np.vstack([np.full(n, w1 * np.cos(phi)), np.full(n, w1 * np.sin(phi)), del_w])
        deriv = np.cross(omega.T, state.T).T
        deriv[0] -= r2 * state[0]
        deriv[1] -= r2 * state[1]
        deriv[2] += r1 * (mth - state[2])
        return deriv

    out = []
    for tp, pul, amp, acq in zip(params["tp"], params["pul"], params["amp"], params["acq"]):
        w1 = 1.0 if amp > 0 else 0.0
        phi = phases.get(int(pul), 0.0)
        steps = int(np.ceil(tp / dt))
        h = tp / steps
        for _ in range(steps):
            k1 = rhs(m, w1, phi)
            k2 = rhs(m + 0.5 * h * k1, w1, phi)
            k3 = rhs(m + 0.5 * h * k2, w1, phi)
            k4 = rhs(m + h * k3, w1, phi)
            m = m + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if acq:
            out.append(m[0] - 1j * m[1])
    return np.asarray(out)


class PulseRelaxationTests(unittest.TestCase):
    """M-45: opt-in T1/T2 relaxation during RF pulses."""

    def test_split_is_exact_for_isotropic_relaxation_against_bloch_reference(self) -> None:
        # With T1 = T2 and mth = 0 the relaxation generator is a multiple of the
        # identity, commutes with every rotation, and the split is exact.
        params = _cpmg_params(np.linspace(-0.5, 0.5, 7), t1=20.0, t2=20.0, mth=0.0)
        reference = _bloch_reference(params)
        relaxed = sim_spin_dynamics_arb10(params, relax_during_pulses=True)
        default = sim_spin_dynamics_arb10(params)
        np.testing.assert_allclose(relaxed, reference, atol=1e-9)
        # MATLAB-parity default: pulses do not relax, so it misses ~T180/T2.
        self.assertGreater(float(np.max(np.abs(default - reference))), 0.1)

    def test_default_is_unchanged_and_option_sources_agree(self) -> None:
        params = _cpmg_params(np.linspace(-0.5, 0.5, 5), t1=40.0, t2=20.0, mth=1.0)
        self.assertFalse(get_arb10_pulse_relaxation())
        explicit = sim_spin_dynamics_arb10(params, relax_during_pulses=True)
        with arb10_pulse_relaxation():
            scoped = sim_spin_dynamics_arb10(params)
            chunked = sim_spin_dynamics_arb10_chunked(params, num_workers=2, min_chunk_size=1)
        self.assertFalse(get_arb10_pulse_relaxation())
        keyed = sim_spin_dynamics_arb10({**params, "relax_during_pulses": True})
        np.testing.assert_array_equal(scoped, explicit)
        np.testing.assert_array_equal(keyed, explicit)
        np.testing.assert_allclose(chunked, explicit, rtol=0.0, atol=1e-15)
        diffusion = sim_spin_dynamics_arb10_diffusion(
            {**params, "gamma": 1.0, "gradient": 0.0, "diffusion_coefficient": 0.0},
            relax_during_pulses=True,
        )
        np.testing.assert_allclose(diffusion, explicit, rtol=0.0, atol=1e-15)

    def test_backends_agree_with_pulse_relaxation(self) -> None:
        params = _cpmg_params(np.linspace(-0.5, 0.5, 5), t1=40.0, t2=20.0, mth=1.0)
        reference = sim_spin_dynamics_arb10(params, relax_during_pulses=True)
        # Numba core algorithm (runs as plain Python when numba is absent).
        rstack = _stack_matrix_elements(params["Rtot"], params["del_w"].size)
        core = nk.arb10_core(
            np.asarray(params["tp"], dtype=np.float64),
            np.asarray(params["pul"], dtype=np.int64),
            np.asarray(params["amp"], dtype=np.float64),
            np.asarray(params["acq"], dtype=np.uint8),
            params["grad"],
            params["del_w"],
            params["del_wg"],
            params["T1n"],
            params["T2n"],
            params["m0"].astype(np.complex128),
            params["mth"].astype(np.complex128),
            rstack,
            3,
            1,
        )
        np.testing.assert_allclose(core, reference, rtol=0.0, atol=1e-13)
        if jk.JAX_AVAILABLE:
            jax_out = sim_spin_dynamics_arb10(params, backend="jax", relax_during_pulses=True)
            np.testing.assert_allclose(jax_out, reference, rtol=0.0, atol=1e-13)
            from spin_dynamics.core.kernels import sim_spin_dynamics_arb10_batched

            batched = sim_spin_dynamics_arb10_batched([params, params], relax_during_pulses=True)
            np.testing.assert_allclose(batched[1], reference, rtol=0.0, atol=1e-13)

    def test_ideal_train_fitted_t2_matches_input_with_pulse_relaxation(self) -> None:
        from spin_dynamics.workflows import run_ideal_cpmg_train

        common = {"numpts": 51, "num_echoes": 8, "rephase_action": "ignore"}
        # Divide out the relaxation-free echo train so only T2 decay remains.
        baseline = run_ideal_cpmg_train(t1_seconds=1e9, t2_seconds=1e9, **common)
        t2 = 1e-3
        fitted = {}
        for relax in (False, True):
            with arb10_pulse_relaxation(relax):
                result = run_ideal_cpmg_train(t1_seconds=t2, t2_seconds=t2, **common)
            ratio = np.abs(result.echo_integrals) / np.abs(baseline.echo_integrals)
            slope = np.polyfit(result.sequence_time, np.log(ratio), 1)[0]
            fitted[relax] = -1.0 / slope
        self.assertAlmostEqual(fitted[True] / t2, 1.0, places=6)
        # Default: inflated by TE / (TE - T180) = 4/3 for the ideal train.
        self.assertAlmostEqual(fitted[False] / t2, 4.0 / 3.0, places=6)


class PulseIndexValidationTests(unittest.TestCase):
    """L-04: pulse segments must select an existing Rtot matrix."""

    def test_out_of_range_pul_is_rejected_by_every_backend(self) -> None:
        params = _cpmg_params(np.linspace(-0.5, 0.5, 5), t1=40.0, t2=20.0, mth=1.0)
        for bad in (0, 3, -1):
            pul = list(params["pul"])
            pul[2] = bad  # first refocusing pulse
            backends = ["numpy"] + (["jax"] if jk.JAX_AVAILABLE else [])
            for backend in backends:
                with self.subTest(pul=bad, backend=backend):
                    with self.assertRaisesRegex(ValueError, "pul must select"):
                        sim_spin_dynamics_arb10({**params, "pul": pul}, backend=backend)
        # Free-precession segments may keep pul = 0.
        sim_spin_dynamics_arb10(params)


class JaxX64ScopeTests(unittest.TestCase):
    """L-06: the JAX kernels use x64 without flipping the global flag."""

    @unittest.skipUnless(jk.JAX_AVAILABLE, "jax not installed")
    def test_kernel_runs_in_complex128_with_global_x64_disabled(self) -> None:
        import jax

        params = _cpmg_params(np.linspace(-0.5, 0.5, 5), t1=40.0, t2=20.0, mth=1.0)
        reference = sim_spin_dynamics_arb10(params)
        with jax.enable_x64(False):
            out = sim_spin_dynamics_arb10(params, backend="jax")
            self.assertFalse(jax.config.jax_enable_x64)
        self.assertEqual(out.dtype, np.complex128)
        np.testing.assert_allclose(out, reference, rtol=0.0, atol=1e-13)


class MatrixExponentialTests(unittest.TestCase):
    """L-01: one robust matrix exponential for relaxation and exchange."""

    def test_defective_jordan_block_matches_closed_form(self) -> None:
        a, t = -1.0, 2.0
        jordan = np.array([[a, 1.0], [0.0, a]])
        expected = np.exp(a * t) * np.array([[1.0, t], [0.0, 1.0]])
        np.testing.assert_allclose(matrix_exponential(jordan, t), expected, atol=1e-14)

    def test_shared_with_exchange(self) -> None:
        from spin_dynamics.exchange import _matrix_exponential

        rng = np.random.default_rng(1)
        matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        np.testing.assert_allclose(
            matrix_exponential(matrix, 0.7), _matrix_exponential(0.7 * matrix), atol=1e-13
        )


class DetuningCallbackTests(unittest.TestCase):
    """L-02: TypeErrors raised inside a detuning callback are not masked."""

    def test_internal_type_error_propagates(self) -> None:
        def broken(time, positions):
            raise TypeError("inner failure")

        with self.assertRaisesRegex(TypeError, "inner failure"):
            _detuning_values(broken, 0.0, np.zeros((3, 2)))

    def test_one_and_two_argument_callbacks(self) -> None:
        positions = np.zeros((3, 2))
        self.assertEqual(_detuning_values(lambda t: 2.0 * t, 1.5, positions), 3.0)
        np.testing.assert_allclose(
            _detuning_values(lambda t, p: t + p[:, 0], 1.0, positions), [1.0, 1.0, 1.0]
        )


class EventPhaseReferenceTests(unittest.TestCase):
    """L-03: RF and ADC frequency-offset phases share the event-start reference."""

    def test_adc_phase_is_referenced_to_its_own_start(self) -> None:
        dwell, delay, offset = 1e-6, 10e-6, 1e3
        block = SequenceBlock(
            duration_seconds=20e-6,
            adc=ADCEvent(
                num_samples=4,
                dwell_seconds=dwell,
                delay_seconds=delay,
                frequency_offset_hz=offset,
            ),
        )
        compiled = compile_sequence(SequenceIR(blocks=(block,)))
        sample_from_start = dwell * (np.arange(4) + 0.5)
        np.testing.assert_allclose(
            compiled.adc.phase_offsets_rad, 2.0 * np.pi * offset * sample_from_start
        )

    def test_rf_and_adc_with_same_delay_and_offset_agree(self) -> None:
        dwell, delay, offset = 1e-6, 10e-6, 2e3
        rf_block = SequenceBlock(
            duration_seconds=20e-6,
            rf=RFPulse(
                samples_hz=np.ones(4),
                dwell_seconds=dwell,
                delay_seconds=delay,
                frequency_offset_hz=offset,
            ),
        )
        adc_block = SequenceBlock(
            duration_seconds=20e-6,
            adc=ADCEvent(
                num_samples=4, dwell_seconds=dwell, delay_seconds=delay, frequency_offset_hz=offset
            ),
        )
        rf = compile_sequence(SequenceIR(blocks=(rf_block,))).rf_hz
        adc = compile_sequence(SequenceIR(blocks=(adc_block,))).adc.phase_offsets_rad
        # RF samples are evaluated at their raster centres, like ADC samples.
        np.testing.assert_allclose(np.angle(rf[rf != 0]), adc, atol=1e-12)


class ReflectorTests(unittest.TestCase):
    """L-05: large overshoots reflect through the centre to the far side."""

    def test_circular_reflector_matches_one_dimensional_specular_reflection(self) -> None:
        radius = 1.0
        reflect = make_circular_reflector((0.0, 0.0), radius)
        # Along a diameter, specular walls at -R and +R: 0.6 -> 0.6, 1.4 -> 0.6,
        # 2.5 -> -0.5 (opposite side), 3.6 -> -0.4, 4.3 -> 0.3.
        start = np.array([0.6, 1.4, 2.5, 3.6, 4.3])
        expected = np.array([0.6, 0.6, -0.5, -0.4, 0.3])
        direction = np.array([0.6, 0.8])
        positions = start[:, np.newaxis] * direction
        np.testing.assert_allclose(reflect(positions), expected[:, np.newaxis] * direction)

    def test_elliptical_reflector_uses_the_same_fold(self) -> None:
        reflect = make_elliptical_reflector((1.0, -1.0), (2.0, 0.5))
        positions = np.array([[1.0 + 2.0 * 2.5, -1.0]])
        np.testing.assert_allclose(reflect(positions), [[1.0 - 2.0 * 0.5, -1.0]])


class EchoRegistrationTests(unittest.TestCase):
    """L-49: the optional exact echo registration matches a direct DFT sum."""

    def test_exact_registration_matches_direct_sum(self) -> None:
        rng = np.random.default_rng(2)
        for numpts, shift in ((101, 0.0), (100, 0.0), (64, 0.3)):
            wvect = np.linspace(-10.0, 10.0, numpts) + shift
            spect = rng.normal(size=numpts) + 1j * rng.normal(size=numpts)
            echo, tvect = calc_time_domain_echo(spect, wvect, registration="exact")
            direct = (4.0 / (4 * numpts)) * np.exp(1j * np.outer(tvect, wvect)) @ spect
            np.testing.assert_allclose(echo, direct, atol=1e-12)
            self.assertAlmostEqual(float(tvect[tvect.size // 2]), 0.0)

    def test_matlab_registration_is_the_default(self) -> None:
        wvect = np.linspace(-10.0, 10.0, 101)
        spect = np.exp(-(wvect**2))
        echo, tvect = calc_time_domain_echo(spect, wvect)
        legacy, legacy_t = calc_time_domain_echo(spect, wvect, registration="matlab")
        np.testing.assert_array_equal(echo, legacy)
        np.testing.assert_array_equal(tvect, legacy_t)
        with self.assertRaises(ValueError):
            calc_time_domain_echo(spect, wvect, registration="bogus")


if __name__ == "__main__":
    unittest.main()
