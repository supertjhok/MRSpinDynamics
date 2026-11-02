"""Property-based checks of physical invariants shared across the package.

Most code-review bugs broke an invariant that holds for every input, not one
example: trace preservation (L-58), the CPMG-IR sign (M-14), a magnetization
above its equilibrium value (M-16), detailed balance. Hypothesis draws the
inputs. The default profile is derandomized so CI is reproducible; set
``HYPOTHESIS_PROFILE=thorough`` for a longer randomized search.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from spin_dynamics.core.echo import calc_time_domain_echo  # noqa: E402
from spin_dynamics.core.rotations import calc_rotation_matrix  # noqa: E402
from spin_dynamics.dnp import (  # noqa: E402
    CoupledElectronNuclearModel,
    ElectronNuclearTransitionRates,
)
from spin_dynamics.nqr import (  # noqa: E402
    FieldDependentRelaxationModel,
    QuadrupolarSite,
    nqr_hamiltonian,
    simulate_field_relaxation,
)
from spin_dynamics.parameters import set_params_ideal  # noqa: E402
from spin_dynamics.workflows import (  # noqa: E402
    run_ideal_cpmg_ir_train,
    run_ideal_cpmg_train,
)

settings.register_profile("ci", max_examples=25, deadline=None, derandomize=True)
settings.register_profile("thorough", max_examples=300, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

_finite = dict(allow_nan=False, allow_infinity=False)


# Offsets whose square underflows (|offset| < ~1e-154) lose the effective-field
# direction in sqrt(w1**2 + offset**2); they are far outside any physical range.
_offset = st.floats(-20.0, 20.0, **_finite).filter(lambda x: x == 0.0 or abs(x) > 1e-100)


@given(
    offsets=st.lists(_offset, min_size=1, max_size=6),
    pulses=st.lists(
        st.tuples(
            st.floats(0.0, 4.0 * np.pi, **_finite),  # duration (normalized)
            st.floats(-np.pi, np.pi, **_finite),  # phase
            st.floats(0.0, 2.0, **_finite),  # relative amplitude
        ),
        min_size=1,
        max_size=5,
    ),
    magnetization=st.tuples(*(st.floats(-1.0, 1.0, **_finite) for _ in range(3))),
)
def test_composite_pulses_rotate_without_changing_length(offsets, pulses, magnetization):
    tp, phi, amp = (np.array(values) for values in zip(*pulses))
    r = calc_rotation_matrix(np.array(offsets), 1.0, tp, phi, amp)
    mx, my, mz = magnetization
    m0, mm, mp = mz, mx - 1j * my, mx + 1j * my
    new_0 = r.R_00 * m0 + r.R_0m * mm + r.R_0p * mp
    new_m = r.R_m0 * m0 + r.R_mm * mm + r.R_mp * mp
    new_p = r.R_p0 * m0 + r.R_pm * mm + r.R_pp * mp
    # A real vector stays real (M- and M+ conjugate) and keeps its length.
    np.testing.assert_allclose(new_0.imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(new_m, np.conj(new_p), atol=1e-12)
    length = mx * mx + my * my + mz * mz
    np.testing.assert_allclose(new_0.real**2 + np.abs(new_p) ** 2, length, atol=1e-11)


_SPIN1 = QuadrupolarSite(
    spin=1.0, isotope="14N", quadrupole_frequency_hz=900.0e3, eta=0.3,
    gamma_hz_per_t=3.0766e6,
)


@given(
    field=st.tuples(*(st.floats(-0.1, 0.1, **_finite) for _ in range(3))),
    temperature=st.floats(1e-5, 1e-3, **_finite),
    thermalization=st.floats(1e-4, 1e-2, **_finite),
    dephasing=st.floats(1e-4, 1e-2, **_finite),
    state=st.lists(st.floats(-1.0, 1.0, **_finite), min_size=6, max_size=6),
    time=st.floats(0.0, 2e-2, **_finite),
)
def test_relaxation_keeps_a_physical_density_matrix(
    field, temperature, thermalization, dephasing, state, time
):
    vector = np.array(state[:3]) + 1j * np.array(state[3:])
    if np.linalg.norm(vector) < 1e-3:
        vector = np.array([1.0, 0.0, 0.0], dtype=complex)
    vector /= np.linalg.norm(vector)
    model = FieldDependentRelaxationModel(
        temperature_kelvin=temperature,
        thermalization_time_seconds=thermalization,
        dephasing_time_seconds=min(dephasing, thermalization),
    )
    hamiltonian = nqr_hamiltonian(_SPIN1, field)
    generator = model.superoperator(hamiltonian)
    gibbs = model.equilibrium_density(hamiltonian).reshape(-1, order="F")
    scale = np.max(np.abs(generator))
    np.testing.assert_allclose(generator @ gibbs / scale, 0.0, atol=1e-12)

    result = simulate_field_relaxation(
        _SPIN1, field, [0.0, time], relaxation=model,
        initial_density=np.outer(vector, vector.conj()),
    )
    for density in result.density_matrices_pas:
        assert abs(np.trace(density) - 1.0) < 1e-10
        np.testing.assert_allclose(density, density.conj().T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(density)) > -1e-10


_rate = st.floats(0.0, 1.0, **_finite)


@given(
    field=st.floats(0.05, 1.0, **_finite),
    temperature=st.floats(1.0, 300.0, **_finite),
    rates=st.tuples(_rate, _rate, _rate, _rate),
    hyperfine=st.floats(-5e5, 5e5, **_finite),
)
def test_overhauser_gibbs_state_is_stationary(field, temperature, rates, hyperfine):
    model = CoupledElectronNuclearModel(
        field,
        temperature_kelvin=temperature,
        secular_hyperfine_hz=hyperfine,
        transitions=ElectronNuclearTransitionRates(*rates),
    )
    density = model.thermal_density()
    assert abs(np.trace(density) - 1.0) < 1e-12
    assert np.min(np.linalg.eigvalsh(density)) > 0.0
    generator = model.liouvillian()
    stationary = generator @ density.reshape(-1, order="F")
    np.testing.assert_allclose(stationary / np.max(np.abs(generator)), 0.0, atol=1e-12)


_train = dict(
    t1=st.floats(1e-3, 0.5, **_finite),
    t2_fraction=st.floats(0.05, 1.0, **_finite),
    num_echoes=st.integers(1, 4),
)


@settings(max_examples=8)
@given(**_train)
def test_ir_train_after_full_recovery_is_the_cpmg_train(t1, t2_fraction, num_echoes):
    t2 = t1 * t2_fraction
    _sp, pp = set_params_ideal(numpts=3)
    period = pp.preDelay + pp.T_180 + pp.postDelay
    common = dict(numpts=21, t1_seconds=t1, t2_seconds=t2, rephase_action="ignore")
    cpmg = run_ideal_cpmg_train(num_echoes=num_echoes, **common)
    ir = run_ideal_cpmg_ir_train(
        num_echoes=num_echoes, echo_spacing_seconds=period, tauvect=[40.0 * t1], **common
    )
    np.testing.assert_allclose(
        ir.echo_integrals[0], cpmg.echo_integrals, rtol=1e-6, atol=1e-9
    )


@settings(max_examples=8)
@given(**_train)
def test_echo_trains_never_exceed_equilibrium_magnetization(t1, t2_fraction, num_echoes):
    common = dict(numpts=21, t1_seconds=t1, t2_seconds=t1 * t2_fraction, rephase_action="ignore")
    cpmg = run_ideal_cpmg_train(num_echoes=num_echoes, **common)
    assert np.max(np.abs(cpmg.mrx)) <= 1.0 + 1e-12
    ir = run_ideal_cpmg_ir_train(
        num_echoes=num_echoes, tauvect=[1e-4, t1, 10.0 * t1], **common
    )
    assert np.max(np.abs(ir.mrx)) <= 1.0 + 1e-12


@given(
    n=st.integers(2, 24),
    spacing=st.floats(0.05, 2.0, **_finite),
    zero_fill=st.integers(1, 4),
    seed=st.integers(0, 2**32 - 1),
)
def test_exact_echo_registration_is_the_direct_sum(n, spacing, zero_fill, seed):
    rng = np.random.default_rng(seed)
    wvect = spacing * (np.arange(n) - n // 2)
    spect = rng.normal(size=n) + 1j * rng.normal(size=n)
    echo, times = calc_time_domain_echo(
        spect, wvect, zero_fill=zero_fill, registration="exact"
    )
    direct = np.exp(1j * np.outer(times, wvect)) @ spect / n
    np.testing.assert_allclose(echo, direct, rtol=1e-9, atol=1e-9 * np.abs(spect).sum())
