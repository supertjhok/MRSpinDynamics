"""The conventions switch selects the physical alternatives together."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.conventions import (
    PROFILES,
    Conventions,
    conventions,
    get_conventions,
    set_conventions,
)
from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.core.kernels import (
    arb10_pulse_relaxation,
    get_arb10_pulse_relaxation,
)
from spin_dynamics.experiment import (
    CPMGTrain,
    Experiment,
    Acquisition,
    Phantom,
    Sample,
    CPMGImaging,
    load_run,
)
from spin_dynamics.workflows import run_ideal_cpmg_imaging, run_ideal_cpmg_train


def test_default_profile_is_matlab_parity() -> None:
    assert get_conventions() == PROFILES["matlab"] == Conventions()


def test_context_manager_switches_everything_and_restores() -> None:
    with conventions("physics") as active:
        assert active == PROFILES["physics"]
        assert get_arb10_pulse_relaxation() is True
    assert get_conventions() == PROFILES["matlab"]
    assert get_arb10_pulse_relaxation() is False


def test_partial_mapping_and_kernel_setter_are_reflected() -> None:
    with conventions({"echo_registration": "exact"}) as active:
        assert active.echo_registration == "exact"
        assert active.imaging_signal_convention == "matlab"
    # A mapping changes only the fields it names.
    with conventions("physics"), conventions({"echo_registration": "matlab"}) as active:
        assert active.echo_registration == "matlab"
        assert active.imaging_signal_convention == "cpmg"
        assert active.relax_during_pulses is True
    with arb10_pulse_relaxation():
        assert get_conventions().relax_during_pulses is True
    assert get_conventions() == PROFILES["matlab"]


def test_set_conventions_returns_previous() -> None:
    previous = set_conventions("physics")
    try:
        assert previous == PROFILES["matlab"]
    finally:
        assert set_conventions(previous) == PROFILES["physics"]


@pytest.mark.parametrize(
    "profile, error",
    [
        ("phyiscs", ValueError),
        ({"echo_registration": "dft"}, ValueError),
        ({"relax_during_pulses": 1}, TypeError),
    ],
)
def test_invalid_profiles_are_rejected(profile: object, error: type) -> None:
    with pytest.raises(error):
        set_conventions(profile)  # type: ignore[arg-type]
    assert get_conventions() == PROFILES["matlab"]


def test_echo_registration_follows_profile_and_explicit_wins() -> None:
    wvect = np.linspace(-5.0, 5.0, 11)
    spect = np.exp(-(wvect**2))
    exact = calc_time_domain_echo(spect, wvect, registration="exact")
    matlab = calc_time_domain_echo(spect, wvect, registration="matlab")
    with conventions("physics"):
        np.testing.assert_array_equal(calc_time_domain_echo(spect, wvect)[0], exact[0])
        np.testing.assert_array_equal(
            calc_time_domain_echo(spect, wvect, registration="matlab")[0], matlab[0]
        )
    np.testing.assert_array_equal(calc_time_domain_echo(spect, wvect)[0], matlab[0])


def test_imaging_options_follow_profile() -> None:
    rho = np.ones((3, 3))
    kwargs = dict(num_echoes=1, ny=3, t1_map=5e-3 * rho, t2_map=5e-3 * rho)
    physical = dict(signal_convention="cpmg", density_normalization="preserve")
    # The profile also relaxes during pulses, which the imaging kernels honor.
    with arb10_pulse_relaxation():
        explicit = run_ideal_cpmg_imaging(rho, **physical, **kwargs)
    with conventions("physics"):
        profiled = run_ideal_cpmg_imaging(rho, **kwargs)
    np.testing.assert_array_equal(profiled.kspace, explicit.kspace)
    rescaled = run_ideal_cpmg_imaging(rho, **physical, **kwargs)
    legacy = run_ideal_cpmg_imaging(rho, **kwargs)
    # Combine halves the signal and preserve divides by ny = 3.
    np.testing.assert_allclose(rescaled.kspace, legacy.kspace / 6.0, rtol=1e-12)


def test_pulse_relaxation_follows_profile() -> None:
    kwargs = dict(
        numpts=21, num_echoes=4, t1_seconds=5e-3, t2_seconds=2e-3, rephase_action="ignore"
    )
    with arb10_pulse_relaxation():
        explicit = run_ideal_cpmg_train(**kwargs)
    with conventions("physics"):
        profiled = run_ideal_cpmg_train(**kwargs)
    np.testing.assert_array_equal(profiled.echo_integrals, explicit.echo_integrals)
    legacy = run_ideal_cpmg_train(**kwargs)
    assert not np.allclose(legacy.echo_integrals, explicit.echo_integrals)


def test_runs_record_and_reproduce_under_their_conventions(tmp_path) -> None:
    experiment = Experiment(
        CPMGTrain(num_echoes=3),
        Sample(t1_seconds=5e-3, t2_seconds=2e-3),
        acquisition=Acquisition(numpts=21, rephase_action="ignore"),
    )
    with conventions("physics"):
        record = experiment.run()
    assert record.provenance["conventions"] == PROFILES["physics"].as_dict()
    path = record.save(str(tmp_path / "physics.npz"))
    # Reproduced under the recorded profile although "matlab" is active now.
    report = load_run(str(path)).verify_reproduction()
    assert report.matches, report.notes


def test_facade_imaging_fields_default_to_the_profile() -> None:
    sample = Sample(phantom=Phantom(rho=np.ones((3, 3))), t1_seconds=5e-3, t2_seconds=5e-3)
    with conventions("physics"):
        profiled = Experiment(CPMGImaging(num_echoes=1, ny=3), sample).run().result
    explicit_sequence = CPMGImaging(
        num_echoes=1, ny=3, signal_convention="cpmg", density_normalization="preserve"
    )
    with arb10_pulse_relaxation():
        explicit = Experiment(explicit_sequence, sample).run().result
    np.testing.assert_array_equal(profiled.kspace, explicit.kspace)
