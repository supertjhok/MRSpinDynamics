"""Regressions for experiment persistence: archive paths, config coverage and
randomness classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spin_dynamics.experiment import (
    CPMG,
    Acquisition,
    Experiment,
    Hardware,
    RxArray,
    RxCoil,
    SampledB0,
    SolenoidCoil,
    cli,
    load_run,
)
from spin_dynamics.experiment.config import (
    _registry,
    dumps_json,
    dumps_toml,
    experiment_from_config,
    experiment_to_config,
)
from spin_dynamics.experiment.provenance import (
    experiment_fingerprint,
    randomness_identity,
)
from spin_dynamics.experiment.serialization import registered_types
from spin_dynamics.receiver_network import ReceiverNetwork
from spin_dynamics.workflows.receiver_arrays import add_receiver_array_noise


def _small_run():
    return Experiment(sequence=CPMG(), acquisition=Acquisition(numpts=11)).run()


# --- archive paths ---------------------------------------------------------
def test_save_run_without_suffix_round_trips(tmp_path: Path) -> None:
    record = _small_run()
    written = record.save(str(tmp_path / "out"))
    assert written == tmp_path / "out.npz"
    assert written.exists()
    loaded = load_run(str(tmp_path / "out"))
    np.testing.assert_array_equal(loaded.result.masy, record.result.masy)
    assert loaded.result_matches is True
    # The explicit .npz spelling keeps working.
    assert load_run(written).result_matches is True


def test_cli_run_reports_the_written_archive(tmp_path: Path, capsys) -> None:
    config = tmp_path / "cpmg.json"
    config.write_text(
        json.dumps({"sequence": {"type": "CPMG"}, "acquisition": {"numpts": 11}}),
        encoding="utf-8",
    )
    output = tmp_path / "run_out"
    assert cli.main(["run", str(config), "-o", str(output)]) == 0
    printed = capsys.readouterr().out
    saved_line = [line for line in printed.splitlines() if line.startswith("saved: ")]
    assert saved_line == [f"saved: {tmp_path / 'run_out.npz'}"]
    assert Path(saved_line[0].removeprefix("saved: ")).exists()
    # The name the user passed is accepted by the follow-up commands.
    assert cli.main(["show", str(output)]) == 0


# --- randomness classification ----------------------------------------------
def test_receiver_array_noise_is_classified_by_its_seed() -> None:
    # Independent reference: the receiver-array noise generator is only
    # reproducible when a seed is given.
    data = np.zeros((2, 3, 4), dtype=np.complex128)
    covariance = np.eye(2)
    unseeded = [add_receiver_array_noise(data, covariance, seed=None) for _ in range(2)]
    seeded = [add_receiver_array_noise(data, covariance, seed=5) for _ in range(2)]
    assert not np.array_equal(*unseeded)
    np.testing.assert_array_equal(*seeded)

    def status(**acquisition):
        return randomness_identity(
            Experiment(sequence=CPMG(), acquisition=Acquisition(**acquisition))
        )["status"]

    assert status(receiver_noise_std=0.1) == "unseeded"
    assert status(receiver_noise_std=0.1, receiver_noise_seed=5) == "seeded"
    assert status(receiver_noise_covariance=np.eye(2)) == "unseeded"
    assert status() == "deterministic"


# --- config coverage ---------------------------------------------------------
def _network() -> ReceiverNetwork:
    impedance = np.array([[1.0 + 10.0j, 0.1 + 1.0j], [0.1 + 1.0j, 1.0 + 10.0j]])
    return ReceiverNetwork(
        frequency_hz=1.0e6,
        coil_impedance_ohm=impedance,
        load_impedance_ohm=50.0 + 0.0j,
    )


def _rich_experiments() -> list[Experiment]:
    coil = RxCoil(SolenoidCoil(radius_m=0.01, length_m=0.02, turns=4))
    return [
        Experiment(
            sequence=CPMG(),
            hardware=Hardware(
                b0=SampledB0(b0_tesla=np.full((2, 3, 3), 0.1), carrier_hz=4.0e6)
            ),
        ),
        Experiment(
            sequence=CPMG(),
            hardware=Hardware(rx_coil=RxArray(channels=(coil, coil), network=_network())),
        ),
        Experiment(
            sequence=CPMG(),
            acquisition=Acquisition(
                receiver_noise_covariance=np.array([[1.0, 0.2j], [-0.2j, 1.0]])
            ),
        ),
    ]


def test_every_serializable_type_is_config_writable() -> None:
    assert set(registered_types()) <= set(_registry())


def test_config_round_trip_matches_canonical_encoding() -> None:
    import tomllib

    for experiment in _rich_experiments():
        expected = experiment_fingerprint(experiment)
        config = experiment_to_config(experiment)
        from_toml = experiment_from_config(tomllib.loads(dumps_toml(config)))
        from_json = experiment_from_config(json.loads(dumps_json(config)))
        # Independent reference: the canonical tagged-JSON encoding.
        assert experiment_fingerprint(from_toml) == expected
        assert experiment_fingerprint(from_json) == expected
