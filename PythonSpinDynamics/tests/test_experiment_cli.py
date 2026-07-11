"""Tests for config-driven runs and the experiment CLI."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.experiment import (
    Acquisition,
    CPMGImaging,
    CPMGTrain,
    ESRFID,
    Experiment,
    Hardware,
    ImagingPlane,
    NQRSLSE,
    PGSE,
    Phantom,
    Sample,
    SolenoidCoil,
    TxCoil,
    UniformB0,
    experiment_from_config,
    experiment_to_config,
    load_config,
    save_config,
)
from spin_dynamics.experiment import cli
from spin_dynamics.experiment.config import ConfigError, dumps_toml
from spin_dynamics.esr import ESRSpinSystem
from spin_dynamics.nqr import QuadrupolarSite

_DISC = ((np.linspace(-1, 1, 6)[:, None] ** 2 + np.linspace(-1, 1, 6)[None, :] ** 2) < 0.7).astype(
    float
)

_EXPERIMENTS = {
    "cpmg_train": Experiment(
        sequence=CPMGTrain(num_echoes=4),
        sample=Sample(t1_seconds=2.0, t2_seconds=2.0),
        hardware=Hardware(probe="tuned", q_value=40.0),
        acquisition=Acquisition(numpts=405, rephase_action="ignore"),
    ),
    "minimal": Experiment(sequence=CPMGTrain()),
    "pgse": Experiment(
        sequence=PGSE(
            num_echoes=2,
            gradient_amplitude=0.03,
            diffusion_time=16e-3,
        ),
        sample=Sample(diffusion_coefficient=1.8e-9, t2_seconds=0.1),
    ),
    "nqr": Experiment(
        sequence=NQRSLSE(
            pulse_duration_seconds=100e-6,
            nutation_hz=2.5e3,
            echo_spacing_seconds=1e-3,
            num_echoes=4,
            orientations="single",
        ),
        sample=Sample(site=QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)),
    ),
    "esr": Experiment(
        sequence=ESRFID(nutation_hz=25e6, pulse_duration_seconds=10e-9, acquisition_seconds=200e-9),
        sample=Sample(esr_system=ESRSpinSystem(g_tensor=(2.0, 2.0, 2.2))),
        hardware=Hardware(b0=UniformB0(field_tesla=0.35)),
    ),
    "imaging": Experiment(
        sequence=CPMGImaging(ny=5),
        sample=Sample(phantom=Phantom(rho=_DISC), t1_seconds=5e-3, t2_seconds=5e-3),
        hardware=Hardware(
            tx_coil=TxCoil(geometry=SolenoidCoil(radius_m=0.015, length_m=0.03, turns=10, axis="x")),
            plane=ImagingPlane(extent_m=(0.02, 0.02)),
        ),
    ),
}


@pytest.mark.smoke
@pytest.mark.parametrize("name", list(_EXPERIMENTS))
def test_config_mapping_round_trip(name: str) -> None:
    experiment = _EXPERIMENTS[name]
    rebuilt = experiment_from_config(experiment_to_config(experiment))
    assert rebuilt == experiment


@pytest.mark.smoke
@pytest.mark.parametrize("suffix", [".toml", ".json"])
@pytest.mark.parametrize("name", list(_EXPERIMENTS))
def test_config_file_round_trip(name: str, suffix: str, tmp_path) -> None:
    experiment = _EXPERIMENTS[name]
    path = tmp_path / f"{name}{suffix}"
    save_config(experiment, str(path))
    assert load_config(str(path)) == experiment


@pytest.mark.smoke
def test_minimal_config_is_readable() -> None:
    text = dumps_toml(experiment_to_config(_EXPERIMENTS["minimal"]))
    assert text.strip() == '[sequence]\ntype = "CPMGTrain"'


@pytest.mark.smoke
def test_config_default_sections_omitted() -> None:
    # ideal probe / default acquisition are the defaults -> not emitted
    config = experiment_to_config(Experiment(sequence=CPMGTrain(num_echoes=2)))
    assert set(config) == {"sequence"}


@pytest.mark.smoke
def test_config_unknown_type_errors() -> None:
    with pytest.raises(ConfigError, match="unknown spec type"):
        experiment_from_config({"sequence": {"type": "NotAThing"}})


@pytest.mark.smoke
def test_config_requires_sequence() -> None:
    with pytest.raises(ConfigError, match="sequence"):
        experiment_from_config({"sample": {"t1_seconds": 1.0}})


@pytest.mark.smoke
def test_config_rejects_bad_extension(tmp_path) -> None:
    with pytest.raises(ConfigError, match="must end in"):
        save_config(_EXPERIMENTS["minimal"], str(tmp_path / "x.yaml"))


def _write(tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.toml"
    save_config(_EXPERIMENTS[name], str(path))
    return str(path)


@pytest.mark.smoke
def test_cli_plan_ok(tmp_path, capsys) -> None:
    code = cli.main(["plan", _write(tmp_path, "cpmg_train")])
    assert code == 0
    assert "workflow: run_tuned_cpmg_train" in capsys.readouterr().out


@pytest.mark.smoke
def test_cli_plan_reports_errors(tmp_path, capsys) -> None:
    # NQR sequence without a site -> plan error -> exit 1
    (tmp_path / "bad.toml").write_text(
        '[sequence]\ntype = "NQRSLSE"\npulse_duration_seconds = 1e-4\n'
        "nutation_hz = 2500.0\necho_spacing_seconds = 1e-3\nnum_echoes = 4\n",
        encoding="utf-8",
    )
    code = cli.main(["plan", str(tmp_path / "bad.toml")])
    assert code == 1
    assert "sample.site" in capsys.readouterr().out


@pytest.mark.smoke
def test_cli_run_and_show(tmp_path, capsys) -> None:
    config = _write(tmp_path, "cpmg_train")
    out = tmp_path / "run.npz"
    code = cli.main(["run", config, "-o", str(out)])
    assert code == 0
    assert out.exists()
    assert "result type: CPMGTrainResult" in capsys.readouterr().out

    code = cli.main(["show", str(out)])
    assert code == 0
    shown = capsys.readouterr().out
    assert "run_tuned_cpmg_train" in shown
    assert "package_version" in shown


@pytest.mark.smoke
def test_cli_run_refuses_bad_plan(tmp_path, capsys) -> None:
    (tmp_path / "bad.toml").write_text(
        '[sequence]\ntype = "NQRSLSE"\npulse_duration_seconds = 1e-4\n'
        "nutation_hz = 2500.0\necho_spacing_seconds = 1e-3\nnum_echoes = 4\n",
        encoding="utf-8",
    )
    code = cli.main(["run", str(tmp_path / "bad.toml")])
    assert code == 1
    assert "not running" in capsys.readouterr().err


@pytest.mark.smoke
def test_cli_convert_toml_to_json(tmp_path) -> None:
    toml_path = _write(tmp_path, "nqr")
    json_path = tmp_path / "nqr.json"
    assert cli.main(["convert", toml_path, str(json_path)]) == 0
    assert load_config(str(json_path)) == _EXPERIMENTS["nqr"]


@pytest.mark.smoke
def test_cli_unknown_type_exit_code(tmp_path, capsys) -> None:
    (tmp_path / "x.toml").write_text('[sequence]\ntype = "Nope"\n', encoding="utf-8")
    code = cli.main(["plan", str(tmp_path / "x.toml")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.smoke
def test_shipped_example_config_plans_cleanly(capsys) -> None:
    from pathlib import Path

    config = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "experiment_config_cpmg.toml"
    )
    assert cli.main(["plan", str(config)]) == 0
    assert "run_tuned_cpmg_train" in capsys.readouterr().out


@pytest.mark.smoke
def test_shipped_pgse_config_plans_cleanly(capsys) -> None:
    from pathlib import Path

    config = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "experiment_config_pgse.toml"
    )
    assert cli.main(["plan", str(config)]) == 0
    output = capsys.readouterr().out
    assert "workflow: run_pgse_moment" in output
    assert "estimate:" in output
