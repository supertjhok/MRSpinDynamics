"""Solver-neutral harmonic electromagnetic field interchange tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.harmonic import (
    HARMONIC_EM_SCHEMA,
    MU_0,
    HarmonicConvergence,
    HarmonicEMSolution,
    HarmonicFieldNormalization,
    HarmonicMaterial,
    HarmonicPort,
    HarmonicSolverProvenance,
    load_harmonic_em,
    load_harmonic_em_hdf5,
    load_harmonic_em_npz,
    save_harmonic_em,
    save_harmonic_em_hdf5,
    save_harmonic_em_npz,
)


def _solution(*, phasor_convention: str = "exp(+iwt)") -> HarmonicEMSolution:
    domain = SpatialDomain((np.array([-0.01, 0.01]), np.array([0.0, 0.02, 0.04])))
    shape = domain.shape + (3,)
    electric = np.zeros(shape, dtype=np.complex128)
    electric[..., 2] = 10.0 - 2.0j
    canonical_b = np.zeros(shape, dtype=np.complex128)
    canonical_b[..., 0] = 2.0e-6 + 4.0e-6j
    canonical_b[..., 1] = 6.0e-6 - 2.0e-6j
    if phasor_convention == "exp(-iwt)":
        electric = np.conjugate(electric)
        stored_b = np.conjugate(canonical_b)
    else:
        stored_b = canonical_b
    return HarmonicEMSolution(
        domain=domain,
        frequency_hz=128.0e6,
        phasor_convention=phasor_convention,  # type: ignore[arg-type]
        electric_field_v_per_m=electric,
        magnetic_flux_density_t=stored_b,
        normalization=HarmonicFieldNormalization(
            "per_ampere",
            reference_value=2.0,
            port_index=0,
            description="normalized to feed current",
        ),
        ports=(
            HarmonicPort(
                0,
                "feed",
                voltage_v=50.0 + 5.0j,
                current_a=1.0 + 0.1j,
                accepted_power_w=25.0,
                metadata={"mode": "lumped"},
            ),
        ),
        materials=(
            HarmonicMaterial(
                "sample",
                relative_permittivity=80.0,
                conductivity_s_per_m=0.5,
                region_id="sample-cylinder",
            ),
        ),
        provenance=HarmonicSolverProvenance(
            "reference",
            backend_version="1.2.3",
            source="unit-test",
            model_hash="abc123",
            metadata={"mesh": "rectilinear"},
        ),
        convergence=HarmonicConvergence(
            True,
            relative_residual=1.0e-8,
            iterations=42,
            mesh_cells=1234,
            minimum_cell_m=1.0e-3,
            maximum_cell_m=4.0e-3,
            energy_balance_relative_error=2.0e-3,
        ),
    )


def test_solution_validates_and_owns_read_only_field_arrays() -> None:
    solution = _solution()

    assert solution.field_shape == (2, 3, 3)
    assert solution.electric_field_v_per_m.dtype == np.complex128
    assert not solution.electric_field_v_per_m.flags.writeable
    assert not solution.domain.axes[0].flags.writeable
    with pytest.raises(ValueError):
        solution.electric_field_v_per_m[0, 0, 0] = 1.0


def test_phasor_conventions_produce_identical_canonical_fields_and_b1() -> None:
    positive = _solution(phasor_convention="exp(+iwt)")
    negative = _solution(phasor_convention="exp(-iwt)")

    np.testing.assert_allclose(positive.electric_field(), negative.electric_field())
    np.testing.assert_allclose(
        positive.magnetic_flux_density(),
        negative.magnetic_flux_density(),
    )
    np.testing.assert_allclose(positive.b1_plus(), negative.b1_plus())
    np.testing.assert_allclose(positive.b1_minus(), negative.b1_minus())


def test_b1_plus_and_minus_follow_package_circular_convention() -> None:
    solution = _solution()
    bx = 2.0e-6 + 4.0e-6j
    by = 6.0e-6 - 2.0e-6j

    np.testing.assert_allclose(solution.b1_plus(), 0.5 * (bx + 1.0j * by))
    np.testing.assert_allclose(solution.b1_minus(), 0.5 * (bx - 1.0j * by))


def test_h_only_solution_requires_or_uses_relative_permeability() -> None:
    domain = SpatialDomain((np.array([0.0, 1.0]),))
    shape = domain.shape + (3,)
    electric = np.zeros(shape, dtype=np.complex128)
    magnetic_h = np.ones(shape, dtype=np.complex128)
    no_mu = HarmonicEMSolution(
        domain,
        1.0e6,
        "exp(+iwt)",
        electric,
        magnetic_field_a_per_m=magnetic_h,
    )
    with pytest.raises(ValueError, match="relative_permeability"):
        no_mu.magnetic_flux_density()

    with_mu = HarmonicEMSolution(
        domain,
        1.0e6,
        "exp(+iwt)",
        electric,
        magnetic_field_a_per_m=magnetic_h,
        relative_permeability_map=np.array([1.0, 2.0]),
    )
    expected = MU_0 * np.broadcast_to(
        np.array([1.0, 2.0])[:, np.newaxis],
        shape,
    )
    np.testing.assert_allclose(with_mu.magnetic_flux_density(), expected)


def test_spatial_field_adapter_uses_nonnegative_circular_magnitudes() -> None:
    solution = _solution()
    maps = solution.to_spatial_field_maps(
        rho=1.0,
        t1_map=0.5,
        t2_map=np.full(solution.domain.shape, 0.1),
        b0_map=3.0,
        diffusion_map=1.0e-9,
        tx_scale=2.0,
        rx_scale=3.0,
    )

    assert maps.domain.shape == solution.domain.shape
    np.testing.assert_allclose(maps.b1_tx_map, 2.0 * np.abs(solution.b1_plus()))
    np.testing.assert_allclose(maps.b1_rx_map, 3.0 * np.abs(solution.b1_minus()))
    np.testing.assert_allclose(maps.rho, 1.0)
    np.testing.assert_allclose(maps.diffusion_map, 1.0e-9)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frequency_hz": 0.0}, "frequency_hz"),
        ({"phasor_convention": "unspecified"}, "phasor_convention"),
        ({"magnetic_flux_density_t": None}, "at least one"),
    ],
)
def test_solution_rejects_incomplete_or_ambiguous_physics(
    changes: dict[str, object],
    message: str,
) -> None:
    domain = SpatialDomain((np.array([0.0]),))
    arguments: dict[str, object] = {
        "domain": domain,
        "frequency_hz": 1.0e6,
        "phasor_convention": "exp(+iwt)",
        "electric_field_v_per_m": np.zeros((1, 3), dtype=np.complex128),
        "magnetic_flux_density_t": np.zeros((1, 3), dtype=np.complex128),
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        HarmonicEMSolution(**arguments)  # type: ignore[arg-type]


def test_npz_round_trip_preserves_fields_and_provenance(tmp_path) -> None:
    original = _solution()
    path = tmp_path / "field.npz"
    save_harmonic_em_npz(path, original)
    loaded = load_harmonic_em_npz(path)

    assert loaded.frequency_hz == original.frequency_hz
    assert loaded.phasor_convention == original.phasor_convention
    assert loaded.normalization == original.normalization
    assert loaded.ports == original.ports
    assert loaded.materials == original.materials
    assert loaded.provenance == original.provenance
    assert loaded.convergence == original.convergence
    np.testing.assert_allclose(
        loaded.electric_field_v_per_m,
        original.electric_field_v_per_m,
    )
    np.testing.assert_allclose(
        loaded.magnetic_flux_density_t,
        original.magnetic_flux_density_t,
    )


def test_npz_loader_rejects_unknown_schema(tmp_path) -> None:
    solution = _solution()
    valid_path = tmp_path / "valid.npz"
    invalid_path = tmp_path / "invalid.npz"
    save_harmonic_em_npz(valid_path, solution)
    with np.load(valid_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["schema"] = np.asarray("unknown/v99")
    np.savez(invalid_path, **arrays)

    with pytest.raises(ValueError, match="unsupported harmonic EM schema"):
        load_harmonic_em_npz(invalid_path)


def test_npz_metadata_is_json_not_pickle(tmp_path) -> None:
    path = tmp_path / "field.npz"
    save_harmonic_em_npz(path, _solution())
    with np.load(path, allow_pickle=False) as archive:
        assert str(archive["schema"].item()) == HARMONIC_EM_SCHEMA
        metadata = json.loads(str(archive["metadata_json"].item()))
    assert metadata["provenance"]["backend"] == "reference"


def test_extension_dispatch_round_trip_and_rejects_unknown_suffix(tmp_path) -> None:
    path = tmp_path / "field.npz"
    save_harmonic_em(path, _solution())
    loaded = load_harmonic_em(path)
    assert loaded.provenance.backend == "reference"

    with pytest.raises(ValueError, match="must use"):
        save_harmonic_em(tmp_path / "field.bin", _solution())
    with pytest.raises(ValueError, match="must use"):
        load_harmonic_em(tmp_path / "field.bin")


def test_hdf5_round_trip_when_h5py_is_available(tmp_path) -> None:
    pytest.importorskip("h5py")
    original = _solution(phasor_convention="exp(-iwt)")
    path = tmp_path / "field.h5"
    save_harmonic_em_hdf5(path, original)
    loaded = load_harmonic_em_hdf5(path)

    assert loaded.normalization == original.normalization
    assert loaded.ports == original.ports
    np.testing.assert_allclose(loaded.electric_field(), original.electric_field())
    np.testing.assert_allclose(loaded.b1_plus(), original.b1_plus())
