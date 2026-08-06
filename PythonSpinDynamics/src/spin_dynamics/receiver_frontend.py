"""Passive reciprocal two-port networks for receiver front ends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _abcd_to_s(abcd: np.ndarray, reference_impedance_ohm: float) -> np.ndarray:
    a, b = abcd[0]
    c, d = abcd[1]
    z0 = reference_impedance_ohm
    denominator = a + b / z0 + c * z0 + d
    if abs(denominator) <= np.finfo(np.float64).tiny:
        raise ValueError("ABCD matrix has a singular S-parameter conversion")
    determinant = a * d - b * c
    return np.array(
        [
            [
                (a + b / z0 - c * z0 - d) / denominator,
                2.0 * determinant / denominator,
            ],
            [
                2.0 / denominator,
                (-a + b / z0 - c * z0 + d) / denominator,
            ],
        ],
        dtype=np.complex128,
    )


@dataclass(frozen=True, eq=False)
class PassiveTwoPort:
    """Reciprocal passive two-port using the chain-matrix convention.

    The convention is ``[V1, I1].T = ABCD @ [V2, I2].T``, where ``I2`` flows
    from the two-port into its load. Passivity is checked through power-wave
    S-parameters at the positive real ``reference_impedance_ohm``.
    """

    abcd: np.ndarray
    reference_impedance_ohm: float = 50.0
    label: str = ""

    def __post_init__(self) -> None:
        matrix = np.asarray(self.abcd, dtype=np.complex128)
        reference = float(self.reference_impedance_ohm)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("abcd must be a finite 2-by-2 matrix")
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError(
                "reference_impedance_ohm must be finite and positive"
            )
        scale = max(1.0, float(np.max(np.abs(matrix))))
        determinant = np.linalg.det(matrix)
        if not np.isclose(determinant, 1.0, rtol=1.0e-10, atol=1.0e-12 * scale):
            raise ValueError("abcd must be reciprocal with determinant one")
        scattering = _abcd_to_s(matrix, reference)
        maximum_gain = float(np.linalg.svd(scattering, compute_uv=False)[0])
        if maximum_gain > 1.0 + 1.0e-10:
            raise ValueError("abcd must describe a passive two-port")
        object.__setattr__(self, "abcd", matrix)
        object.__setattr__(self, "reference_impedance_ohm", reference)
        object.__setattr__(self, "label", str(self.label))

    @property
    def s_parameters(self) -> np.ndarray:
        """Return equal-reference power-wave S-parameters."""

        return _abcd_to_s(self.abcd, self.reference_impedance_ohm)

    @property
    def matched_insertion_loss_db(self) -> float:
        """Return ``-20 log10(abs(S21))`` at the reference impedance."""

        magnitude = abs(self.s_parameters[1, 0])
        if magnitude == 0.0:
            return float("inf")
        return float(-20.0 * np.log10(magnitude))

    @property
    def is_lossless(self) -> bool:
        """Return whether the equal-reference scattering matrix is unitary."""

        scattering = self.s_parameters
        return bool(
            np.allclose(
                scattering @ scattering.conj().T,
                np.eye(2),
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        )

    def transformed_load_impedance_ohm(
        self,
        load_impedance_ohm: complex,
    ) -> complex:
        """Return the load impedance transformed to port 1."""

        load = _finite_complex(load_impedance_ohm, "load_impedance_ohm")
        a, b = self.abcd[0]
        c, d = self.abcd[1]
        denominator = c * load + d
        if abs(denominator) <= np.finfo(np.float64).tiny:
            raise ValueError("two-port load transformation is singular")
        return complex((a * load + b) / denominator)

    def voltage_transfer_to_load(
        self,
        load_impedance_ohm: complex,
    ) -> complex:
        """Return ``V2/V1`` with port 2 terminated by ``load_impedance_ohm``."""

        load = _finite_complex(load_impedance_ohm, "load_impedance_ohm")
        a, b = self.abcd[0]
        denominator = a * load + b
        if abs(denominator) <= np.finfo(np.float64).tiny:
            raise ValueError("two-port voltage transfer is singular")
        return complex(load / denominator)

    def cascade(self, *following: "PassiveTwoPort", label: str = "") -> "PassiveTwoPort":
        """Cascade this two-port with networks ordered from source to load."""

        matrix = self.abcd.copy()
        for network in following:
            if not isinstance(network, PassiveTwoPort):
                raise TypeError("following networks must be PassiveTwoPort values")
            if not np.isclose(
                network.reference_impedance_ohm,
                self.reference_impedance_ohm,
            ):
                raise ValueError("cascaded two-ports must use one reference impedance")
            matrix = matrix @ network.abcd
        return PassiveTwoPort(
            matrix,
            reference_impedance_ohm=self.reference_impedance_ohm,
            label=label,
        )


def identity_two_port(
    *,
    reference_impedance_ohm: float = 50.0,
    label: str = "identity",
) -> PassiveTwoPort:
    """Return an ideal through connection."""

    return PassiveTwoPort(
        np.eye(2, dtype=np.complex128),
        reference_impedance_ohm=reference_impedance_ohm,
        label=label,
    )


def series_impedance_two_port(
    impedance_ohm: complex,
    *,
    reference_impedance_ohm: float = 50.0,
    label: str = "series impedance",
) -> PassiveTwoPort:
    """Return the two-port for a passive series impedance."""

    impedance = _finite_complex(impedance_ohm, "impedance_ohm")
    if impedance.real < -1.0e-12 * max(1.0, abs(impedance)):
        raise ValueError("impedance_ohm must be passive")
    return PassiveTwoPort(
        np.array([[1.0, impedance], [0.0, 1.0]], dtype=np.complex128),
        reference_impedance_ohm=reference_impedance_ohm,
        label=label,
    )


def shunt_admittance_two_port(
    admittance_siemens: complex,
    *,
    reference_impedance_ohm: float = 50.0,
    label: str = "shunt admittance",
) -> PassiveTwoPort:
    """Return the two-port for a passive shunt admittance."""

    admittance = _finite_complex(admittance_siemens, "admittance_siemens")
    if admittance.real < -1.0e-12 * max(1.0, abs(admittance)):
        raise ValueError("admittance_siemens must be passive")
    return PassiveTwoPort(
        np.array([[1.0, 0.0], [admittance, 1.0]], dtype=np.complex128),
        reference_impedance_ohm=reference_impedance_ohm,
        label=label,
    )


def ideal_transformer_two_port(
    turns_ratio_primary_to_secondary: float,
    *,
    reference_impedance_ohm: float = 50.0,
    label: str = "ideal transformer",
) -> PassiveTwoPort:
    """Return an ideal transformer with ``V1/V2`` equal to the turns ratio."""

    ratio = float(turns_ratio_primary_to_secondary)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(
            "turns_ratio_primary_to_secondary must be finite and positive"
        )
    return PassiveTwoPort(
        np.array([[ratio, 0.0], [0.0, 1.0 / ratio]], dtype=np.complex128),
        reference_impedance_ohm=reference_impedance_ohm,
        label=label,
    )


def transmission_line_two_port(
    characteristic_impedance_ohm: float,
    electrical_length_rad: float,
    *,
    attenuation_db: float = 0.0,
    reference_impedance_ohm: float = 50.0,
    label: str = "transmission line",
) -> PassiveTwoPort:
    """Return a uniform reciprocal transmission line.

    ``attenuation_db`` is the one-way matched insertion loss of this line
    section. Thus a matched line has ``abs(S21)=10**(-attenuation_db/20)``.
    """

    characteristic = float(characteristic_impedance_ohm)
    phase = float(electrical_length_rad)
    attenuation = float(attenuation_db)
    if not np.isfinite(characteristic) or characteristic <= 0.0:
        raise ValueError(
            "characteristic_impedance_ohm must be finite and positive"
        )
    if not np.isfinite(phase):
        raise ValueError("electrical_length_rad must be finite")
    if not np.isfinite(attenuation) or attenuation < 0.0:
        raise ValueError("attenuation_db must be finite and non-negative")
    propagation = attenuation * np.log(10.0) / 20.0 + 1j * phase
    hyperbolic_cosine = np.cosh(propagation)
    hyperbolic_sine = np.sinh(propagation)
    return PassiveTwoPort(
        np.array(
            [
                [hyperbolic_cosine, characteristic * hyperbolic_sine],
                [hyperbolic_sine / characteristic, hyperbolic_cosine],
            ],
            dtype=np.complex128,
        ),
        reference_impedance_ohm=reference_impedance_ohm,
        label=label,
    )