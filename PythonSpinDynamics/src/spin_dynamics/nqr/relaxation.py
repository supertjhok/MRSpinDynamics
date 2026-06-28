"""Backward-compatible NQR imports for shared relaxation helpers."""

from __future__ import annotations

from spin_dynamics.relaxation import (
    DipolarRelaxationSource,
    IsotropicLiquidMotionalAveraging,
    MotionalAveragingModel,
    NQRRelaxationLike,
    NQRRelaxationModel,
    NQRRelaxationSuperoperator,
    PhenomenologicalRelaxationModel,
    RedfieldDipolarRelaxationModel,
    RelaxationModelLike,
    RelaxationSuperoperator,
    RigidSolidMotionalAveraging,
    cycle_superoperator,
    dipolar_coupling_hz,
    dipolar_coupling_tensor,
    effective_decay_time,
    liouville_hamiltonian,
    liouville_superoperator,
    matrix_exponential,
    propagate_density_liouville,
    relaxation_superoperator,
)

__all__ = [
    "DipolarRelaxationSource",
    "IsotropicLiquidMotionalAveraging",
    "MotionalAveragingModel",
    "NQRRelaxationLike",
    "NQRRelaxationModel",
    "NQRRelaxationSuperoperator",
    "PhenomenologicalRelaxationModel",
    "RedfieldDipolarRelaxationModel",
    "RelaxationModelLike",
    "RelaxationSuperoperator",
    "RigidSolidMotionalAveraging",
    "cycle_superoperator",
    "dipolar_coupling_hz",
    "dipolar_coupling_tensor",
    "effective_decay_time",
    "liouville_hamiltonian",
    "liouville_superoperator",
    "matrix_exponential",
    "propagate_density_liouville",
    "relaxation_superoperator",
]
