"""Parahydrogen hyperpolarization and long-lived singlet-state workflows."""

from spin_dynamics.hyperpolarization.lls import (
    SLICLongLivedStateResult,
    simulate_slic_lls,
    store_singlet_order,
)
from spin_dynamics.hyperpolarization.phip import (
    CouplingRegime,
    HydrogenativePHIPState,
    PHIPAcquisitionResult,
    PHIPFieldSegment,
    PHIPProtocol,
    acquire_phip_fid,
    apply_hard_pulse,
    evolve_phip_field_trajectory,
    hydrogenative_phip_state,
    secularize_high_field_product,
    simulate_hydrogenative_phip,
)

from spin_dynamics.hyperpolarization.singlet import (
    ParahydrogenState,
    parahydrogen_state,
    singlet_order_amplitude,
    singlet_order_operator,
    singlet_population,
    singlet_projector,
    spin_pair_swap_operator,
    triplet_population,
    triplet_projector,
)

__all__ = [
    "CouplingRegime",
    "HydrogenativePHIPState",
    "PHIPAcquisitionResult",
    "PHIPFieldSegment",
    "PHIPProtocol",
    "ParahydrogenState",
    "SLICLongLivedStateResult",
    "acquire_phip_fid",
    "apply_hard_pulse",
    "evolve_phip_field_trajectory",
    "hydrogenative_phip_state",
    "parahydrogen_state",
    "secularize_high_field_product",
    "simulate_hydrogenative_phip",
    "simulate_slic_lls",
    "singlet_order_amplitude",
    "singlet_order_operator",
    "singlet_population",
    "singlet_projector",
    "spin_pair_swap_operator",
    "store_singlet_order",
    "triplet_population",
    "triplet_projector",
]
