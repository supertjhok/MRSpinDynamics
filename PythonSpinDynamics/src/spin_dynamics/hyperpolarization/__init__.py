"""Parahydrogen hyperpolarization and long-lived singlet-state primitives."""

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
    "ParahydrogenState",
    "parahydrogen_state",
    "singlet_order_amplitude",
    "singlet_order_operator",
    "singlet_population",
    "singlet_projector",
    "spin_pair_swap_operator",
    "triplet_population",
    "triplet_projector",
]
