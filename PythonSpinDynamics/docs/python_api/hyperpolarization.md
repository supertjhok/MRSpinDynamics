# Hyperpolarization and Singlet-State Primitives

`spin_dynamics.hyperpolarization` provides the shared quantum-state foundation
for long-lived singlet states, PHIP, and future SABRE workflows. The current
surface is deliberately limited to rigorously normalized state and observable
primitives; reaction and quantum-exchange workflows are documented as future
work in [the implementation plan](../phip_sabre_singlet_states_plan.md).

## Density conventions

Physical densities have unit trace. Parahydrogen helpers return their
trace-zero deviation density separately:

```python
from spin_dynamics.hyperpolarization import parahydrogen_state

state = parahydrogen_state(para_fraction=0.50)
state.density_matrix       # physical, unit trace
state.deviation_density    # trace zero
state.para_excess          # 0.50 - 0.25 = 0.25
```

Run `python examples/singlet_parahydrogen_states.py` for a compact table of
physical populations and deviation-order amplitude versus enrichment.

For para fraction `f_p`,

```text
rho = f_p P_S + (1 - f_p) P_T / 3
    = I / 4 + (f_p - 1/4) Q_S,
```

where `Q_S = P_S - P_T / 3`. Consequently, 25% para-H2 is the
unpolarized statistical mixture, not 25% of the maximum usable singlet order.

## Pair operators and observables

The singlet and triplet operators can be embedded in a larger spin-1/2 system:

```python
from spin_dynamics.hyperpolarization import (
    singlet_order_amplitude,
    singlet_order_operator,
    singlet_population,
    singlet_projector,
    triplet_projector,
)

projector = singlet_projector(nspin=4, pair=(1, 3))
order = singlet_order_operator(nspin=4, pair=(1, 3))
population = singlet_population(rho, pair=(1, 3))
amplitude = singlet_order_amplitude(deviation_rho, pair=(1, 3))
```

`singlet_population` requires a Hermitian, positive, unit-trace physical
density. `singlet_order_amplitude` accepts either a physical or trace-zero
deviation density and returns its Hilbert-Schmidt projection onto normalized
pair singlet order.

`spin_pair_swap_operator` exposes the pair-permutation symmetry directly. The
singlet subspace has swap eigenvalue -1, while every state in the triplet
subspace has swap eigenvalue +1.

## SLIC convention

For a nearly equivalent two-spin pair, SLIC uses a spin-lock nutation frequency
near `|J|`, at the mean pair resonance. The chemical-shift difference mixes the
singlet and triplet sectors and controls the transfer time. See
[J-Coupling Models](j_coupling.md) for the simulator.

## Current boundary

Implemented now:

- embedded singlet/triplet projectors and singlet order;
- spin-pair swap symmetry;
- physical singlet/triplet populations;
- parahydrogen physical and deviation states; and
- corrected analytical SLIC matching conventions.

Not implemented yet:

- singlet-state relaxation and complete preparation/storage/readout workflows;
- hydrogenation and PASADENA/ALTADENA;
- PH-INEPT and other heteronuclear transfer workflows; or
- quantum-manifold exchange and SABRE.
