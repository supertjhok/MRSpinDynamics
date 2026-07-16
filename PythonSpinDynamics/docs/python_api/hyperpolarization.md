# Hyperpolarization and Long-Lived Singlet Workflows

Ordinary thermal polarization stores excess population along the magnetic
field. Hyperpolarization and long-lived singlet experiments can instead store
order in correlated two-spin states, where independent Bloch vectors cannot
represent the physics.

Use this page to construct physical singlet/triplet and parahydrogen states,
simulate SLIC preparation and storage, or map parahydrogen order into
hydrogenative PHIP products. The implementations are dense reference models for
small spin systems. SABRE shares the state foundation but still requires its
own catalyst-bound/free-pool exchange workflow.

Read [J-Coupling Models](j_coupling.md) first if scalar-coupled Hamiltonians or
SLIC matching are unfamiliar.

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

## LLS preparation, storage, and readout

`simulate_slic_lls` performs a complete two-spin reference workflow. It starts
from transverse magnetization, applies the matched SLIC spin lock, optionally
purges non-singlet deviation operators, stores the selected mode with a
measured phenomenological `T_S`, and reconverts it with a second SLIC period:

```python
import numpy as np
from spin_dynamics.coupling import coupled_spin_system
from spin_dynamics.hyperpolarization import simulate_slic_lls

pair = coupled_spin_system(
    offsets_hz=[-0.35, 0.35],
    couplings_hz=[[0.0, 7.0], [7.0, 0.0]],
)
result = simulate_slic_lls(
    pair,
    storage_times_seconds=np.linspace(0.0, 60.0, 61),
    singlet_lifetime_seconds=18.0,
)
```

The storage channel preserves trace and Hermiticity. For a physical singlet it
relaxes the singlet population toward `1/4`; for a trace-zero prepared state it
damps the retained singlet mode as `exp(-t/T_S)`. This is an empirical LLS
lifetime model, not a prediction from independent-spin `T1` and `T2`.

## Hydrogenative PHIP

`hydrogenative_phip_state` maps a parahydrogen pair into explicitly selected
product sites. The deviation density scales as

```text
pairwise_addition_fraction * (para_fraction - 1/4) * Q_S.
```

`simulate_hydrogenative_phip` then supplies two distinct protocols:

- `pasadena` secularizes the newly formed state at high field, evolves it under
  the weak-coupling product Hamiltonian, applies a hard pulse, and acquires an
  antiphase FID;
- `altadena` requires an explicit sequence of `PHIPFieldSegment` values, so
  transport rate and field history are never silently assumed.

```python
from spin_dynamics.hyperpolarization import (
    PHIPFieldSegment,
    simulate_hydrogenative_phip,
)

ramp = [PHIPFieldSegment(scale, 1.5e-3) for scale in np.linspace(0.02, 1.0, 48)]
altadena = simulate_hydrogenative_phip(
    product_system,
    times_seconds=np.arange(1024) * 2.5e-4,
    protocol="altadena",
    para_fraction=0.90,
    pairwise_addition_fraction=0.72,
    reaction_time_seconds=0.05,
    field_trajectory=ramp,
    t2_seconds=0.12,
)
```

Hard pulses and detection can select arbitrary product indices. This supports
heteronuclear product systems at the density-matrix level, but a validated
PH-INEPT convenience sequence is still future work.

## Current boundary

Implemented now:

- embedded singlet/triplet projectors and singlet order;
- spin-pair swap symmetry;
- physical singlet/triplet populations;
- parahydrogen physical and deviation states; and
- corrected analytical SLIC matching conventions;
- SLIC preparation, empirical `T_S` storage, purge, and readout;
- pairwise-yield-aware hydrogenative product mapping;
- high-field PASADENA with product-basis secularization; and
- trajectory-defined ALTADENA transport, hard pulses, and selected-spin FIDs.

Not implemented yet:

- PH-INEPT and other heteronuclear transfer workflows; or
- quantum-manifold exchange and SABRE.

Run `python examples/plot_lls_phip_workflows.py` for the complete reference
workflow and [consult the implementation plan](https://github.com/supertjhok/MRSpinDynamics/blob/main/PythonSpinDynamics/docs/phip_sabre_singlet_states_plan.md)
for the remaining physics and validation roadmap.
