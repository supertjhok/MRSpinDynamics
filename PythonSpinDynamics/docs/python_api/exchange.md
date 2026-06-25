# Chemical / Site Exchange

`spin_dynamics.exchange` adds Bloch-McConnell site exchange: a bath of
magnetically distinct sites that swap magnetization at finite kinetic rates
while each site relaxes with its own `T1`/`T2` and precesses at its own offset.
It is the relaxation-domain counterpart to the diffusion-exchange (DEXSY)
walker example and reuses the package's existing 2D inverse-Laplace solver for
the inverse problem.

## System definition

An `ExchangeSystem` holds a tuple of `ExchangeSite` objects and a square
`exchange_rates_hz` matrix whose entry `(i, j)` (for `i != j`) is the rate
constant `k_{i->j}` in inverse seconds. Site populations are normalized to sum
to one, and detailed balance (`p_i k_{i->j} = p_j k_{j->i}`) is checked unless
`balance="off"`.

```python
from spin_dynamics.exchange import two_site_exchange

system = two_site_exchange(
    offset_a_hz=0.0,
    offset_b_hz=0.0,
    k_ab_hz=8.0,
    population_a=0.5,      # derives the balanced backward rate
    t2_a_seconds=0.010,
    t2_b_seconds=0.200,
    labels=("fast", "slow"),
)
```

The kinetic generator `exchange_generator(system)` is column-conserving
(`dm/dt = X @ m`, columns sum to zero), so exchange alone preserves total
magnetization. `transverse_generator(system)` adds per-site offset precession
(`+i 2 pi offset`) and transverse relaxation for lineshape work.

## Lineshape exchange (coalescence)

`simulate_exchange_fid` integrates the transverse Bloch-McConnell equations and
`exchange_spectrum` returns the Fourier spectrum. Increasing the exchange rate
moves the system from two resolved lines (slow exchange) through broadening to a
single population-averaged line (fast exchange):

```python
from spin_dynamics.exchange import exchange_spectrum, two_site_exchange

slow = two_site_exchange(offset_a_hz=-120, offset_b_hz=120, k_ab_hz=2, k_ba_hz=2,
                         t2_a_seconds=0.5, t2_b_seconds=0.5)
fast = two_site_exchange(offset_a_hz=-120, offset_b_hz=120, k_ab_hz=4000, k_ba_hz=4000,
                         t2_a_seconds=0.5, t2_b_seconds=0.5)
freqs_slow, spec_slow = exchange_spectrum(slow)   # peaks near +/-120 Hz
freqs_fast, spec_fast = exchange_spectrum(fast)   # one peak near 0 Hz
```

## T2-T2 relaxation exchange (REXSY)

`simulate_relaxation_exchange_2d` builds the encode-mix-detect data set. During
the longitudinal mixing interval the `mixing_propagator` `G` redistributes
stored magnetization between sites (and optionally relaxes it with `T1`). The
forward model is

```text
S(t1, t2) = sum_b exp(-t2 / T2_b) sum_a G[b, a] p_a exp(-t1 / T2_a)
```

Inverting `S(t1, t2)` with `spin_dynamics.analysis.invert_t2_t2` yields a
`T2`-`T2` map: diagonal peaks for spins whose `T2` is unchanged across mixing,
and off-diagonal cross peaks for spins that changed site. The cross-peak
intensity grows with the exchange rate and mixing time, giving a direct readout
of exchange kinetics.

```python
import numpy as np
from spin_dynamics.analysis import invert_t2_t2
from spin_dynamics.exchange import simulate_relaxation_exchange_2d

encode = np.linspace(0.0, 0.06, 28)
detect = np.linspace(0.0, 0.8, 28)
rexsy = simulate_relaxation_exchange_2d(system, encode, detect, mixing_time=0.06)

t2_axis = np.logspace(-3, 0, 48)
ilt = invert_t2_t2(rexsy.data, encode, detect, t2_axis, regularization=1e-3)
exchange_map = ilt.distribution   # off-diagonal mass == exchange
```

See `examples/plot_t2_t2_exchange.py` for a runnable end-to-end demonstration.

## Scope and limits

- The model uses phenomenological per-site `T1`/`T2`, consistent with the
  package's other relaxation helpers; it is not a microscopic Redfield model.
- Exchange is first-order (linear kinetics) between an arbitrary number of
  sites supplied through the rate matrix.
- The REXSY forward model assumes refocused encode/detect trains (offsets do
  not dephase the stored amplitude) and exchange confined to the longitudinal
  mixing interval, the standard REXSY approximation. Use the transverse engine
  directly when offset evolution during encoding matters.
