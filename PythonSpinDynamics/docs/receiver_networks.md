# Coupled receiver networks

Phase 4 connects the geometric receiver maps from reciprocity to an explicit
multiport electrical model. Coupling, loaded sensitivity, and correlated noise
come from the same impedance solve rather than from independent correction
factors.

## Multiport coil impedance

For simple conductor paths, `extract_multiport_impedance` generalizes the PEEC
chain formulation to several explicit ports:

\[
  Z_\mathrm{coil}(\omega)=R(\omega)+i\omega L(\omega).
\]

Each `Conductor` is one port. Its cross-section is divided into parallel
sub-filaments, and exact partial mutual inductances are evaluated between every
filament pair across all conductors. If \(B\) maps branch currents to terminal
currents, the terminal reduction is

\[
  Z_\mathrm{port}
  = \left(B^\mathsf{T}Z_\mathrm{branch}^{-1}B\right)^{-1}.
\]

```python
from spin_dynamics.fields import (
    conductor_from_segments,
    extract_multiport_impedance,
)

conductors = tuple(
    conductor_from_segments(
        channel.geometry.segments(),
        wire_radius=0.4e-3,
        n_radial=1,
        n_angular=4,
    )
    for channel in receivers.channels
)
ports = extract_multiport_impedance(conductors, [2.0e6])
z_coil = ports.impedance[0]
```

The result contains frequency-leading impedance, dissipative resistance, and
inductance matrices plus DC resistance and inductance limits. The extracted
matrix is complex symmetric for reciprocal geometry; its Hermitian dissipative
part is positive semidefinite for a passive conductor system.

## Loading and effective sensitivity

`ReceiverNetwork` represents one receive frequency. It accepts:

- the coil impedance matrix \(Z_c\);
- a series multiport impedance \(Z_s\) for tuning, matching, decoupling, and
  electrically short cable models;
- a load or preamplifier input impedance \(Z_L\);
- temperature and noise bandwidth;
- optional preamplifier voltage- and current-noise covariance densities.

Scalars and vectors expand to diagonal matrices. Full matrices allow reciprocal
cross-port elements. Every passive impedance must be reciprocal and have a
positive-semidefinite Hermitian dissipative part.

For open-circuit coil voltage \(e\),

\[
  i = (Z_c+Z_s+Z_L)^{-1}e,
  \qquad
  v_\mathrm{out}=Z_L i.
\]

Therefore the loaded transfer matrix is

\[
  H=Z_L(Z_c+Z_s+Z_L)^{-1},
\]

and the effective complex receive maps are

\[
  s_\mathrm{eff}(r)=H\,s_\mathrm{geom}(r).
\]

This transformation includes attenuation, phase rotation, and channel mixing.
The same effective maps are used by ordinary channel combination and Cartesian
SENSE.

```python
import numpy as np

from spin_dynamics.workflows import ReceiverNetwork

omega = 2.0 * 3.141592653589793 * 2.0e6
series_tuning = np.diag(-1j * np.imag(np.diag(z_coil)))
network = ReceiverNetwork(
    frequency_hz=2.0e6,
    coil_impedance_ohm=z_coil,
    series_impedance_ohm=series_tuning,
    load_impedance_ohm=50.0,
    temperature_k=293.15,
    noise_bandwidth_hz=10.0e3,
)

solution = network.solve(geometric_sensitivities)
loaded_maps = solution.effective_sensitivities
```

For diagonal series capacitors, use
`series_impedance_ohm=diag(-1j / (omega * capacitance))`. A reciprocal
off-diagonal impedance can represent a lumped decoupling element in the same
terminal model.

## Thermal and preamplifier noise

At uniform temperature, fluctuation-dissipation gives the passive output
covariance over bandwidth \(\Delta f\):

\[
  Z_\mathrm{out}
  =\left(Z_\mathrm{source}^{-1}+Z_L^{-1}\right)^{-1},
  \qquad
  \Psi_\mathrm{passive}
  =4k_B T\Delta f\,
    \frac{Z_\mathrm{out}+Z_\mathrm{out}^H}{2}.
\]

This includes the passive source, matching network, and load at the stated
temperature, including unequal loads. Preamplifier voltage noise is added at
the output. Current noise is transformed through the same source/load parallel
output impedance. `ReceiverNetworkSolution` reports each contribution, their
sum in \(\mathrm{V}^2\), and the complex channel correlation matrix.

The imaging Bloch kernel currently produces relative signal units, not absolute
induced volts. Consequently:

- `receiver_network_noise_covariance_v2` always preserves the physical
  circuit-derived covariance;
- with `receiver_noise_std=0`, the physical covariance supplies the relative
  Roemer/SENSE weighting but no synthetic noise is injected;
- with `receiver_noise_std > 0`, the physical covariance is scaled so its mean
  channel variance equals `receiver_noise_std**2`, preserving its correlation
  and unequal-channel variance structure.

An explicitly supplied `receiver_noise_covariance` is rejected when a network
is present because it would bypass the circuit-derived covariance.

## Experiment facade

Attach the network to the ordered `RxArray` ports:

```python
from spin_dynamics.experiment import (
    Acquisition,
    Hardware,
    ReceiverNetwork,
    RxArray,
)

receivers = RxArray(
    channels=(rx0, rx1),
    network=ReceiverNetwork(
        frequency_hz=2.0e6,
        coil_impedance_ohm=z_coil,
        series_impedance_ohm=z_series,
        load_impedance_ohm=50.0,
        noise_bandwidth_hz=10.0e3,
    ),
)

hardware = Hardware(rx_coil=receivers, plane=plane)
acquisition = Acquisition(receiver_noise_std=0.01, receiver_noise_seed=7)
```

The port count must equal the channel count. The network round-trips through
the experiment JSON representation. Receiver-array CPMG still uses
`probe="ideal"` because this explicit network owns the receive tuning and
loading; the older scalar `probe="tuned"` and `probe="matched"` paths describe
different single-channel acquisition kernels.

The receiver-array result now distinguishes:

- `geometric_receiver_sensitivities`: unit-current reciprocal maps;
- `receiver_sensitivities`: loaded effective maps used by reconstruction;
- `receiver_transfer_matrix`;
- source, total, and output impedance matrices;
- physical network noise covariance and correlation;
- the injected simulation covariance, when requested, in `noise_covariance`.

## Resonant modes

`coupled_resonant_modes(L, C)` solves the lossless series-resonator eigenproblem.
For two identical loops it reproduces the familiar split modes

\[
  f_\pm=\frac{1}{2\pi\sqrt{C(L\pm M)}}.
\]

This is useful for checking coupling and decoupling networks before a loaded
frequency-domain simulation.

## Runnable example

```bash
python examples/plot_receiver_network_coupling.py \
  --pixels 6 --frequency-mhz 2.0 --noise-std 0.015 \
  --output results/receiver_network_coupling.png
```

The example extracts a two-port PEEC matrix for two offset solenoids, tunes and
loads the ports, adds an illustrative shared sample-loss matrix, solves the
geometric and effective maps, runs CPMG imaging, and plots the transfer,
sensitivity, noise-correlation, and image effects.

## Current boundary

The multiport PEEC extractor currently gives one terminal port per simple
`Conductor` path and uses the path-constant chain formulation. `ReceiverNetwork`
is a dense, single-frequency terminal model. It can represent reciprocal
lumped series networks, loads, and preamplifier noise, but it does not yet
provide:

- arbitrary node/branch graphs;
- the per-segment full PEEC formulation across several conductors;
- distributed transmission lines or measured S-parameter import;
- automatic sample-loss impedance extraction;
- correlated preamplifier voltage/current cross spectra.

Those extensions can build on the same terminal API. Phase 5 adds the explicit
branched graph and prescribed-current reference model needed for birdcage
resonators.
