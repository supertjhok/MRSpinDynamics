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

## Active LNA inputs

`ActiveReceiverNetwork` separates active input loading from amplifier noise.
For source impedance matrix \(Z_s\), diagonal active input impedance \(Z_{in}\),
and gain matrix \(G\),

\[
  H=Z_{in}(Z_s+Z_{in})^{-1},\qquad
  Z_p=(Z_s^{-1}+Z_{in}^{-1})^{-1}.
\]

Only the passive coil, sample, tuning, and matching network generates source
thermal noise:

\[
  \Psi_s=G H\,[4k_B T\Delta f\,\operatorname{Re}_H(Z_s)]\,H^H G^H.
\]

The LNA input resistance is not assigned a physical temperature. Instead an
`LNAInputModel` supplies voltage-noise spectrum \(C_{ee}\), current-noise
spectrum \(C_{ii}\), and their complex cross spectrum \(C_{ei}\). Their
input-node contribution is

\[
  C_{ee}+Z_pC_{ii}Z_p^H+C_{ei}Z_p^H+Z_pC_{ei}^H,
\]

which is then propagated through gain. Optional output-referred downstream
noise is added after gain. This prevents the common error of counting both the
active input resistance's Johnson noise and the LNA's specified noise.

```python
from spin_dynamics.workflows import ActiveReceiverNetwork, LNAInputModel

high_z = LNAInputModel(
    input_resistance_ohm=1.0e6,
    input_capacitance_f=5.0e-12,
    voltage_noise_density_v_per_sqrt_hz=1.2e-9,
    current_noise_density_a_per_sqrt_hz=5.0e-15,
    voltage_gain_v_per_v=100.0,
)
network = ActiveReceiverNetwork(
    frequency_hz=2.0e6,
    coil_impedance_ohm=z_coil,
    series_impedance_ohm=z_tuning,
    lna_input_models=high_z,
    noise_bandwidth_hz=10.0e3,
)
solution = network.solve(reciprocal_maps)
```

The solution reports every noise contribution, the loaded maps, input and
output transfer matrices, channel correlation, and per-channel noise figure.
`optimal_channel_snr` evaluates the covariance-optimal array SNR for a channel
vector or channel-leading map.
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
  --pixels 7 --frequency-mhz 2.0 --noise-std 0.25 \
  --output results/receiver_network_coupling.png
```

The example extracts a two-port PEEC matrix for two offset solenoids, tunes and
loads the ports, adds an illustrative shared sample-loss matrix, and solves the
geometric and effective maps plus circuit-derived covariance. Its compact CPMG
acquisition uses an odd DFT-matched grid. The figure includes the spin density
and clean/noisy Roemer reconstructions on one normalized intensity scale, along
with impedance, transfer, sensitivity, and noise-correlation diagnostics. A
printed scale-independent shape error guards the reconstruction.

## Resonant cancellation sweep

Phase 4.5 adds a physical shared R-L-C mesh branch and a passive frequency
sweep:

```python
from spin_dynamics.workflows import (
    analyze_receiver_coupling_sweep,
    mutual_cancellation_capacitance,
    shared_capacitor_mesh_impedance,
)

cancellation_capacitance = mutual_cancellation_capacitance(
    mutual_inductance_h,
    target_frequency_hz,
)
branch = shared_capacitor_mesh_impedance(
    frequencies_hz,
    cancellation_capacitance,
)
result = analyze_receiver_coupling_sweep(
    frequencies_hz,
    source_impedance_before_ohm,
    source_impedance_after_ohm + branch,
    load_impedance_ohm=50.0,
)
```

The branch contribution is \(Z_b q q^\mathsf{T}\), where \(q\) is its signed
mesh-incidence vector. At the design frequency,
\(C_d=1/(\omega_0^2|M|)\) cancels the mutual reactance. The sweep reports
residual mutual impedance, induced-current coupling, isolation improvement,
loaded transfer matrices, passive covariance, and noise correlation. Mutual
resistance remains and can therefore leave correlated noise after reactive
cancellation.

Run the study example with:

```bash
python examples/plot_receiver_resonant_cancellation.py \
  --frequency-mhz 2.0 --tolerance-percent 5 \
  --output results/receiver_resonant_cancellation.png
```

See the
[receiver decoupling and LNA study plan](receiver_decoupling_lna_study_plan.md)
for the subsequent robustness and active-front-end comparisons.

## Matched versus high-impedance active front ends

The comparison should be interpreted as an architecture trade rather than a
universal ranking:

| Architecture | Main advantages | Main costs and risks |
| --- | --- | --- |
| Matched 50 ohm | Standardized measurement and interconnect environment; predictable cable termination; low sensitivity to input capacitance; many low-noise devices are optimized near 50 ohm. | Resistive loading reduces open-circuit signal and loaded Q; matching components can add loss; a remote first stage incurs pre-LNA cable loss; loading alone is not the same as tuned preamplifier decoupling. |
| On-coil high-Z | Preserves coil voltage; strongly suppresses terminal current in this voltage-mode model; removes most pre-LNA cable length; very low current-noise devices can suit high source impedance. | Voltage noise can dominate a low-resistance coil; input capacitance limits impedance and bandwidth; large RF voltage raises linearity/recovery concerns; stability, biasing, heating, and transmit protection move onto the coil. |

A conventional MRI preamplifier-decoupling network is a third architecture:
a low device input impedance is transformed through a narrowband network into a
high impedance at the coil port. That transformation, along with cable phase,
is part of the next robustness stage rather than being approximated by either
row above.
Run the Phase 4.5 active-front-end study with:

```bash
python examples/plot_receiver_lna_architectures.py \
  --frequency-mhz 2.0 --points 61 --pixels 31 \
  --output results/receiver_lna_architectures.png
```

The example uses the same two-coil PEEC source and reciprocal maps for both
front ends. It compares input impedance, signal transfer, induced-current
coupling, noise figure, separated noise components, spatial array SNR, and the
high-Z/matched SNR ratio. Device noise values are deliberately illustrative.
The important result is the trade-off: a high input impedance can suppress
coil current and preserve open-circuit voltage while a larger voltage-noise
density can still reduce SNR. A high-Z LNA by itself is not a model of a
high-impedance-coil topology; the coil resonance and matching network remain
explicit parts of `source_impedance_ohm`.

## Current boundary

The multiport PEEC extractor currently gives one terminal port per simple
`Conductor` path and uses the path-constant chain formulation. `ReceiverNetwork`
is a dense, single-frequency terminal model. The passive Phase 4.5 sweep
compares frequency-leading terminal matrices and one physical shared mesh
branch. These APIs can represent reciprocal lumped series networks, loads, and
preamplifier noise, but they do not yet provide:

- arbitrary node/branch graphs;
- the per-segment full PEEC formulation across several conductors;
- distributed transmission lines, transformers, or measured S-parameter import;
- automatic sample-loss impedance extraction;
- standard LNA noise-parameter conversion, cross-channel active-noise
  covariance, stability, compression, and dynamic-range diagnostics;
- direct `Experiment`/Roemer/SENSE integration for `ActiveReceiverNetwork`.

Those extensions can build on the same terminal API. Phase 5 adds the explicit
branched graph and prescribed-current reference model needed for birdcage
resonators.
