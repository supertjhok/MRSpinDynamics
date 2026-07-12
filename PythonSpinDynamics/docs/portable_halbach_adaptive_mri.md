# Capstone: a complete portable MRI design study in a 15–20 minute run budget

This capstone turns the low-cost C8 Halbach scanner in *Portable Low-Field MRI
Scanners* into one executable system model. A single run connects magnet and
coil geometry, PEEC electrical properties, ferrite RF loss, gradient-driver
requirements, receiver gain, thermal drift, noisy acquisition, compressed
sensing, and automatic stopping. The workflow is deliberately compact enough
to fit comfortably inside a 15–20 minute design-loop budget; the optimized
64×64 example completed in about 5.4 seconds on the development workstation,
leaving ample time for parameter sweeps and higher-resolution reruns.

The important distinction throughout this chapter is:

- **predicted** values come from field, circuit, thermal, or reconstruction
  equations;
- **book-calibrated** values are measured component parameters used as inputs;
- **validation targets** are never used to generate simulated noise or images.

## Run the capstone

From the `PythonSpinDynamics` directory:

```powershell
python examples\plot_portable_halbach_adaptive_mri.py `
  --matrix-size 64 `
  --output results\portable_halbach_adaptive_mri.png `
  --design-output results\portable_halbach_design_dashboard.png `
  --data-output results\portable_halbach_capstone.npz
```

The command prints RF, gradient, receiver, and mass tables; writes both figures;
and saves the pulse-length sweep and reconstruction histories to the NPZ file.

![End-to-end fields, heating, detuning, reconstruction, and stopping](images/portable_halbach_capstone_imaging.png)

*Figure 1. End-to-end system result. The noisy acquisition stops at 32% k-space
after held-out quality stops improving. The reference image is used only for the
reported NRMSE, never for the stopping decision.*

![RF, SNR, active-volume, gradient, receiver, and mass dashboard](images/portable_halbach_capstone_design.png)

*Figure 2. Designer dashboard generated from the same run. The 5–6 µs region
is close to the 200 W transmitter limit while retaining substantially more
active sample volume than longer, narrower-band pulses.*

## Model boundary and coordinates

The bore axis is `y`; the reconstructed plane is `x–z`. Gx and Gz are explicit
phase-encoding coils. There is no independently driven gradient along the bore
axis. The model therefore calls the thickness selected by the combination of
static-field inhomogeneity, RF bandwidth, and coil sensitivity the **effective
slice thickness**. This is not an ideal rectangular slice.

The active-volume metric counts water voxels whose combined transmit and
off-resonance excitation is at least 50% of the on-resonance center value. The
effective thickness is active cross-sectional area divided by the 9 mm sample
diameter. Both definitions are configurable analysis conventions rather than
new hardware specifications.

## Hardware represented

| Subsystem | Model | Source |
|---|---|---|
| C8 Halbach | Eight 10×10×100 mm ferrite blocks, radius 21.7 mm, Br=385 mT | Book Table 3.2 |
| Receive coil | 30-turn AWG-24 solenoid, radius 6.3 mm, length 24 mm | Book Table 4.4; PEEC solve |
| Transmit coil | Four-turn, 120° saddle pair, radius 7.7 mm, length 50 mm | Book Table 4.4; Biot–Savart field |
| Gx/Gz coils | Two-turn four-wire saddles, radius 10.3 mm, length 83 mm | Book Table 4.4; Biot–Savart field |
| Receiver | 200 kHz tuned response, 18 µs acquisition window, 3 dB noise figure | Book operating point plus explicit assumption |
| Thermal plant | RF coil, ferrite magnet, and ambient lumped nodes | System-level RC approximation |
| Reconstruction | Variable-density Cartesian sampling and L1-Haar FISTA | Dependency-free package implementation |

The measured 0.88 T/m construction gradient is superposed on the ideal C8
field. It is distinct from the actively driven 15 mT/m/A phase-encoding coils.

## RF coil and SNR results

The receive helix is solved with full PEEC proximity resolution. Copper-only
loss predicts Q≈217. Ferrite magnetic loss is added as

```text
R_ferrite = omega * mu0 * mu_r'' * integral_ferrite(|H_RF|^2 dV).
```

The book does not give complex C8 permeability at 4.158 MHz. By default,
`mu_r''` is inferred from the independently measured in-magnet receive Q=128;
the result is reported rather than hidden. Set
`ferrite_imaginary_relative_permeability` to measured material data to make
this step fully predictive.

| Property | Transmit | Receive copper-only | Receive ferrite-loaded |
|---|---:|---:|---:|
| Inductance | 3.20 µH | 2.99 µH | 2.99 µH |
| Series resistance | 1.115 Ω | 0.361 Ω | 0.610 Ω |
| Q at 4.158 MHz | 75 | 217 | 128 |
| Center B1/I | 0.374 mT/A | 1.355 mT/A | 1.355 mT/A |

The transmit L and Q are book-calibrated; its resistance follows `R=omega L/Q`.
Receive L and copper resistance are PEEC predictions. The ferrite contribution
is 0.250 Ω for the default inferred `mu_r''≈0.32`.

The predicted water echo at the tuned-probe output is 402 µV peak. The predicted
single-scan noise is 3.00 µV RMS, giving SNR≈134 for the 5 µs capstone run. The
measured reference is 84. That remaining factor is useful: it quantifies the
combined construction, receiver, transmit-noise coupling, and unmodeled loss
budget instead of silently forcing the simulation to match the experiment.

## Pulse-length design sweep

The drive current is changed at every point so the center of the transmit coil
receives a 90° rotation. Peak forward power uses the 50 Ω source convention

```text
I_peak = pi / (2 gamma B1_per_amp t90)
P_forward = I_peak^2 Z0 / 2.
```

| 90° pulse | Peak current | Peak forward power | Predicted SNR | Active volume | Effective slice |
|---:|---:|---:|---:|---:|---:|
| 3 µs | 5.24 A | 685 W | 160 | 0.613 mL | 6.81 mm |
| 4 µs | 3.93 A | 386 W | 148 | 0.581 mL | 6.46 mm |
| 5 µs | 3.14 A | 247 W | 134 | 0.513 mL | 5.70 mm |
| 6 µs | 2.62 A | 171 W | 119 | 0.440 mL | 4.89 mm |
| 8 µs | 1.96 A | 96 W | 96 | 0.337 mL | 3.75 mm |
| 10 µs | 1.57 A | 62 W | 84 | 0.279 mL | 3.10 mm |
| 12 µs | 1.31 A | 43 W | 74 | 0.230 mL | 2.56 mm |

This makes the hardware trade-off unusually clear: shortening the pulse excites
more of the inhomogeneous sample and increases SNR, but power rises as `1/t90²`.
A nominal 5–6 µs pulse is consistent with the reported 200 W-class transmitter;
3–4 µs operation would require a substantially larger peak-power stage.

## Gradient coil and driver results

The full nonlinear Gx and Gz fields are used in the forward encoding model.
Their center efficiencies independently reproduce the book value.

| Property | Gx | Gz |
|---|---:|---:|
| Simulated efficiency | 15.06 mT/m/A | 15.06 mT/m/A |
| Book value | 15 mT/m/A | 15 mT/m/A |
| Inductance | 1.30 µH | 1.30 µH |
| Resistance | 0.20 Ω | 0.20 Ω |

For the 64×64, 10 mm FOV acquisition and 1.6 ms encoding area:

| Driver requirement | Value |
|---|---:|
| Peak current | 3.12 A |
| Peak voltage, including 100 µs rise | 0.66 V |
| Peak winding I²R | 1.94 W |
| Average winding heat | 20.8 mW |

The voltage estimate is `IR + L dI/dt`. It excludes cable inductance, switch
overhead, current-regulator headroom, and protection margin; a practical driver
still benefits from the book's ±5 V, ±5 A compliance.

Ferrite eddy-current heating remains zero by default because hard ferrite has
low electrical conductivity. This is separate from ferrite **RF magnetic loss**,
which is included in receive Q and SNR.

## Receiver gain and ADC utilization

The default ADC full-scale voltage is 2.5 V. The target peak is one-half scale,
or 1.25 V, leaving headroom for drift, sample variation, and transients.

| Receiver quantity | Value |
|---|---:|
| Predicted probe peak | 402 µV |
| ADC target peak | 1.25 V |
| Required voltage gain | 3,110× |
| Required gain | 69.9 dB |

The calculation is `20 log10(V_target/V_probe)`. It is an amplitude budget, not
an automatic recommendation for one high-gain stage. A practical receiver
should distribute the gain around filtering, blanking, and overload recovery.

## Mass budget

| Item | Mass |
|---|---:|
| Magnet system | 0.6 kg |
| Transmitter | 0.7 kg |
| Other electronics | 0.6 kg |
| Batteries | 1.7 kg |
| Aluminum baseplate | 1.2 kg |
| **Total** | **4.8 kg** |
| **Without baseplate** | **3.6 kg** |

These are the measured system masses in Book Table 10.12, carried into the
design summary so electrical improvements can be evaluated against portability.

## Thermal, acquisition, and stopping result

At the default stop point:

| Quantity | Value |
|---|---:|
| Acquired k-space | 32% |
| Elapsed acquisition time | 4.37 min |
| RF coil temperature | 35.4 °C |
| Ferrite temperature | 25.9 °C |
| Larmor drift | −49.1 kHz |
| CS reference NRMSE | 0.355 |

A fixed, distributed validation set is acquired but withheld from FISTA. After
each batch, the reconstruction predicts those samples and reports

```text
quality = 1 / (1 + held-out k-space NRMSE).
```

The score is available during a real scan and uses the same noisy, thermally
detuned data as the reconstruction. Acquisition stops after quality fails to
make a material improvement for the configured patience interval, subject to a
minimum sampling fraction. Ground truth is used only after the simulation to
report `reference_nrmse`.

## Programmatic use

```python
from spin_dynamics.workflows.portable_halbach import (
    PortableHalbachMRIConfig,
    simulate_portable_halbach_mri,
    summarize_portable_halbach_design,
)

result = simulate_portable_halbach_mri(
    PortableHalbachMRIConfig(matrix_size=64)
)
design = summarize_portable_halbach_design(result)

print(design.receiver.required_gain_db)
print(design.gradients.peak_current_a)
print(design.pulse_sweep.peak_forward_power_w)
```

## Limitations and next measurements

- Transmit L and Q and gradient L/R are book-calibrated inputs; receive copper
  impedance is PEEC-predicted.
- Default ferrite `mu_r''` is inferred from loaded Q because complex-permeability
  data are unavailable. A measured C8 coupon sweep would remove that inference.
- The thermal RC network is system-level, not a fit to logged coil and magnet
  temperatures. Fit its capacities and conductances before safety decisions.
- Cable inductance, gradient-driver headroom, ADC anti-alias response, overload
  recovery, transmit leakage, and component tolerances are not yet Monte Carlo
  variables.
- Active volume and effective slice use a 50% excitation threshold. Report the
  threshold whenever comparing designs.
- This model is a design and validation tool, not a medical-device safety model.
