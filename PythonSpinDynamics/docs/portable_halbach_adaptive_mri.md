# End-to-end portable Halbach MRI

`simulate_portable_halbach_mri` joins the package's finite-magnet field solver,
Biot-Savart RF and gradient-coil fields, PEEC receive-coil impedance, a lumped
thermal network, tuned receiver response, predicted receiver noise, incomplete
Cartesian acquisition, compressed-sensing
reconstruction, and a reference-free stopping rule.

The defaults are based on Chapter 10 of *Portable Low-Field MRI Scanners:
Hardware, Imaging Methods, and Applications*: 4.158 MHz operation, a 10 mm
field of view, the measured 0.88 T/m C8 intrinsic gradient, 5 us RF pulses,
100 ms scan spacing, two scans per complex phase-encode point, 18.5 W average
RF dissipation, and the ferrite magnet coefficient of -0.2 %/K. The reported
single-echo, single-scan SNR of approximately 84 is retained as a validation
target, not used to set the complex noise. The book's Section 11.3.1 comparison at
32% k-space motivates the L1 sparse reconstruction.

## What is modeled

1. The C8 magnet follows Table 3.2: eight 10 mm-square, 100 mm-long ferrite
   blocks at radius 21.7 mm with 385 mT remanence and the Halbach polarization
   pattern. Its mean field is normalized to the reported operating point; the
   measured first-order construction gradient is then included. The RF geometry
   follows Table 4.4: a 30-turn, 6.3 mm-radius,
   24 mm-long AWG-24 receive solenoid and a four-turn, 7.7 mm-radius, 50 mm-long,
   120-degree saddle transmitter.
2. The two gradients are the assembly-#1 two-turn, four-wire saddle windings
   (10.3 mm radius and 83 mm length). Biot-Savart predicts their full nonlinear
   fields and the forward acquisition encodes with those fields, rather than
   assuming an ideal FFT. The predicted central gain is close to the book's
   15 mT/m/A value. Current follows the 1.6 ms encoding area implied by the two
   800 us pulses, and gradient heating is calculated as duty-cycled `I^2 R`
   using the reported 0.2 ohm winding resistance.
3. A coil–magnet–ambient thermal network integrates heat from the RF and
   gradient electronics throughout the scan.
   Ferrite gradient eddy-current heat is zero by default because ferrite's low
   electrical conductivity makes that loss channel small.
4. Ferrite temperature changes shift the Larmor frequency. A 200 kHz tuned
   receiver converts that shift into complex amplitude and phase errors.
5. A full PEEC solve of the connected AWG-24 receive helix predicts inductance,
   copper AC resistance, proximity loss, and copper-only Q. Ferrite magnetic
   loss is added from `omega * mu0 * mu-r'' * integral(|H-rf|^2 dV)` over the
   eight physical blocks. Because the book does not report complex C8
   permeability at 4.158 MHz, the default infers `mu-r''` from the independently
   measured in-magnet receive Q=128; users with material data can set
   `ferrite_imaginary_relative_permeability` for a fully predictive calculation.
   The default separates copper-only Q (about 217) from ferrite-loaded Q (128)
   and reports the inferred loss resistance and `mu-r''` explicitly.
   Reciprocity predicts the water signal;
   Johnson noise, acquisition noise bandwidth, and receiver noise figure predict
   SNR. The default model gives the same order of SNR as the book's theoretical
   value of 160;
   the measured value of 84 remains lower for the same unmodeled effects noted
   in the book (receiver details, imperfect orthogonality/noise coupling, and
   additional construction loss). Complex noise is then added before CS, with
   the two-scan averaging applied explicitly.
6. FISTA solves an L1-Haar inverse problem from variable-density samples.

## Real-time auto-stop

A small, distributed validation set is acquired but withheld from FISTA. After
each acquisition batch, the current image predicts those samples. The live
quality score is

```text
quality = 1 / (1 + held-out k-space NRMSE)
```

This is available on a scanner because it compares acquired data with a model
prediction; it does not use the synthetic ground-truth image. Acquisition stops
after the score fails to make a configured material improvement for a chosen
number of batches, subject to a minimum sampling fraction. `reference_nrmse` is
reported only for validation of simulations and is never used by the stop rule.

Run the demonstration with:

```powershell
python examples\plot_portable_halbach_adaptive_mri.py `
  --output results\portable_halbach_adaptive_mri.png `
  --data-output results\portable_halbach_adaptive_mri.npz
```

The thermal RC values are a documented system-level approximation, not a fit to
an individual scanner's temperature log. Replace them with measured capacities
and conductances before using the result for hardware limits or patient safety.
