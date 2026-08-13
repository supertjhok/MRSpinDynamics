# Full-wave validation

Full-wave solver completion is not evidence that a field map is quantitatively
reliable. PythonSpinDynamics therefore keeps validation separate from field
import and promotes `HarmonicConvergence.converged` only after every required
check in a `FullWaveValidationReport` passes.

## Phase 4 checks

The validation layer provides five complementary checks:

1. **Time-domain termination.** Parse the openEMS mesh, timestep, iteration
   count, and final energy decay from retained solver logs. Reaching the energy
   criterion is necessary but does not establish mesh convergence.
2. **Mesh convergence.** Interpolate complex E and B from successive meshes at
   common physical probe points and compare relative RMS changes. For the
   loaded reference, probes are inside the sample with a boundary exclusion
   equal to the coarsest requested cell size; this avoids treating the
   discontinuous dielectric-interface E field as a pointwise-convergent value.
   Normalization, frequency, and probe locations must be identical.
3. **Accepted power and conductive loss.** Dump both E and conduction-current
   density J, integrate `0.5 Re(E dot J*)`, and require that volume loss not
   exceed accepted port power. The nonnegative remainder can include radiation,
   conductor loss, and PML absorption, so this is a hard consistency bound rather
   than an invented exact balance.
4. **Reciprocity.** Run a two-port model twice with the source and receiver
   interchanged, then compare the complex transfer impedances.
5. **Low-frequency analytical limit.** Compare the unloaded loop-center B/I
   magnitude with `mu0/(2R)` from Biot--Savart.

Use `apply_validation_report()` to attach the report to a copied harmonic
solution. A failed required check remains embedded in the saved provenance; it
is never converted into a warning-only success.

## Reproducing the live suite

With the Linux openEMS environment installed during Phase 3/4:

```bash
source ~/opt/openEMS/venv/bin/activate
python examples/validate_openems_fullwave.py \
  --output .tmp/openems_phase4_resolved \
  --python ~/opt/openEMS/venv/bin/python \
  --run
```

The default loaded mesh series is 3.0, 2.5, and 2.0 mm. The finest case has
about three million FDTD cells, so the full workflow takes several minutes.
Rerun the report without `--run` to reuse existing solver output.

## Current reference result (2026-08-13)

The corrected reference uses a 6 mm wide planar PEC annular strip, an
area-resolved feed gap, and an x-axis sample cylinder that does not intersect
the conductor. All implemented Phase 4 checks pass in the declared
loaded-sample interior region:

| Check | Result | Evidence |
| --- | --- | --- |
| FDTD energy termination | Pass | All reported live cases reached the requested -50 dB criterion. |
| E/J loss consistency | Pass | At 2.0 mm, conductive loss was 94.83% of accepted power. |
| Two-port reciprocity | Pass | Reciprocal transfer impedances differed by 0.527%. |
| Loaded mesh convergence | Pass | For 2.5 to 2.0 mm, E changed 1.19% and B changed 0.81% across 3,363 sample-interior probes, below 5%. |
| 32 MHz Biot--Savart limit | Pass | Loop-center B/I differed from `mu0/(2R)` by 1.88% at `ka = 0.0436`. |

The report records a 3 mm exclusion from the sample boundary. Mesh convergence
over the complete rectangular dump remains unsuitable as a gate because it
includes air and points on a staircased dielectric interface where normal E is
discontinuous. The saved solution is therefore quantitative only for the
stated sample-interior region, not for dielectric-interface E.

An independent published or cross-solver high-frequency field benchmark still
remains before broadening this result beyond the reference geometry. The
original thin-wire and 3 mm-strip diagnostic cases remain useful negative
controls but are not quantitative references.
