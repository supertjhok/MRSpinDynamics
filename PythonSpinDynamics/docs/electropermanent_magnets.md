# Electropermanent AlNiCo magnets

PythonSpinDynamics models the static geometry and retained state of
electropermanent magnets (EPMs) before attempting pulse programming. This first
phase targets AlNiCo rods and close-packed rod bundles used as reconfigurable
`B0` sources.

## Why retained state is separate from material data

The nominal remanence of AlNiCo-5 is not automatically the operating remanence
of a finite EPM rod. Aspect ratio, self-demagnetizing field, partial
programming, neighboring magnets, temperature, and pulse history can all move
the retained state away from nominal material `Br`.

The API therefore separates:

- `AlNiCoMaterial`: nominal `Br`, coercivity, recoil permeability,
  conductivity, density, and optional tabulated field-solver B-H data;
- `RemanenceState`: signed retained axial remanence, reversal-point metadata,
  temperature, calibration identifier, uncertainty, and evidence;
- `ElectropermanentRod`: finite cylindrical geometry and orientation;
- `ElectropermanentBundle`: explicit rods plus bundle-level programming-coil
  metadata; and
- `EvidenceRecord`: measured, simulated, specified, or inferred provenance.

This distinction is important for AlNiCo. For example, the
`variable_field_nmr_rod()` preset has nominal material `Br = 1.27 T`, but its
default effective retained remanence is `0.33 T`, inferred from the published
field of approximately 150 mT at 1 mm from the rod face.

## Documented presets

| Preset | Geometry and evidence |
|---|---|
| `variable_field_nmr_rod()` | One 25.4 mm diameter, 152.4 mm long AlNiCo-5 rod; 50 turns of 14 AWG wire; nominal 25 uH. Geometry is specified by Ropp et al. (2019); effective retained remanence is explicitly inferred. |
| `weinberg_37_rod_bundle()` | 37 close-packed AlNiCo-5 rods, each 3.175 mm diameter and 101.6 mm long; approximately 60 turns of 16 AWG wire and 50 uH. Geometry comes from the 2020 Weinberg project records. The preset defaults to zero retained remanence because no single calibrated retained state applies to all archived tests. |
| `ALNICO5_AC500` | Nominal AC500 values from the archived magnet-coil calculator. |
| `ALNICO5_FEMM_2019` | AlNiCo-5 material record and 26-point B-H table recovered from the solved FEMM project. The table is static field-solver input, not a hysteresis loop. |

## Static field calculation

`electropermanent_field()` represents each finite rod by a volume cubature of
point dipoles. Rod axes may have any 3-D orientation, and fields from rods or
bundles superpose. Increase `n_cross` and `n_length` near magnet faces and check
convergence. The approximation is intended for points outside the magnetic
material.

For an axially magnetized cylinder, `finite_cylinder_on_axis_field()` provides
the exact uniform-magnetization axial result. It is useful for fast profiles
and numerical convergence tests.

```python
import numpy as np

from spin_dynamics.fields import (
    RemanenceState,
    finite_cylinder_on_axis_field,
    weinberg_37_rod_bundle,
)

state = RemanenceState(
    0.42,
    branch="partial",
    calibration_id="my-calibration-v1",
)
bundle = weinberg_37_rod_bundle(state=state)

points = np.array([
    [0.0, 0.0, 0.060],
    [0.010, 0.0, 0.060],
])
b_vector = bundle.field_vectors(points, n_cross=5, n_length=21)

equivalent = bundle.equivalent_cylinder()
b_axis = finite_cylinder_on_axis_field(0.060, equivalent)
```

The area-equivalent cylinder preserves total AlNiCo volume and magnetic dipole
moment. It is a useful far-field and optimization approximation, but it does
not preserve detailed near-field structure at the bundle face.

## Imaging and motion compatibility

`sample_electropermanent_field()` samples one or more sources on a 1-D, 2-D,
or 3-D Cartesian grid. Its result contains vector `B0`, projected `B0`, field
magnitude, magnitude-gradient, and proton Larmor-frequency maps.

```python
from spin_dynamics.fields import sample_electropermanent_field

maps = sample_electropermanent_field(
    (x_axis, z_axis),
    (bundle,),
    field_direction=(0, 0, 1),
)

motion_maps = maps.to_motion_field_maps()
imaging_maps = maps.to_spatial_field_maps(
    rho=phantom,
    t1_map=t1,
    t2_map=t2,
)
```

Both adapters convert tesla to angular-frequency off-resonance using the same
convention as existing workflows. By default, the projected field at the grid
sample nearest the origin is removed. Pass `reference_field_t=0` to retain
absolute angular frequency.

## Plotting example

The magnet-only example compares explicit rods with the equivalent cylinder,
shows partial-remanence states and neighbor contributions, and verifies
cubature against the exact finite-cylinder expression:

```powershell
python examples\plot_electropermanent_magnet.py \
  --remanence-t 0.42 \
  --output results\electropermanent_magnet.png
```

It prints the provenance class, retained fraction of nominal `Br`, bundle
dimensions and magnetic moment, inferred published surface-field benchmark,
equivalent-cylinder error, and the field contribution of an opposed neighbor.

![Static electropermanent-magnet model](images/example_electropermanent_magnet.png)

## Current limitations and next phase

This phase does **not** infer pulse-to-remanence behavior. In particular:

- `RemanenceState` records reversal history but does not yet update it;
- the static B-H table is not treated as a major or minor hysteresis loop;
- neighbor fields are calculated, but their effect on programming is not yet
  solved self-consistently;
- programming-coil R/L values are metadata rather than a pulse-power model; and
- conductive shielding is not yet connected to transient eddy-current loss.

The next phase adds the capacitor/H-bridge/RLC pulse waveform, followed by a
calibrated saturate-then-demagnetize state transition. That implementation will
use the archived 220/400/600 V oscilloscope traces and the measured
duty-cycle-versus-retained-field curve as separate validation fixtures.
