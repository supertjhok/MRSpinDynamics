# Sequence IR and Pulseq Interchange

A pulse sequence must mean the same thing to a timeline plotter, a simulator,
an exported Pulseq file, and eventually hardware. The sequence intermediate
representation (IR) is the shared description that prevents each backend from
inventing its own timing and unit conventions.

Use this page when importing Pulseq, constructing general RF/gradient/ADC
timelines, or adapting one sequence to multiple execution engines. For a
prebuilt CPMG, imaging, NQR, or ESR experiment, start with the workflow catalog
instead.

The IR follows the block model of the open, vendor-independent
[Pulseq specification](https://pulseq.github.io/specification.pdf): blocks
execute sequentially, while RF, X/Y/Z gradients, and ADC events may overlap
within a block.

The IR deliberately preserves Pulseq's physical units:

- RF envelopes: complex Hz;
- gradients: Hz/m;
- event timing: seconds;
- RF/receiver phase: radians;
- frequency offsets: Hz plus optional PPM contributions.

Backends own unit conversion. For example, `compiled_to_motion_steps` converts
RF and gradient frequencies to angular units for the moving-isochromat engine.

Probe realization is an execution policy, not a second waveform convention.
`SequenceIR.hardware_effects` is a `HardwareEffectsPolicy` with independent
`transmit` and `receive` modes (`"ignore"` or `"apply"`). The nominal RF and
ADC events are unchanged in either mode. A probe-aware backend applies its
separately supplied transfer model when requested; a backend that cannot do so
must reject `"apply"` rather than silently simulate ideal hardware. This also
makes ideal/probe-aware A/B comparisons reuse exactly the same IR.

## Initial API

```python
import dataclasses

from spin_dynamics.sequences import (
    HardwareEffectsPolicy,
    compile_sequence,
    compiled_to_motion_steps,
    read_pulseq,
    write_pulseq,
)

sequence = read_pulseq("experiment.seq")
sequence = dataclasses.replace(
    sequence,
    hardware_effects=HardwareEffectsPolicy(transmit="apply", receive="apply"),
)
compiled = compile_sequence(sequence, system_frequency_hz=2.0e6)
motion_steps = compiled_to_motion_steps(compiled, spatial_dimensions=3)

# Both the source IR and compiled timeline expose the same plotting hook.
figure, axes = sequence.plot(show_blocks=True)

# Export a native or imported IR through the standard Pulseq 1.5.0 format.
write_pulseq(sequence, "canonical.seq")
```

The policy is MRSpinDynamics execution metadata and is not written as a
non-standard Pulseq extension; reading a `.seq` file therefore defaults both
paths to `"ignore"`. Attach the desired policy after import, as above.

Native sequences can be constructed from `SequenceIR`, `SequenceBlock`,
`RFPulse`, `GradientWaveform`, and `ADCEvent`. `compile_sequence` partitions
blocks at event raster edges and ADC sample centers, then produces a
piecewise-constant RF/gradient timeline with exact receive-sample metadata.

## Execute Through the Experiment Facade

`SequenceIRExecution` makes the compiled moving-isochromat target available
through the same `Experiment.plan()` / `run()` / provenance workflow as the
specialized sequences. The spatial model is explicit: `SequenceDomain` carries
one to three physical axes, spin density, optional B0/B1 maps, constant
velocity, and the mapping from its dimensions to physical gradient channels.

```python
import numpy as np
from spin_dynamics.experiment import (
    Experiment, Sample, SequenceDomain, SequenceIRExecution,
)
from spin_dynamics.sequences import read_pulseq

sequence = read_pulseq("protocol.seq")
x = np.linspace(-10e-3, 10e-3, 65)
study = Experiment(
    sequence=SequenceIRExecution(
        ir=sequence, system_frequency_hz=42.58e6, seed=17,
    ),
    sample=Sample(
        t1_seconds=1.0,
        t2_seconds=0.1,
        diffusion_coefficient=1.5e-9,
        sequence_domain=SequenceDomain(
            axes=(x,), density=np.ones(x.size), gradient_channels=("x",),
        ),
    ),
)
print(study.plan().report())
record = study.run()
record.save("protocol_run.npz")
```

The result contains clean and noisy complex ADC samples, compiled receiver
offset/phase metadata, step timing, and the final weighted ensemble. ADC
frequency and phase offsets, including per-sample Pulseq 1.5 phase shapes, are
applied during nominal receiver demodulation. Seeded diffusion, initialization,
and white acquisition noise reproduce through saved archives.

The current target deliberately fails closed when
`HardwareEffectsPolicy(transmit="apply")` or `receive="apply"` is requested:
probe transfer functions need a future probe-aware compiler target. It also
rejects probe-noise models; white receive noise is supported. Pulseq extensions
are not executed. Embedded IR is supported in friendly JSON configs; use Python
or JSON rather than TOML for nested block/event definitions.

## Timeline Visualizer

`plot_sequence` presents five aligned lanes on one time axis: RF in-phase and
quadrature envelopes, the physical Gx/Gy/Gz gradient channels, and ADC sample
ticks. Pulseq or native block boundaries can be drawn across every lane, making
overlap, unexpected gaps, phase changes, and acquisition alignment visible.

The package-level function and the `.plot()` methods are equivalent:

```python
from spin_dynamics.sequences import plot_sequence, read_pulseq

sequence = read_pulseq("experiment.seq")
figure, axes = plot_sequence(sequence, time_unit="ms")
figure.savefig("experiment_timeline.png", dpi=150)
```

The example script follows the other plotting examples and works with either a
file or a built-in spin echo:

```powershell
python examples\plot_sequence_timeline.py --output results\demo_sequence.png
python examples\plot_sequence_timeline.py experiment.seq --output results\experiment.png
python examples\experiment_sequence_ir.py experiment.seq `
  --system-frequency-hz 42580000 `
  --timeline-output results\experiment.png `
  --output results\experiment_run.npz
```

## Pulseq Coverage

The importer supports Pulseq 1.4.x and 1.5.x text files with:

- explicit block durations;
- default-raster RF events;
- default and half-raster arbitrary gradients;
- trapezoidal gradients;
- ADC events, including Pulseq 1.5 per-sample phase shapes;
- compressed and uncompressed shapes;
- absolute and PPM frequency/phase offsets.

`serialize_pulseq` and `write_pulseq` export core Pulseq 1.5.0 files using
uncompressed standard shapes. Export preserves RF effective centers, gradient
boundary amplitudes, RF/ADC offsets, ADC phase modulation, and declared
rasters. Timing is checked exactly: off-raster blocks, delays, or dwells raise
`PulseqFormatError` rather than being rounded silently. Optional extensions are
not emitted yet. Files are MD5-signed by default following the Pulseq signature
convention; pass `create_signature=False` when unsigned text is specifically
needed.

Current intentional limitations:

- explicit RF or gradient time-shape IDs are rejected;
- optional extensions are retained as metadata but not executed;
- files declaring required extensions are rejected;
- Pulseq 1.5.1 rotation and RF-shimming extensions are not implemented;
- imported signatures are parsed as inert sections and are not yet verified;
- only the moving-isochromat backend adapter is implemented.

The IR is additive. Existing MATLAB-validated workflows remain the scientific
reference while adapters are added incrementally and checked for result parity.
