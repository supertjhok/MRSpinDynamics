from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spin_dynamics.sequences import (
    ADCEvent,
    GradientWaveform,
    HardwareEffectsPolicy,
    PulseqFormatError,
    RFPulse,
    SequenceBlock,
    SequenceIR,
    compile_sequence,
    compiled_to_motion_steps,
    parse_pulseq,
    plot_sequence,
    read_pulseq,
    serialize_pulseq,
    sequence_plot_data,
    write_pulseq,
)


PULSEQ_15_FID = """
[VERSION]
major 1
minor 5
revision 1

[DEFINITIONS]
AdcRasterTime 1e-7
BlockDurationRaster 1e-5
GradientRasterTime 1e-5
RadiofrequencyRasterTime 1e-6
Name compact_fid

[BLOCKS]
1 10 1 0 0 0 0 0
2 10 0 0 0 0 1 0

[RF]
1 1000 1 2 0 2 0 0 0 0 0 e

[ADC]
1 4 20000 0 0 0 0 0 0

[SHAPES]
shape_id 1
num_samples 4
1
1
1
1

shape_id 2
num_samples 4
0
0
0
0
"""


class SequenceIRTests(unittest.TestCase):
    def test_hardware_effects_policy_is_explicit_and_independent(self):
        sequence = SequenceIR(
            blocks=(SequenceBlock(duration_seconds=20e-6),),
            hardware_effects=HardwareEffectsPolicy(
                transmit="apply", receive="ignore"
            ),
        )
        compiled = compile_sequence(sequence)
        self.assertEqual(compiled.hardware_effects, sequence.hardware_effects)
        with self.assertRaisesRegex(NotImplementedError, "transmit"):
            compiled_to_motion_steps(compiled)

    def test_hardware_effects_default_to_ideal_and_validate_modes(self):
        sequence = SequenceIR(blocks=(SequenceBlock(duration_seconds=20e-6),))
        self.assertEqual(sequence.hardware_effects, HardwareEffectsPolicy())
        with self.assertRaisesRegex(ValueError, "receive"):
            HardwareEffectsPolicy(receive="sometimes")
        with self.assertRaisesRegex(TypeError, "HardwareEffectsPolicy"):
            SequenceIR(
                blocks=(SequenceBlock(duration_seconds=20e-6),),
                hardware_effects={"transmit": "apply"},  # type: ignore[arg-type]
            )

    def test_compile_preserves_concurrent_rf_gradient_and_adc_timing(self):
        sequence = SequenceIR(
            blocks=(
                SequenceBlock(
                    duration_seconds=4e-3,
                    rf=RFPulse([100.0, 200.0], dwell_seconds=1e-3),
                    gradients=(
                        GradientWaveform([10.0, 20.0], dwell_seconds=1e-3),
                        None,
                        None,
                    ),
                    adc=ADCEvent(2, dwell_seconds=1e-3, delay_seconds=2e-3),
                ),
            )
        )

        compiled = compile_sequence(sequence)

        np.testing.assert_allclose(compiled.adc.times_seconds, [2.5e-3, 3.5e-3])
        self.assertAlmostEqual(compiled.duration_seconds, 4e-3)
        self.assertTrue(np.any(compiled.rf_hz == 200.0))
        self.assertTrue(np.any(compiled.gradients_hz_per_m[:, 0] == 20.0))

    def test_motion_adapter_converts_cycles_to_angular_units(self):
        sequence = SequenceIR(
            blocks=(
                SequenceBlock(
                    duration_seconds=1e-3,
                    rf=RFPulse([100.0], dwell_seconds=1e-3),
                    gradients=(
                        GradientWaveform([3.0], dwell_seconds=1e-3),
                        None,
                        None,
                    ),
                ),
            )
        )

        steps = compiled_to_motion_steps(
            compile_sequence(sequence), spatial_dimensions=2
        )

        self.assertEqual(len(steps), 1)
        self.assertAlmostEqual(steps[0].rf_amplitude, 2.0 * np.pi * 100.0)
        np.testing.assert_allclose(steps[0].gradient, [2.0 * np.pi * 3.0, 0.0])

    def test_block_rejects_event_that_runs_past_end(self):
        with self.assertRaisesRegex(ValueError, "extends beyond"):
            SequenceBlock(
                duration_seconds=1e-3,
                rf=RFPulse([1.0, 1.0], dwell_seconds=1e-3),
            )

    def test_ppm_offsets_require_system_frequency(self):
        sequence = SequenceIR(
            blocks=(
                SequenceBlock(
                    duration_seconds=1e-3,
                    rf=RFPulse(
                        [1.0],
                        dwell_seconds=1e-3,
                        frequency_offset_ppm=2.0,
                    ),
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "system_frequency_hz"):
            compile_sequence(sequence)
        compiled = compile_sequence(sequence, system_frequency_hz=10e6)
        expected_phase = 2.0 * np.pi * 20.0 * 0.5e-3
        self.assertAlmostEqual(np.angle(compiled.rf_hz[0]), expected_phase)

    def test_plot_data_uses_shared_time_axis_and_block_metadata(self):
        sequence = SequenceIR(
            blocks=(
                SequenceBlock(
                    duration_seconds=1e-3,
                    rf=RFPulse([1.0j], dwell_seconds=1e-3),
                    label="pulse",
                ),
                SequenceBlock(
                    duration_seconds=1e-3,
                    adc=ADCEvent(1, dwell_seconds=1e-3),
                    label="acquire",
                ),
            )
        )

        data = sequence_plot_data(sequence)

        self.assertEqual(data.time_unit, "ms")
        np.testing.assert_allclose(data.interval_edges, [0.0, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(data.rf_q_hz[:1], [1.0])
        np.testing.assert_allclose(data.adc_times, [1.5])
        np.testing.assert_allclose(data.block_boundaries, [0.0, 1.0, 2.0])
        self.assertEqual(data.block_labels, ("pulse", "acquire"))

    def test_plot_sequence_returns_five_aligned_lanes(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Matplotlib is optional")
        matplotlib.use("Agg")
        sequence = SequenceIR(
            blocks=(
                SequenceBlock(
                    duration_seconds=1e-3,
                    rf=RFPulse([100.0], dwell_seconds=1e-3),
                ),
            )
        )

        figure, axes = plot_sequence(sequence)

        self.assertEqual(len(axes), 5)
        self.assertEqual(axes[-1].get_xlabel(), "Time (ms)")
        import matplotlib.pyplot as plt

        plt.close(figure)


class PulseqImportTests(unittest.TestCase):
    def test_imports_pulseq_15_rf_adc_and_compiles(self):
        sequence = parse_pulseq(PULSEQ_15_FID)

        self.assertEqual(sequence.source_version, (1, 5, 1))
        self.assertEqual(sequence.blocks[0].rf.use, "excitation")
        np.testing.assert_allclose(sequence.blocks[0].rf.samples_hz, 1000.0)
        self.assertEqual(sequence.blocks[1].adc.num_samples, 4)

        compiled = compile_sequence(sequence)
        np.testing.assert_allclose(
            compiled.adc.times_seconds,
            1e-4 + np.asarray([10.0, 30.0, 50.0, 70.0]) * 1e-6,
        )

    def test_imports_legacy_pulseq_14_core_records(self):
        text = (
            PULSEQ_15_FID.replace("minor 5", "minor 4")
            .replace("revision 1", "revision 0")
            .replace("1 1000 1 2 0 2 0 0 0 0 0 e", "1 1000 1 2 0 0 0 0")
            .replace("1 4 20000 0 0 0 0 0 0", "1 4 20000 0 0 0")
        )

        sequence = parse_pulseq(text)

        self.assertEqual(sequence.source_version, (1, 4, 0))
        self.assertEqual(sequence.blocks[1].adc.num_samples, 4)

    def test_decompresses_run_length_encoded_shape(self):
        text = PULSEQ_15_FID.replace(
            "num_samples 4\n1\n1\n1\n1",
            "num_samples 6\n1\n0\n0\n3",
            1,
        ).replace(
            "num_samples 4\n0\n0\n0\n0",
            "num_samples 6\n0\n0\n4",
            1,
        )
        sequence = parse_pulseq(text)
        np.testing.assert_allclose(sequence.blocks[0].rf.samples_hz, 1000.0)

    def test_imports_trapezoid_gradient(self):
        text = PULSEQ_15_FID.replace(
            "1 10 1 0 0 0 0 0", "1 10 1 1 0 0 0 0"
        ).replace(
            "[ADC]", "[TRAP]\n1 1000 10 60 10 0\n\n[ADC]"
        )
        sequence = parse_pulseq(text)
        gradient = sequence.blocks[0].gradients[0]
        self.assertIsNotNone(gradient)
        self.assertEqual(gradient.samples_hz_per_m.size, 8)
        self.assertAlmostEqual(np.max(gradient.samples_hz_per_m), 1000.0)

    def test_rejects_required_extensions(self):
        text = PULSEQ_15_FID.replace(
            "Name compact_fid", "Name compact_fid\nRequiredExtensions ROTATIONS"
        )
        with self.assertRaisesRegex(PulseqFormatError, "ROTATIONS"):
            parse_pulseq(text)


class PulseqExportTests(unittest.TestCase):
    def test_round_trip_preserves_core_events_and_metadata(self):
        sequence = SequenceIR(
            blocks=(
                SequenceBlock(
                    duration_seconds=40e-6,
                    rf=RFPulse(
                        [100.0, 100.0j, -50.0],
                        dwell_seconds=1e-6,
                        delay_seconds=1e-6,
                        frequency_offset_hz=250.0,
                        phase_offset_rad=0.2,
                        use="excitation",
                        center_seconds=1.25e-6,
                    ),
                    gradients=(
                        GradientWaveform(
                            [1000.0, -500.0, 250.0, 100.0],
                            dwell_seconds=10e-6,
                            first_hz_per_m=0.0,
                            last_hz_per_m=20.0,
                        ),
                        None,
                        None,
                    ),
                    adc=ADCEvent(
                        2,
                        dwell_seconds=1e-6,
                        delay_seconds=4e-6,
                        phase_offsets_rad=[0.0, np.pi / 2.0],
                    ),
                ),
                SequenceBlock(
                    duration_seconds=20e-6,
                    gradients=(
                        GradientWaveform(
                            [100.0, 0.0],
                            dwell_seconds=10e-6,
                            first_hz_per_m=20.0,
                            last_hz_per_m=0.0,
                        ),
                        None,
                        None,
                    ),
                ),
            ),
            definitions={"Name": "round_trip"},
        )

        restored = parse_pulseq(serialize_pulseq(sequence))

        np.testing.assert_allclose(
            restored.blocks[0].rf.samples_hz, sequence.blocks[0].rf.samples_hz
        )
        self.assertAlmostEqual(restored.blocks[0].rf.center_seconds, 1.25e-6)
        self.assertEqual(restored.blocks[0].rf.use, "excitation")
        gradient = restored.blocks[0].gradients[0]
        np.testing.assert_allclose(
            gradient.samples_hz_per_m, [1000.0, -500.0, 250.0, 100.0]
        )
        self.assertAlmostEqual(gradient.first_hz_per_m, 0.0)
        self.assertAlmostEqual(gradient.last_hz_per_m, 20.0)
        self.assertAlmostEqual(restored.blocks[1].gradients[0].first_hz_per_m, 20.0)
        np.testing.assert_allclose(
            restored.blocks[0].adc.phase_offsets_rad, [0.0, np.pi / 2.0]
        )

    def test_export_signature_covers_sequence_text(self):
        sequence = SequenceIR(blocks=(SequenceBlock(duration_seconds=20e-6),))
        text = serialize_pulseq(sequence)
        payload, signature = text.split("\n[SIGNATURE]\n")
        fields = dict(line.split(maxsplit=1) for line in signature.splitlines())

        self.assertEqual(fields["Type"], "md5")
        self.assertEqual(fields["Hash"], hashlib.md5(payload.encode()).hexdigest())

    def test_write_pulseq_creates_readable_file(self):
        sequence = SequenceIR(blocks=(SequenceBlock(duration_seconds=20e-6),))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "delay.seq"
            write_pulseq(sequence, path)
            restored = read_pulseq(path)
        self.assertAlmostEqual(restored.duration_seconds, 20e-6)

    def test_export_rejects_off_raster_timing(self):
        sequence = SequenceIR(blocks=(SequenceBlock(duration_seconds=15e-6),))
        with self.assertRaisesRegex(PulseqFormatError, "block 1 duration"):
            serialize_pulseq(sequence)

    def test_export_rejects_disconnected_gradient_boundaries(self):
        sequence = SequenceIR(
            blocks=(
                SequenceBlock(
                    duration_seconds=20e-6,
                    gradients=(
                        GradientWaveform(
                            [1.0, 1.0],
                            dwell_seconds=10e-6,
                            last_hz_per_m=1.0,
                        ),
                        None,
                        None,
                    ),
                ),
            )
        )
        with self.assertRaisesRegex(PulseqFormatError, "end with zero"):
            serialize_pulseq(sequence)


if __name__ == "__main__":
    unittest.main()
