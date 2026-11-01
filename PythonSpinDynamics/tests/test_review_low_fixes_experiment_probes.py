"""Regression tests for code-review items L-28 to L-35.

* L-28: the config TOML writer did not escape control characters or quote keys.
* L-29: ``frequency_domain_canceller`` crashed for ``segment_length=1`` and
  silently left samples uncancelled when ``hop >= segment_length``.
* L-30: ``spectral_phase_shift`` at the Nyquist bin (documented, not a bug).
* L-31: NaN/inf arrays were written as non-standard JSON while scalars were
  rejected, so provenance fingerprinting raised after the workflow ran.
* L-32: tuned/untuned receiver noise omitted the damping resistor's 4kT/Rd.
* L-33: the probe parameter helpers had drifted between modules.
* L-34: SQUID / ideal-Faraday field-noise PSDs used boolean-mask indexing that
  ``jax.jit`` cannot trace.
* L-35: the coil-field solve cache was unbounded.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.detection import IdealFaradayCoil, SQUIDMagnetometer
from spin_dynamics.experiment import CPMG, Acquisition, Experiment, Phantom, Sample
from spin_dynamics.experiment import wiring
from spin_dynamics.experiment.config import (
    dumps_toml,
    experiment_from_config,
    experiment_to_config,
)
from spin_dynamics.experiment.provenance import experiment_fingerprint
from spin_dynamics.experiment.serialization import decode, encode
from spin_dynamics.interference._signal import spectral_phase_shift
from spin_dynamics.interference.cancellers import frequency_domain_canceller
from spin_dynamics.parameters import set_params_tuned_orig
from spin_dynamics.probes import _common
from spin_dynamics.probes.tuned import tuned_probe_rx
from spin_dynamics.probes.untuned import untuned_probe_rx

K_B = 1.380649e-23


# --- L-28 --------------------------------------------------------------------
def test_toml_writer_escapes_strings_and_quotes_keys() -> None:
    label = 'tab\tnew\nline "quoted" back\\slash bell\x07 del\x7f'
    noise = {"sigma value": 0.1, "a.b": 2, "x=y": "z", "plain_key": 1}
    experiment = Experiment(
        sequence=CPMG(),
        sample=Sample(label=label),
        acquisition=Acquisition(noise=noise),
    )
    text = dumps_toml(experiment_to_config(experiment))
    parsed = tomllib.loads(text)  # independent reference parser
    assert parsed["sample"]["label"] == label
    assert parsed["acquisition"]["noise"] == noise
    rebuilt = experiment_from_config(parsed)
    assert experiment_fingerprint(rebuilt) == experiment_fingerprint(experiment)


# --- L-29 --------------------------------------------------------------------
def _gain_record(n=512, gain=2.0 - 0.5j, seed=3):
    rng = np.random.default_rng(seed)
    references = rng.normal(size=(1, n))
    return gain * references[0], references


def test_frequency_canceller_single_sample_segments_cancel_a_pure_gain() -> None:
    primary, references = _gain_record()
    result = frequency_domain_canceller(primary, references, segment_length=1)
    # Independent reference: a frequency-flat gain is removed exactly.
    np.testing.assert_allclose(result.cleaned, 0.0, atol=1e-12)


def test_frequency_canceller_rejects_hops_that_leave_gaps() -> None:
    primary, references = _gain_record()
    for hop in (64, 128):
        with pytest.raises(ValueError, match="hop must be smaller"):
            frequency_domain_canceller(primary, references, segment_length=64, hop=hop)
    # An overlapping hop still covers every sample after the first.
    result = frequency_domain_canceller(primary, references, segment_length=64, hop=48)
    np.testing.assert_allclose(result.cleaned[1:], 0.0, atol=1e-10)


# --- L-30 --------------------------------------------------------------------
@pytest.mark.parametrize("phase", [0.3, np.pi / 2, 2.0])
def test_spectral_phase_shift_matches_sampled_closed_form(phase: float) -> None:
    n = np.arange(64)
    for omega in (2 * np.pi * 5 / 64, np.pi):  # mid-band tone and Nyquist
        shifted = spectral_phase_shift(np.cos(omega * n), phase)
        np.testing.assert_allclose(shifted, np.cos(omega * n - phase), atol=1e-12)


# --- L-31 --------------------------------------------------------------------
def _strict_loads(text: str):
    def reject(token: str):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(text, parse_constant=reject)


def test_non_finite_values_encode_as_standard_json_and_round_trip() -> None:
    rho = np.ones((3, 3))
    t1_map = np.full((3, 3), 1.0)
    t1_map[0, 0] = np.nan
    t1_map[1, 2] = np.inf
    t1_map[2, 1] = -np.inf
    experiment = Experiment(
        sequence=CPMG(),
        sample=Sample(phantom=Phantom(rho=rho, t1_map=t1_map)),
        acquisition=Acquisition(noise={"sigma": float("nan")}),
    )
    payload = experiment.to_json()
    _strict_loads(payload)  # standard JSON: no NaN/Infinity tokens
    experiment_fingerprint(experiment)  # previously raised ValueError
    back = Experiment.from_json(payload)
    np.testing.assert_array_equal(back.sample.phantom.t1_map, t1_map)
    assert np.isnan(back.acquisition.noise["sigma"])

    complex_values = np.array([1.0 + 1.0j, complex(np.nan, -np.inf)])
    decoded = decode(json.loads(json.dumps(encode(complex_values), allow_nan=False)))
    np.testing.assert_array_equal(decoded, complex_values)


# --- L-32 --------------------------------------------------------------------
def _snr_with_noise_psd(mrx, pnoise, del_w, f):
    mf = np.conj(mrx)
    mf = mf / np.sqrt(trapezoid(np.abs(mf) ** 2, del_w))
    vsig = trapezoid(mrx * mf, del_w)
    return float(np.real(vsig) / np.sqrt(trapezoid(pnoise * np.abs(mf) ** 2, f)))


def _common_sp(**extra):
    w0 = 2 * np.pi * 1.0e6
    base = {
        "k": K_B,
        "T": 300.0,
        "L": 1.0e-6,
        "R": 0.2,
        "Cin": 5e-12,
        "Rin": 1e30,  # amplifier input treated as noiseless (vn = in = 0)
        "Rd": 1.0e3,
        "vn": 0.0,
        "in": 0.0,
        "mf_type": 1,
        "w0": w0,
        "del_w": np.linspace(-5.0, 5.0, 41),
    }
    base.update(extra)
    return base


def test_tuned_receiver_noise_obeys_nyquist_theorem() -> None:
    # Independent reference: with only thermal resistors (R and Rd) at T, the
    # node voltage noise is 4kT Re(Z_node) (Nyquist/Bode theorem).
    sp = _common_sp(C=1.0 / ((2 * np.pi * 1.0e6) ** 2 * 1.0e-6))
    pp = {"T_90": 25e-6}
    del_w = sp["del_w"]
    mrx, snr, _ = tuned_probe_rx(sp, pp, np.ones(del_w.size))
    s = 1j * (sp["w0"] + del_w * (np.pi / 2) / pp["T_90"])
    y_node = (
        s * sp["C"] + 1 / sp["Rd"] + 1 / (s * sp["L"] + sp["R"]) + s * sp["Cin"] + 1 / sp["Rin"]
    )
    pnoise = 4 * K_B * sp["T"] * np.real(1 / y_node)
    expected = _snr_with_noise_psd(mrx, pnoise, del_w, np.imag(s) / (2 * np.pi))
    assert snr == pytest.approx(expected, rel=1e-9)


def test_untuned_receiver_noise_obeys_nyquist_theorem() -> None:
    # Ideal-transformer limit where the untuned model is an exact passive
    # network: no tank capacitance and no transformer winding loss.
    sp = _common_sp(C=0.0, Rdup=0.0, Nrx=3.0, krx=1.0, L1=2e-6, R1=0.0)
    pp = {"T_90": 25e-6}
    del_w = sp["del_w"]
    mrx, snr, _ = untuned_probe_rx(sp, pp, np.ones(del_w.size))
    s = 1j * (sp["w0"] + del_w * (np.pi / 2) / pp["T_90"])
    nv = sp["krx"] * sp["Nrx"] * sp["L1"] / (sp["L"] + sp["L1"])
    z_source = (s * sp["L"] + sp["R"]) * nv**2
    y_node = 1 / z_source + s * sp["Cin"] + 1 / sp["Rin"] + 1 / sp["Rd"]
    pnoise = 4 * K_B * sp["T"] * np.real(1 / y_node)
    expected = _snr_with_noise_psd(mrx, pnoise, del_w, np.imag(s) / (2 * np.pi))
    assert snr == pytest.approx(expected, rel=1e-9)


# --- L-33 --------------------------------------------------------------------
def test_probe_parameter_helpers_share_one_contract() -> None:
    _params, _sp, pp = set_params_tuned_orig(numpts=5)
    updated = _common.with_fields(pp, tref=np.array([1.0]))
    assert type(updated) is type(pp)  # dataclass fields -> dataclasses.replace
    np.testing.assert_array_equal(updated.tref, [1.0])
    added = _common.with_fields(pp, tp=np.array([2.0]))  # not a field -> dict
    assert isinstance(added, dict) and added["N"] == pp.N
    namespace = _common.with_fields(SimpleNamespace(a=1), b=2)
    assert namespace == {"a": 1, "b": 2}
    assert _common.with_fields({"a": 1}, a=3) == {"a": 3}
    assert _common.field({"in": 7.0}, "in_") == 7.0
    assert _common.field(SimpleNamespace(in_=8.0), "in_") == 8.0
    assert dataclasses.is_dataclass(pp)


# --- L-34 --------------------------------------------------------------------
@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
def test_detector_noise_psds_trace_under_jax_jit() -> None:
    import jax
    import jax.numpy as jnp

    freqs = np.array([0.0, 10.0, 100.0, 1.0e3])
    for detector in (
        SQUIDMagnetometer(one_over_f_knee_hz=5.0),
        IdealFaradayCoil(field_noise_T_per_rtHz_ref=1.0, f_ref_hz=1.0e3),
    ):
        expected = detector.field_noise_psd(freqs)
        jitted = jax.jit(lambda f, d=detector: d.field_noise_psd(f, xp=jnp))
        # jax defaults to float32; compare at single precision.
        actual = np.asarray(jitted(jnp.asarray(freqs)), dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
        # f = 0 is clamped to the smallest positive grid frequency.
        np.testing.assert_allclose(expected[0], expected[1])


# --- L-35 --------------------------------------------------------------------
def test_solve_cache_is_bounded_lru(monkeypatch) -> None:
    monkeypatch.setattr(wiring, "_SOLVE_CACHE", {})
    limit = wiring._SOLVE_CACHE_MAX_ENTRIES
    for index in range(limit):
        wiring._cache_put(f"k{index}", index)
    assert wiring._cache_get("k0") == 0  # refresh k0 as most recently used
    for index in range(limit, limit + 5):
        wiring._cache_put(f"k{index}", index)
    assert len(wiring._SOLVE_CACHE) == limit
    assert wiring._cache_get("k0") == 0
    assert wiring._cache_get("k1") is None  # least recently used were evicted
    assert wiring._cache_get(f"k{limit + 4}") == limit + 4
