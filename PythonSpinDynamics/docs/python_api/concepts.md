# Concepts and Units

The Python package preserves the MATLAB model:

- spins are represented as uncoupled spin-1/2 magnetization vectors;
- the sample is represented by an offset grid of isochromats;
- RF and free-precession intervals are applied as rotations in coherence space;
- acquired spectra can be converted to time-domain echoes or FID traces.

## Coherence Ordering

Low-level kernels use MATLAB's coherence ordering:

```text
M0, M-, M+
```

This ordering is documented in the relevant Python dataclasses and kernel
docstrings because it is easy to transpose accidentally when porting MATLAB.

## Complex Spectra and Probe Phase

Spectra and asymptotic magnetization values are complex. Probe circuit response
can rotate most of the signal into the imaginary component, especially in tuned
and untuned probe examples. For comparison plots, magnitude is often the least
misleading first view; real and imaginary components are still useful for
debugging phase conventions.

## Time Normalization

The core MATLAB routines often normalize time to the nominal `w1 = 1`
convention. In this convention a 90-degree pulse has length `pi / 2`, and a
180-degree pulse has length `pi`.

Parameter constructors that expose physical seconds convert to normalized units
inside workflow helpers. Current examples:

- `calc_masy_ideal` converts CPMG pulse timings using `T_90`.
- `calc_macq_fid` converts ideal FID segment timings and relaxation constants.

Keep new Python function names explicit when a value is not dimensionless.
