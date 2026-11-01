"""Time-domain echo utilities.

MATLAB reference:
    MATLABSpinDynamics/Version_3/code/calc_echo/calc_time_domain_echo.m
"""

from __future__ import annotations

import numpy as np


def calc_time_domain_echo(
    spect: np.ndarray,
    wvect: np.ndarray,
    *,
    zero_fill: int = 4,
    registration: str = "matlab",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an offset-domain spectrum into a time-domain echo.

    This mirrors the numerical path in MATLAB `calc_time_domain_echo.m` with
    plotting removed. `spect` and `wvect` are treated as one-dimensional arrays.

    ``registration="matlab"`` (default, Octave parity) keeps MATLAB's layout:
    the last spectral sample is dropped for odd lengths, ``omega = 0`` lands
    ``zero_fill * (n mod 2) / 2`` bins off the DFT centre (2 bins for odd
    ``n`` at ``zero_fill = 4``), which multiplies the echo by a linear phase
    ramp, and ``tvect`` spans ``±pi / max(wvect)`` on a grid that is not the
    DFT time grid (about ``1/n`` stretched). ``registration="exact"`` assumes
    a uniform ``wvect``, places every sample at its own frequency bin, and
    returns the true DFT time grid ``2*pi*k / (zero_fill * n * dw)`` centred
    on ``t = 0``, so ``echo(t) = sum(spect * exp(1j * wvect * t))`` exactly at
    each returned time (scaled by ``zero_fill / (zero_fill * n)`` as in MATLAB).
    """

    if registration == "exact":
        return _calc_time_domain_echo_exact(spect, wvect, zero_fill)
    if registration != "matlab":
        raise ValueError("registration must be 'matlab' or 'exact'")

    spect = np.asarray(spect, dtype=np.complex128).reshape(-1)
    wvect = np.asarray(wvect, dtype=np.float64).reshape(-1)
    if spect.size != wvect.size:
        raise ValueError("spect and wvect must have the same length")
    if spect.size == 0:
        raise ValueError("spect and wvect must be non-empty")
    if zero_fill <= 0:
        raise ValueError("zero_fill must be positive")

    half = spect.size // 2
    ts = np.pi / (zero_fill * np.max(wvect))

    spect_zf = np.zeros(zero_fill * spect.size, dtype=np.complex128)
    start = (zero_fill - 1) * half
    stop = (zero_fill + 1) * half
    spect_zf[start:stop] = spect[: 2 * half]

    tvect = ts * np.linspace(-zero_fill * half, zero_fill * half, zero_fill * spect.size)
    echo = zero_fill * np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(spect_zf)))
    return echo, tvect


def _calc_time_domain_echo_exact(
    spect: np.ndarray,
    wvect: np.ndarray,
    zero_fill: int,
) -> tuple[np.ndarray, np.ndarray]:
    spect = np.asarray(spect, dtype=np.complex128).reshape(-1)
    wvect = np.asarray(wvect, dtype=np.float64).reshape(-1)
    if spect.size != wvect.size:
        raise ValueError("spect and wvect must have the same length")
    if spect.size < 2:
        raise ValueError("registration='exact' needs at least two offsets")
    if zero_fill <= 0:
        raise ValueError("zero_fill must be positive")
    dw = float(np.mean(np.diff(wvect)))
    if dw <= 0 or not np.allclose(np.diff(wvect), dw, rtol=1e-9, atol=0.0):
        raise ValueError("registration='exact' needs a uniform increasing wvect")

    n = spect.size
    total = int(zero_fill) * n
    centre = n // 2
    # Frequencies are (k - centre) * dw + residual; the residual (non-zero for
    # grids not containing omega = 0) is restored as a time-domain phase.
    residual = float(wvect[centre])
    spect_zf = np.zeros(total, dtype=np.complex128)
    start = total // 2 - centre
    spect_zf[start : start + n] = spect
    ts = 2.0 * np.pi / (total * dw)
    tvect = ts * (np.arange(total, dtype=np.float64) - total // 2)
    echo = zero_fill * np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(spect_zf)))
    echo = echo * np.exp(1j * residual * tvect)
    return echo, tvect


def calc_time_domain_echo_arb(
    mrx: np.ndarray,
    wvect: np.ndarray,
    tacq: float,
    tdw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a time-domain echo by direct summation.

    Mirrors MATLAB `calc_echo/calc_time_domain_echo_arb.m`, with plotting
    removed.
    """

    mrx = np.asarray(mrx, dtype=np.complex128).reshape(-1)
    wvect = np.asarray(wvect, dtype=np.float64).reshape(-1)
    if mrx.size != wvect.size:
        raise ValueError("mrx and wvect must have the same length")
    if mrx.size == 0:
        raise ValueError("mrx and wvect must be non-empty")
    if tacq <= 0:
        raise ValueError("tacq must be positive")
    if tdw <= 0:
        raise ValueError("tdw must be positive")

    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    echo = np.sum(mrx[:, np.newaxis] * np.exp(1j * wvect[:, np.newaxis] * tvect), axis=0)
    return echo, tvect


def calc_fid_time_domain(
    mrx: np.ndarray,
    wvect: np.ndarray,
    tacq: float,
    tdw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a time-domain FID by direct summation.

    Mirrors MATLAB `calc_FID_decay/calc_FID_time_domain.m`, with plotting
    removed. Unlike `calc_time_domain_echo_arb`, the acquisition window starts
    at zero and runs to `tacq`.
    """

    mrx = np.asarray(mrx, dtype=np.complex128).reshape(-1)
    wvect = np.asarray(wvect, dtype=np.float64).reshape(-1)
    if mrx.size != wvect.size:
        raise ValueError("mrx and wvect must have the same length")
    if mrx.size == 0:
        raise ValueError("mrx and wvect must be non-empty")
    if tacq <= 0:
        raise ValueError("tacq must be positive")
    if tdw <= 0:
        raise ValueError("tdw must be positive")

    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(0, tacq, nacq)
    echo = np.sum(mrx[:, np.newaxis] * np.exp(1j * wvect[:, np.newaxis] * tvect), axis=0)
    return echo, tvect
