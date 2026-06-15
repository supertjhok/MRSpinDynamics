"""CPMG imaging workflows."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import calc_rotation_matrix
from spin_dynamics.parameters import (
    set_params_ideal,
)
from spin_dynamics.probes.matched import matching_network_design2
from spin_dynamics.probes.tuned import tuned_probe_lp, tuned_probe_rx_tf
from spin_dynamics.workflows.acquisition import (
    calc_macq_ideal_probe_relax4,
    calc_macq_matched_probe_relax4,
    calc_macq_tuned_probe_relax4,
)
from spin_dynamics.workflows.cpmg import (
    _calc_matched_pulse_shape,
)


@dataclass(frozen=True)
class IdealCPMGImagingResult:
    """Ideal-probe CPMG imaging result."""

    rho: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray
    kspace: np.ndarray
    image: np.ndarray
    magnitude: np.ndarray
    gradx: np.ndarray
    gradz: np.ndarray
    del_w: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    probe: str


@dataclass(frozen=True)
class ProbeCPMGImagingResult:
    """Probe-aware CPMG imaging result."""

    rho: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray
    kspace: np.ndarray
    image: np.ndarray
    magnitude: np.ndarray
    gradx: np.ndarray
    gradz: np.ndarray
    del_w: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    probe: str


IdealPhaseEncodedCPMGImagingResult = IdealCPMGImagingResult
ProbePhaseEncodedCPMGImagingResult = ProbeCPMGImagingResult


@dataclass(frozen=True)
class ImagingFieldMaps:
    """Spatial sample and field maps for CPMG imaging workflows.

    `b0_map` contains normalized off-resonance offsets added to the generated
    isochromat offset axis. `b1_tx_map` and `b1_rx_map` are relative transmit
    and receive sensitivity maps. All maps are two-dimensional and share the
    same shape as `rho`.
    """

    rho: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray
    del_wx: np.ndarray
    del_wz: np.ndarray

    def kernel_maps(self, ny: int, maxoffs: float) -> dict[str, np.ndarray]:
        """Return flattened arrays consumed by the arbitrary-pulse kernels."""

        if ny <= 0:
            raise ValueError("ny must be positive")
        rho = self.rho
        reps = int(ny)
        del_w0y = np.linspace(-float(maxoffs), float(maxoffs), reps)
        b0 = self.b0_map.reshape(-1)
        return {
            "del_w": np.concatenate([offset + b0 for offset in del_w0y]),
            "del_wx": np.tile(self.del_wx.reshape(-1), reps),
            "del_wz": np.tile(self.del_wz.reshape(-1), reps),
            "w_1": np.tile(self.b1_tx_map.reshape(-1), reps),
            "w_1r": np.tile(self.b1_rx_map.reshape(-1), reps),
            "m0": np.tile(rho.reshape(-1), reps),
            "mth": np.tile(rho.reshape(-1), reps),
            "T1": np.tile(self.t1_map.reshape(-1), reps),
            "T2": np.tile(self.t2_map.reshape(-1), reps),
        }


def _as_map(value: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _default_gradient_maps(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nx, nz = shape
    del_wx = np.tile(np.linspace(-1, 1, nx)[:, np.newaxis], (1, nz))
    del_wz = np.tile(np.linspace(-1, 1, nz)[np.newaxis, :], (nx, 1))
    return del_wx, del_wz


def _default_b1_map(shape: tuple[int, int]) -> np.ndarray:
    nx, nz = shape
    x = np.arange(1, nx + 1, dtype=np.float64)
    z = np.arange(1, nz + 1, dtype=np.float64)
    zz, xx = np.meshgrid(z, x)
    sigma_x = max(float(nx), 1.0)
    sigma_z = max(float(nz), 1.0)
    rf = np.exp(
        -0.5
        * (
            ((xx - round(nx / 2)) / sigma_x) ** 2
            + ((zz - round(nz / 2)) / sigma_z) ** 2
        )
    )
    return rf / np.max(rf)


def make_imaging_field_maps(
    rho: Iterable[float] | np.ndarray,
    *,
    t1_map: Iterable[float] | np.ndarray | None = None,
    t2_map: Iterable[float] | np.ndarray | None = None,
    b0_map: Iterable[float] | np.ndarray | None = None,
    b1_tx_map: Iterable[float] | np.ndarray | None = None,
    b1_rx_map: Iterable[float] | np.ndarray | None = None,
    del_wx: Iterable[float] | np.ndarray | None = None,
    del_wz: Iterable[float] | np.ndarray | None = None,
) -> ImagingFieldMaps:
    """Validate and assemble spatial maps for CPMG imaging.

    `b0_map` is a normalized angular offset map. If omitted, zero additional
    off-resonance is used. If `b1_tx_map` is omitted, the same synthetic
    single-sided map used by the existing imaging examples is generated.
    `b1_rx_map` defaults to `b1_tx_map`.
    """

    rho_arr = _as_map(rho, "rho")
    shape = rho_arr.shape
    t1_arr = 5e-3 * np.ones_like(rho_arr) if t1_map is None else _as_map(t1_map, "t1_map")
    t2_arr = 5e-3 * np.ones_like(rho_arr) if t2_map is None else _as_map(t2_map, "t2_map")
    b0_arr = np.zeros_like(rho_arr) if b0_map is None else _as_map(b0_map, "b0_map")
    b1_tx_arr = (
        _default_b1_map(shape)
        if b1_tx_map is None
        else _as_map(b1_tx_map, "b1_tx_map")
    )
    b1_rx_arr = b1_tx_arr.copy() if b1_rx_map is None else _as_map(b1_rx_map, "b1_rx_map")
    del_wx_arr, del_wz_arr = _default_gradient_maps(shape)
    if del_wx is not None:
        del_wx_arr = _as_map(del_wx, "del_wx")
    if del_wz is not None:
        del_wz_arr = _as_map(del_wz, "del_wz")

    for name, arr in [
        ("t1_map", t1_arr),
        ("t2_map", t2_arr),
        ("b0_map", b0_arr),
        ("b1_tx_map", b1_tx_arr),
        ("b1_rx_map", b1_rx_arr),
        ("del_wx", del_wx_arr),
        ("del_wz", del_wz_arr),
    ]:
        if arr.shape != shape:
            raise ValueError(f"{name} must have the same shape as rho")
    if np.any(b1_tx_arr < 0) or np.any(b1_rx_arr < 0):
        raise ValueError("B1 maps must be non-negative")
    if np.any(t1_arr <= 0) or np.any(t2_arr <= 0):
        raise ValueError("t1_map and t2_map must be positive")

    return ImagingFieldMaps(
        rho=rho_arr,
        t1_map=t1_arr,
        t2_map=t2_arr,
        b0_map=b0_arr,
        b1_tx_map=b1_tx_arr,
        b1_rx_map=b1_rx_arr,
        del_wx=del_wx_arr,
        del_wz=del_wz_arr,
    )


def load_imaging_field_maps_npz(
    path: str | Path,
    *,
    rho_key: str = "rho",
    t1_key: str = "t1_map",
    t2_key: str = "t2_map",
    b0_key: str = "b0_map",
    b1_tx_key: str = "b1_tx_map",
    b1_rx_key: str = "b1_rx_map",
    del_wx_key: str = "del_wx",
    del_wz_key: str = "del_wz",
) -> ImagingFieldMaps:
    """Load imaging field maps from a NumPy `.npz` archive."""

    with np.load(Path(path)) as archive:

        def optional(key: str) -> np.ndarray | None:
            return archive[key] if key in archive.files else None

        if rho_key not in archive.files:
            raise ValueError(f"NPZ archive must contain '{rho_key}'")
        return make_imaging_field_maps(
            archive[rho_key],
            t1_map=optional(t1_key),
            t2_map=optional(t2_key),
            b0_map=optional(b0_key),
            b1_tx_map=optional(b1_tx_key),
            b1_rx_map=optional(b1_rx_key),
            del_wx=optional(del_wx_key),
            del_wz=optional(del_wz_key),
        )


def _field_maps(
    rho: np.ndarray,
    t1_map: np.ndarray,
    t2_map: np.ndarray,
    ny: int,
    maxoffs: float,
) -> dict[str, np.ndarray]:
    return make_imaging_field_maps(
        rho,
        t1_map=t1_map,
        t2_map=t2_map,
    ).kernel_maps(ny, maxoffs)


def _coerce_field_maps(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    t1_map: Iterable[float] | np.ndarray | None,
    t2_map: Iterable[float] | np.ndarray | None,
    field_maps: ImagingFieldMaps | None,
) -> ImagingFieldMaps:
    if isinstance(rho, ImagingFieldMaps):
        if field_maps is not None or t1_map is not None or t2_map is not None:
            raise ValueError(
                "do not provide t1_map, t2_map, or field_maps when rho is ImagingFieldMaps"
            )
        return rho
    if field_maps is not None:
        rho_arr = _as_map(rho, "rho")
        if t1_map is not None or t2_map is not None:
            raise ValueError("t1_map and t2_map are supplied by field_maps")
        if rho_arr.shape != field_maps.rho.shape:
            raise ValueError("rho must have the same shape as field_maps.rho")
        if not np.allclose(rho_arr, field_maps.rho):
            raise ValueError("rho must match field_maps.rho")
        return field_maps
    return make_imaging_field_maps(rho, t1_map=t1_map, t2_map=t2_map)


def _field_maps_from_container(
    maps: ImagingFieldMaps,
    ny: int,
    maxoffs: float,
) -> dict[str, np.ndarray]:
    return maps.kernel_maps(ny, maxoffs)


def reconstruct_image_from_kspace(kspace: np.ndarray, echo_index: int = 0) -> np.ndarray:
    """Reconstruct an image from one echo of CPMG imaging k-space."""

    kspace = np.asarray(kspace, dtype=np.complex128)
    if kspace.ndim != 3:
        raise ValueError("kspace must have shape (px, pz, num_echoes)")
    return np.fft.ifftshift(np.fft.ifft2(kspace[:, :, int(echo_index)]))


def _validate_imaging_inputs(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    t1_map: Iterable[float] | np.ndarray | None,
    t2_map: Iterable[float] | np.ndarray | None,
    num_echoes: int,
    ny: int,
    fov: tuple[float, float] | Iterable[float],
) -> tuple[ImagingFieldMaps, np.ndarray]:
    if num_echoes <= 0 or ny <= 0:
        raise ValueError("num_echoes and ny must be positive")
    fov_arr = np.asarray(tuple(fov), dtype=np.float64).reshape(-1)
    if fov_arr.size != 2 or np.any(fov_arr <= 0):
        raise ValueError("fov must contain two positive values")
    return _coerce_field_maps(rho, t1_map, t2_map, None), fov_arr


def _finish_imaging_result(
    result_type,
    field_maps: ImagingFieldMaps,
    kspace: np.ndarray,
    gradx: np.ndarray,
    gradz: np.ndarray,
    del_w: np.ndarray,
    sequence_time: np.ndarray,
    probe: str,
):
    image = np.stack(
        [reconstruct_image_from_kspace(kspace, echo_index=idx) for idx in range(kspace.shape[2])],
        axis=2,
    )
    return result_type(
        rho=field_maps.rho,
        t1_map=field_maps.t1_map,
        t2_map=field_maps.t2_map,
        b0_map=field_maps.b0_map,
        b1_tx_map=field_maps.b1_tx_map,
        b1_rx_map=field_maps.b1_rx_map,
        kspace=kspace,
        image=image,
        magnitude=np.abs(image),
        gradx=gradx,
        gradz=gradz,
        del_w=del_w,
        echo_integrals=kspace,
        sequence_time=sequence_time,
        probe=probe,
    )


def _set_params_tuned_jmr() -> tuple[SimpleNamespace, SimpleNamespace]:
    gamma = 42.577e6 * 2 * np.pi
    f0 = 0.5e6
    fin = 0.5e6
    w0 = 2 * np.pi * fin
    L = 10e-6
    Q = 50.0
    T_90 = 25e-6
    T_180 = 2 * T_90
    Vs = 1.0
    sens = ((np.pi / 2) / T_90) * (2 * w0 * L) / (gamma * Vs)
    sp = SimpleNamespace(
        k=1.381e-23,
        T=300.0,
        gamma=gamma,
        f0=f0,
        fin=fin,
        w0=w0,
        L=L,
        Q=Q,
        R=2 * np.pi * f0 * L / Q,
        C=1 / ((2 * np.pi * f0) ** 2 * L),
        Rs=2.0,
        Vs=Vs,
        Rin=1e6,
        Cin=5e-12,
        Rd=1e6,
        NF=1.0,
        vn=0.5e-9,
        in_=2e-15,
        m0=1.0,
        mth=1.0,
        numpts=10_000,
        maxoffs=10.0,
        del_w=np.linspace(-10.0, 10.0, 10_000),
        sens=sens,
        mf_type=2,
        plt_tx=0,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=0,
    )
    pp = SimpleNamespace(
        w=w0,
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=75e-6,
        postDelay=75e-6,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tqs=1e-6,
        trd=2e-6,
        tref=np.array([75e-6, T_180, 75e-6], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        Rsref=np.array([2.0, 2.0, 20.0], dtype=np.float64),
        pcycle=1,
        tacq=np.array([3 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    return sp, pp


def _set_params_matched_imaging() -> tuple[SimpleNamespace, SimpleNamespace]:
    f0 = 7.95e6
    L = 10e-6
    Q = 20.0
    T_90 = 10e-6
    T_180 = 2 * T_90
    sp = SimpleNamespace(
        k=1.381e-23,
        T=300.0,
        gamma=2 * np.pi * 42.6e6,
        grad=1.0,
        D=2e-12,
        f0=f0,
        fin=f0,
        L=L,
        Q=Q,
        R=2 * np.pi * f0 * L / Q,
        Rs=50.0,
        Rin=50.0,
        NF=1.0,
        m0=1.0,
        mth=1.0,
        numpts=2000,
        maxoffs=10.0,
        del_w=np.linspace(-10.0, 10.0, 2000),
        mf_type=2,
        plt_tx=0,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=0,
    )
    pp = SimpleNamespace(
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tref=np.array([3 * T_180, T_180, 3 * T_180], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        tacq=np.array([5 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    return sp, pp


def _calc_tuned_imaging_pulse_shape(
    sp: dict[str, np.ndarray | float],
    pp: SimpleNamespace,
    delay_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the tuned-imaging shaped pulse used by MATLAB's imaging script.

    `sim_cpmg_tuned_probe_img.m` assigns temporary `tp`/`phi`/`amp` fields, but
    `tuned_probe_lp.m` reads the default `tref`/`pref`/`aref` fields instead.
    This helper mirrors that workflow-level behavior so fixture validation
    targets the MATLAB script as executed.
    """

    _tvect_avg, _icr_avg, tvect, icr = tuned_probe_lp(sp, pp)
    if tvect.size < 2:
        raise ValueError("tuned probe pulse shape must contain at least two samples")

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / float(pp.T_90)
    tp = delt * np.ones(tvect.size, dtype=np.float64)
    phi = np.arctan2(np.imag(icr), np.real(icr))
    amp = np.abs(icr)
    amp[amp < float(pp.amp_zero)] = 0

    amp_span = float(np.max(amp) - np.min(amp))
    if amp_span > 0:
        amp = (amp - np.min(amp)) / amp_span
    amp[amp < float(pp.amp_zero)] = 0

    delay_normalized = (np.pi / 2) * float(delay_seconds) / float(pp.T_90)
    return (
        np.concatenate([tp, [-delay_normalized]]),
        np.concatenate([phi, [0.0]]),
        np.concatenate([amp, [0.0]]),
    )


def run_ideal_phase_encoded_cpmg_imaging(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    *,
    t1_map: Iterable[float] | np.ndarray | None = None,
    t2_map: Iterable[float] | np.ndarray | None = None,
    num_echoes: int = 2,
    echo_spacing_seconds: float = 0.2e-3,
    gradient_duration_seconds: float = 0.5e-3,
    fov: tuple[float, float] | Iterable[float] = (20.0, 20.0),
    ny: int = 9,
    maxoffs: float = 5.0,
    num_workers: int | None = 1,
    phase_workers: int | None = 1,
) -> IdealCPMGImagingResult:
    """Run a compact ideal-probe phase-encoded CPMG imaging simulation.

    Mirrors the sequence assembly in MATLAB
    `Sim_CPMG/sim_cpmg_ideal_probe_img.m`, but returns arrays without plotting.
    """

    field_maps, fov_arr = _validate_imaging_inputs(
        rho,
        t1_map,
        t2_map,
        num_echoes,
        ny,
        fov,
    )
    rho_arr = field_maps.rho

    sp0, pp0 = set_params_ideal(numpts=1)
    t90 = float(pp0.T_90)
    t180 = 2 * t90
    if echo_spacing_seconds <= t180:
        raise ValueError("echo_spacing_seconds must be longer than T_180")
    if np.ravel(pp0.tacq)[0] > (echo_spacing_seconds - t180):
        tacq_seconds = echo_spacing_seconds - t180
    else:
        tacq_seconds = float(np.ravel(pp0.tacq)[0])

    maps = _field_maps_from_container(field_maps, ny, maxoffs)
    del_w = maps["del_w"]
    w_1 = maps["w_1"]
    sp = {
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": w_1,
        "T1": maps["T1"],
        "T2": maps["T2"],
        "m0": maps["m0"],
        "mth": maps["mth"],
    }

    rtot = [
        calc_rotation_matrix(del_w, w_1, np.array([np.pi / 2]), np.array([np.pi / 2]), np.array([1.0])),
        calc_rotation_matrix(del_w, w_1, np.array([np.pi / 2]), np.array([3 * np.pi / 2]), np.array([1.0])),
        calc_rotation_matrix(del_w, w_1, np.array([np.pi]), np.array([0.0]), np.array([1.0])),
        calc_rotation_matrix(del_w, w_1, np.array([np.pi]), np.array([np.pi / 2]), np.array([1.0])),
    ]

    texc = np.array([np.pi / 2, -1.0], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tgradn = (np.pi / 2) * gradient_duration_seconds / t90
    tenc = np.array([tgradn, np.pi, tgradn], dtype=np.float64)
    aenc = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    penc1 = np.array([0, 3, 0], dtype=np.int64)
    penc2 = np.array([0, 4, 0], dtype=np.int64)
    acq_enc = np.array([0, 0, 0], dtype=np.int64)
    grad_enc = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    tfp = (np.pi / 2) * (echo_spacing_seconds - t180) / (2 * t90)
    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    pref1 = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
    pref2 = np.tile(np.array([0, 4, 0], dtype=np.int64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": t90,
        "tp": np.concatenate([texc, tenc, tref]),
        "amp": np.concatenate([aexc, aenc, aref]),
        "acq": np.concatenate([acq_exc, acq_enc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_enc, grad_ref]),
        "Rtot": rtot,
    }
    pul1 = np.concatenate([pexc1, penc1, pref1])
    pul2 = np.concatenate([pexc2, penc1, pref1])
    pul3 = np.concatenate([pexc1, penc2, pref2])
    pul4 = np.concatenate([pexc2, penc2, pref2])

    px, pz = rho_arr.shape
    wxmax = np.pi * px**2 / (2 * fov_arr[0] * tgradn)
    wzmax = np.pi * pz**2 / (2 * fov_arr[1] * tgradn)
    gradx = wxmax * np.linspace(-1, 1, px)
    gradz = wzmax * np.linspace(-1, 1, pz)

    tacq = float((np.pi / 2) * tacq_seconds / t90)
    tdw = float((np.pi / 2) * pp0.tdw / t90)
    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    isoc = np.exp(1j * tvect[:, np.newaxis] * del_w[np.newaxis, :])

    def run_point(index: tuple[int, int]) -> np.ndarray:
        ix, iz = index
        sp_case = {
            **sp,
            "del_wg": gradx[ix] * maps["del_wx"] + gradz[iz] * maps["del_wz"],
        }
        pp1 = {**pp_common, "pul": pul1}
        pp2 = {**pp_common, "pul": pul2}
        pp3 = {**pp_common, "pul": pul3}
        pp4 = {**pp_common, "pul": pul4}
        mrx1 = calc_macq_ideal_probe_relax4(sp_case, pp1, num_workers=num_workers)
        mrx2 = calc_macq_ideal_probe_relax4(sp_case, pp2, num_workers=num_workers)
        mrx3 = calc_macq_ideal_probe_relax4(sp_case, pp3, num_workers=num_workers)
        mrx4 = calc_macq_ideal_probe_relax4(sp_case, pp4, num_workers=num_workers)
        echo_x = isoc @ (mrx1 - mrx2).T
        echo_y = isoc @ (mrx3 - mrx4).T
        echo_xy = np.real(echo_x) + 1j * np.imag(echo_y)
        return trapezoid(echo_xy, tvect, axis=0)

    indices = [(ix, iz) for ix in range(px) for iz in range(pz)]
    workers = 1 if phase_workers is None else int(phase_workers)
    if workers <= 1:
        rows = [run_point(index) for index in indices]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(run_point, indices))

    kspace = np.zeros((px, pz, int(num_echoes)), dtype=np.complex128)
    for (ix, iz), values in zip(indices, rows):
        kspace[ix, iz, :] = values
    sequence_time = echo_spacing_seconds * (np.arange(int(num_echoes), dtype=np.float64) + 1)
    return _finish_imaging_result(
        IdealCPMGImagingResult,
        field_maps,
        kspace,
        gradx,
        gradz,
        del_w,
        sequence_time,
        "ideal",
    )


def run_ideal_cpmg_imaging(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    *,
    t1_map: Iterable[float] | np.ndarray | None = None,
    t2_map: Iterable[float] | np.ndarray | None = None,
    num_echoes: int = 2,
    echo_spacing_seconds: float = 0.2e-3,
    gradient_duration_seconds: float = 0.5e-3,
    fov: tuple[float, float] | Iterable[float] = (20.0, 20.0),
    ny: int = 9,
    maxoffs: float = 5.0,
    num_workers: int | None = 1,
    phase_workers: int | None = 1,
) -> IdealCPMGImagingResult:
    """Compatibility alias for `run_ideal_phase_encoded_cpmg_imaging`."""

    return run_ideal_phase_encoded_cpmg_imaging(
        rho,
        t1_map=t1_map,
        t2_map=t2_map,
        num_echoes=num_echoes,
        echo_spacing_seconds=echo_spacing_seconds,
        gradient_duration_seconds=gradient_duration_seconds,
        fov=fov,
        ny=ny,
        maxoffs=maxoffs,
        num_workers=num_workers,
        phase_workers=phase_workers,
    )


def _probe_imaging(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    *,
    probe: str,
    t1_map: Iterable[float] | np.ndarray | None,
    t2_map: Iterable[float] | np.ndarray | None,
    num_echoes: int,
    echo_spacing_seconds: float,
    gradient_duration_seconds: float,
    fov: tuple[float, float] | Iterable[float],
    ny: int,
    maxoffs: float,
    num_workers: int | None,
    phase_workers: int | None,
) -> ProbeCPMGImagingResult:
    field_maps, fov_arr = _validate_imaging_inputs(
        rho,
        t1_map,
        t2_map,
        num_echoes,
        ny,
        fov,
    )
    rho_arr = field_maps.rho

    if probe == "tuned":
        sp0, pp0 = _set_params_tuned_jmr()
    elif probe == "matched":
        sp0, pp0 = _set_params_matched_imaging()
    else:
        raise ValueError("probe must be 'tuned' or 'matched'")

    t90 = float(pp0.T_90)
    t180 = float(pp0.T_180)
    if echo_spacing_seconds <= t180:
        raise ValueError("echo_spacing_seconds must be longer than T_180")
    tacq_seconds = min(float(np.ravel(pp0.tacq)[0]), echo_spacing_seconds - t180)

    maps = _field_maps_from_container(field_maps, ny, maxoffs)
    del_w = maps["del_w"]
    w_1 = maps["w_1"]
    sp = {
        **sp0.__dict__,
        "numpts": del_w.size,
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": w_1,
        "w_1r": maps["w_1r"],
        "T1": maps["T1"],
        "T2": maps["T2"],
        "m0": maps["m0"],
        "mth": maps["mth"],
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }
    if probe == "matched":
        c1, c2 = matching_network_design2(sp0.L, sp0.Q, sp0.f0, sp0.Rs)
        sp["C1"] = c1
        sp["C2"] = c2

    if probe == "tuned":
        exc_y = _calc_tuned_imaging_pulse_shape(sp, pp0, 2 * t90)
        exc_minus_y = _calc_tuned_imaging_pulse_shape(sp, pp0, 2 * t90)
        ref_x = _calc_tuned_imaging_pulse_shape(sp, pp0, 2 * t90)
        ref_y = _calc_tuned_imaging_pulse_shape(sp, pp0, 2 * t90)
        sp["tf"] = tuned_probe_rx_tf(sp, pp0)
    else:
        matched_delay = 2 * t90
        exc_y_tp, exc_y_phi, exc_y_amp, tf1, tf2 = _calc_matched_pulse_shape(
            sp,
            pp0,
            t90,
            np.pi / 2,
            1.0,
            matched_delay,
        )
        exc_y = (exc_y_tp, exc_y_phi, exc_y_amp)
        exc_minus_y = _calc_matched_pulse_shape(
            sp, pp0, t90, 3 * np.pi / 2, 1.0, matched_delay
        )[:3]
        ref_x = _calc_matched_pulse_shape(sp, pp0, t180, 0.0, 1.0, matched_delay)[:3]
        ref_y = _calc_matched_pulse_shape(sp, pp0, t180, np.pi / 2, 1.0, matched_delay)[:3]
        sp["tf1"] = tf1
        sp["tf2"] = tf2

    rtot = [
        calc_rotation_matrix(del_w, w_1, *exc_y),
        calc_rotation_matrix(del_w, w_1, *exc_minus_y),
        calc_rotation_matrix(del_w, w_1, *ref_x),
        calc_rotation_matrix(del_w, w_1, *ref_y),
    ]

    texc = np.array([np.pi / 2, -1.0], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tgradn = (np.pi / 2) * gradient_duration_seconds / t90
    tenc = np.array([tgradn, np.pi, tgradn], dtype=np.float64)
    aenc = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    penc1 = np.array([0, 3, 0], dtype=np.int64)
    penc2 = np.array([0, 4, 0], dtype=np.int64)
    acq_enc = np.array([0, 0, 0], dtype=np.int64)
    grad_enc = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    tfp = (np.pi / 2) * (echo_spacing_seconds - t180) / (2 * t90)
    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    pref1 = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
    pref2 = np.tile(np.array([0, 4, 0], dtype=np.int64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": t90,
        "tp": np.concatenate([texc, tenc, tref]),
        "amp": np.concatenate([aexc, aenc, aref]),
        "acq": np.concatenate([acq_exc, acq_enc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_enc, grad_ref]),
        "Rtot": rtot,
    }
    pul1 = np.concatenate([pexc1, penc1, pref1])
    pul2 = np.concatenate([pexc2, penc1, pref1])
    pul3 = np.concatenate([pexc1, penc2, pref2])
    pul4 = np.concatenate([pexc2, penc2, pref2])

    px, pz = rho_arr.shape
    wxmax = np.pi * px**2 / (2 * fov_arr[0] * tgradn)
    wzmax = np.pi * pz**2 / (2 * fov_arr[1] * tgradn)
    gradx = wxmax * np.linspace(-1, 1, px)
    gradz = wzmax * np.linspace(-1, 1, pz)

    tacq = float((np.pi / 2) * tacq_seconds / t90)
    tdw = float((np.pi / 2) * pp0.tdw / t90)
    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    isoc = np.exp(1j * tvect[:, np.newaxis] * del_w[np.newaxis, :])
    calc_macq = calc_macq_tuned_probe_relax4 if probe == "tuned" else calc_macq_matched_probe_relax4

    def run_point(index: tuple[int, int]) -> np.ndarray:
        ix, iz = index
        sp_case = {
            **sp,
            "del_wg": gradx[ix] * maps["del_wx"] + gradz[iz] * maps["del_wz"],
        }
        pp1 = {**pp_common, "pul": pul1}
        pp2 = {**pp_common, "pul": pul2}
        pp3 = {**pp_common, "pul": pul3}
        pp4 = {**pp_common, "pul": pul4}
        macq1, mrx1 = calc_macq(sp_case, pp1, num_workers=num_workers)
        _macq2, mrx2 = calc_macq(sp_case, pp2, num_workers=num_workers)
        macq3, mrx3 = calc_macq(sp_case, pp3, num_workers=num_workers)
        _macq4, mrx4 = calc_macq(sp_case, pp4, num_workers=num_workers)
        if probe == "tuned":
            echo_x = isoc @ macq1.T
            echo_y = isoc @ macq3.T
        else:
            echo_x = isoc @ (mrx1 - mrx2).T
            echo_y = isoc @ (mrx3 - mrx4).T
        echo_xy = np.imag(echo_x) - 1j * np.real(echo_y)
        if probe == "tuned":
            # Match MATLAB's raw tuned-probe current phase convention.
            echo_xy = -echo_xy
        return trapezoid(echo_xy, tvect, axis=0)

    indices = [(ix, iz) for ix in range(px) for iz in range(pz)]
    workers = 1 if phase_workers is None else int(phase_workers)
    if workers <= 1:
        rows = [run_point(index) for index in indices]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(run_point, indices))

    kspace = np.zeros((px, pz, int(num_echoes)), dtype=np.complex128)
    for (ix, iz), values in zip(indices, rows):
        kspace[ix, iz, :] = values
    sequence_time = echo_spacing_seconds * (np.arange(int(num_echoes), dtype=np.float64) + 1)
    return _finish_imaging_result(
        ProbeCPMGImagingResult,
        field_maps,
        kspace,
        gradx,
        gradz,
        del_w,
        sequence_time,
        probe,
    )


def run_tuned_phase_encoded_cpmg_imaging(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    *,
    t1_map: Iterable[float] | np.ndarray | None = None,
    t2_map: Iterable[float] | np.ndarray | None = None,
    num_echoes: int = 2,
    echo_spacing_seconds: float = 0.2e-3,
    gradient_duration_seconds: float = 0.5e-3,
    fov: tuple[float, float] | Iterable[float] = (20.0, 20.0),
    ny: int = 9,
    maxoffs: float = 5.0,
    num_workers: int | None = 1,
    phase_workers: int | None = 1,
) -> ProbeCPMGImagingResult:
    """Run a compact tuned-probe phase-encoded CPMG imaging simulation."""

    return _probe_imaging(
        rho,
        probe="tuned",
        t1_map=t1_map,
        t2_map=t2_map,
        num_echoes=num_echoes,
        echo_spacing_seconds=echo_spacing_seconds,
        gradient_duration_seconds=gradient_duration_seconds,
        fov=fov,
        ny=ny,
        maxoffs=maxoffs,
        num_workers=num_workers,
        phase_workers=phase_workers,
    )


def run_tuned_cpmg_imaging(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    *,
    t1_map: Iterable[float] | np.ndarray | None = None,
    t2_map: Iterable[float] | np.ndarray | None = None,
    num_echoes: int = 2,
    echo_spacing_seconds: float = 0.2e-3,
    gradient_duration_seconds: float = 0.5e-3,
    fov: tuple[float, float] | Iterable[float] = (20.0, 20.0),
    ny: int = 9,
    maxoffs: float = 5.0,
    num_workers: int | None = 1,
    phase_workers: int | None = 1,
) -> ProbeCPMGImagingResult:
    """Compatibility alias for `run_tuned_phase_encoded_cpmg_imaging`."""

    return run_tuned_phase_encoded_cpmg_imaging(
        rho,
        t1_map=t1_map,
        t2_map=t2_map,
        num_echoes=num_echoes,
        echo_spacing_seconds=echo_spacing_seconds,
        gradient_duration_seconds=gradient_duration_seconds,
        fov=fov,
        ny=ny,
        maxoffs=maxoffs,
        num_workers=num_workers,
        phase_workers=phase_workers,
    )


def run_matched_phase_encoded_cpmg_imaging(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    *,
    t1_map: Iterable[float] | np.ndarray | None = None,
    t2_map: Iterable[float] | np.ndarray | None = None,
    num_echoes: int = 2,
    echo_spacing_seconds: float = 0.2e-3,
    gradient_duration_seconds: float = 0.5e-3,
    fov: tuple[float, float] | Iterable[float] = (20.0, 20.0),
    ny: int = 9,
    maxoffs: float = 5.0,
    num_workers: int | None = 1,
    phase_workers: int | None = 1,
) -> ProbeCPMGImagingResult:
    """Run a compact matched-probe phase-encoded CPMG imaging simulation."""

    return _probe_imaging(
        rho,
        probe="matched",
        t1_map=t1_map,
        t2_map=t2_map,
        num_echoes=num_echoes,
        echo_spacing_seconds=echo_spacing_seconds,
        gradient_duration_seconds=gradient_duration_seconds,
        fov=fov,
        ny=ny,
        maxoffs=maxoffs,
        num_workers=num_workers,
        phase_workers=phase_workers,
    )


def run_matched_cpmg_imaging(
    rho: Iterable[float] | np.ndarray | ImagingFieldMaps,
    *,
    t1_map: Iterable[float] | np.ndarray | None = None,
    t2_map: Iterable[float] | np.ndarray | None = None,
    num_echoes: int = 2,
    echo_spacing_seconds: float = 0.2e-3,
    gradient_duration_seconds: float = 0.5e-3,
    fov: tuple[float, float] | Iterable[float] = (20.0, 20.0),
    ny: int = 9,
    maxoffs: float = 5.0,
    num_workers: int | None = 1,
    phase_workers: int | None = 1,
) -> ProbeCPMGImagingResult:
    """Compatibility alias for `run_matched_phase_encoded_cpmg_imaging`."""

    return run_matched_phase_encoded_cpmg_imaging(
        rho,
        t1_map=t1_map,
        t2_map=t2_map,
        num_echoes=num_echoes,
        echo_spacing_seconds=echo_spacing_seconds,
        gradient_duration_seconds=gradient_duration_seconds,
        fov=fov,
        ny=ny,
        maxoffs=maxoffs,
        num_workers=num_workers,
        phase_workers=phase_workers,
    )
