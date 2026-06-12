"""Python equivalents of active MATLAB parameter constructors.

MATLAB reference folder:
    SpinDynamicsUpdated/Version_2/code/Params
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SystemParameters:
    """Simulation/system parameters corresponding to MATLAB `sp`."""

    k: float
    T: float
    gamma: float
    grad: float
    D: float
    f0: float
    fin: float
    m0: float
    mth: float
    numpts: int
    maxoffs: float
    del_w: np.ndarray
    mf_type: int
    plt_tx: int
    plt_rx: int
    plt_sequence: int
    plt_axis: int
    plt_mn: int
    plt_echo: int


@dataclass(frozen=True)
class PulseParameters:
    """Pulse-sequence parameters corresponding to MATLAB `pp`."""

    N: int
    T_90: float
    T_180: float
    psi: float
    preDelay: float
    postDelay: float
    texc: np.ndarray
    pexc: np.ndarray
    aexc: np.ndarray
    tcorr: float
    tref: np.ndarray
    pref: np.ndarray
    aref: np.ndarray
    pcycle: int
    tacq: np.ndarray
    tdw: float
    amp_zero: float


@dataclass(frozen=True)
class FIDSystemParameters:
    """Simulation/system parameters corresponding to ideal FID MATLAB `sp`."""

    k: float
    T: float
    f0: float
    fin: float
    m0: float
    mth: float
    numpts: int
    maxoffs: float
    del_w: np.ndarray
    w_1: np.ndarray
    w_1r: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    mf_type: int
    plt_tx: int
    plt_rx: int
    plt_sequence: int
    plt_axis: int
    plt_mn: int
    plt_echo: int


@dataclass(frozen=True)
class FIDPulseParameters:
    """Pulse-sequence parameters corresponding to ideal FID MATLAB `pp`."""

    N: int
    T_90: float
    acqDelay: float
    acqTpTime: float
    psi: float
    tacq: float
    tdw: float
    amp_zero: float


@dataclass(frozen=True)
class TunedOrigParameters:
    """Compact tuned-probe parameters corresponding to MATLAB `params`."""

    texc: np.ndarray
    pexc: np.ndarray
    aexc: np.ndarray
    trd: float
    tref: np.ndarray
    pref: np.ndarray
    aref: np.ndarray
    tfp: float
    tqs: float
    tacq: np.ndarray
    Rs: np.ndarray
    pcycle: int


@dataclass(frozen=True)
class TunedSystemParameters:
    """Tuned-probe system parameters corresponding to MATLAB `sp`."""

    k: float
    T: float
    gamma: float
    f0: float
    fin: float
    w0: float
    L: float
    Q: float
    R: float
    C: float
    Rs: float
    Vs: float
    Rin: float
    Cin: float
    Rd: float
    NF: float
    vn: float
    in_: float
    m0: float
    mth: float
    numpts: int
    maxoffs: float
    del_w: np.ndarray
    mf_type: int
    plt_tx: int
    plt_rx: int
    plt_sequence: int
    plt_axis: int
    plt_mn: int
    plt_echo: int
    sens: float


@dataclass(frozen=True)
class TunedPulseParameters:
    """Tuned-probe pulse parameters corresponding to MATLAB `pp`."""

    w: float
    N: int
    T_90: float
    T_180: float
    psi: float
    preDelay: float
    postDelay: float
    texc: np.ndarray
    pexc: np.ndarray
    aexc: np.ndarray
    tcorr: float
    tqs: float
    trd: float
    tref: np.ndarray
    pref: np.ndarray
    aref: np.ndarray
    Rsref: np.ndarray
    pcycle: int
    tacq: np.ndarray
    tdw: float
    amp_zero: float


@dataclass(frozen=True)
class UntunedOrigParameters:
    """Compact untuned-probe parameters corresponding to MATLAB `params`."""

    texc: np.ndarray
    pexc: np.ndarray
    aexc: np.ndarray
    trd: float
    tref: np.ndarray
    pref: np.ndarray
    aref: np.ndarray
    tfp: float
    tqs: float
    tacq: np.ndarray
    Rs: np.ndarray
    pcycle: int


@dataclass(frozen=True)
class UntunedSystemParameters:
    """Untuned-probe system parameters corresponding to MATLAB `sp`."""

    k: float
    T: float
    gamma: float
    f0: float
    fin: float
    w0: float
    L: float
    Q: float
    R: float
    C: float
    Rs: float
    Vs: float
    Rin: float
    Cin: float
    Rd: float
    Rdup: float
    Nrx: float
    krx: float
    L1: float
    R1: float
    L2: float
    R2: float
    NF: float
    vn: float
    in_: float
    m0: float
    mth: float
    numpts: int
    maxoffs: float
    del_w: np.ndarray
    mf_type: int
    plt_tx: int
    plt_rx: int
    plt_sequence: int
    plt_axis: int
    plt_mn: int
    plt_echo: int
    sens: float


@dataclass(frozen=True)
class UntunedPulseParameters:
    """Untuned-probe pulse parameters corresponding to MATLAB `pp`."""

    w: float
    N: int
    T_90: float
    T_180: float
    psi: float
    preDelay: float
    postDelay: float
    texc: np.ndarray
    pexc: np.ndarray
    aexc: np.ndarray
    tcorr: float
    tqs: float
    trd: float
    tref: np.ndarray
    pref: np.ndarray
    aref: np.ndarray
    Rsref: np.ndarray
    tacq: np.ndarray
    tdw: float
    amp_zero: float


@dataclass(frozen=True)
class MatchedSystemParameters:
    """Matched-probe system parameters corresponding to MATLAB `sp`."""

    k: float
    T: float
    gamma: float
    grad: float
    D: float
    f0: float
    fin: float
    L: float
    Q: float
    R: float
    Rs: float
    Rin: float
    NF: float
    m0: float
    mth: float
    numpts: int
    maxoffs: float
    del_w: np.ndarray
    mf_type: int
    plt_tx: int
    plt_rx: int
    plt_sequence: int
    plt_axis: int
    plt_mn: int
    plt_echo: int


@dataclass(frozen=True)
class MatchedPulseParameters:
    """Matched-probe pulse parameters corresponding to MATLAB `pp`."""

    N: int
    T_90: float
    T_180: float
    psi: float
    preDelay: float
    postDelay: float
    texc: np.ndarray
    pexc: np.ndarray
    aexc: np.ndarray
    tcorr: float
    trd: float
    tref: np.ndarray
    pref: np.ndarray
    aref: np.ndarray
    tacq: np.ndarray
    tdw: float
    amp_zero: float


def set_params_ideal(numpts: int = 10_000) -> tuple[SystemParameters, PulseParameters]:
    """Construct default ideal no-probe CPMG parameters.

    Mirrors MATLAB `Params/set_params_ideal.m`. The optional `numpts` argument
    keeps tests lightweight while preserving MATLAB's default when omitted.
    """

    maxoffs = 10.0
    T_90 = 25e-6
    T_180 = 2 * T_90
    pre_delay = 75e-6
    post_delay = 75e-6

    sp = SystemParameters(
        k=1.381e-23,
        T=300.0,
        gamma=2 * np.pi * 42.577e6,
        grad=1.0,
        D=2e-12,
        f0=0.5e6,
        fin=0.5e6,
        m0=1.0,
        mth=1.0,
        numpts=int(numpts),
        maxoffs=maxoffs,
        del_w=np.linspace(-maxoffs, maxoffs, int(numpts)),
        mf_type=2,
        plt_tx=0,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=0,
    )
    pp = PulseParameters(
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=pre_delay,
        postDelay=post_delay,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tref=np.array([pre_delay, T_180, post_delay], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        pcycle=1,
        tacq=np.array([3 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    return sp, pp


def set_params_ideal_fid(
    numpts: int = 20_000,
) -> tuple[FIDSystemParameters, FIDPulseParameters]:
    """Construct default ideal no-probe FID parameters.

    Mirrors MATLAB `Params/set_params_ideal_FID.m`. The optional `numpts`
    argument keeps validation fixtures lightweight while preserving MATLAB's
    default when omitted.
    """

    T1 = 2000e-3
    T2 = 2000e-3
    maxoffs = 10.0
    del_w = np.linspace(-maxoffs, maxoffs, int(numpts))
    T_90 = 25e-6

    sp = FIDSystemParameters(
        k=1.381e-23,
        T=300.0,
        f0=10e6,
        fin=10e6,
        m0=1.0,
        mth=1.0,
        numpts=int(numpts),
        maxoffs=maxoffs,
        del_w=del_w,
        w_1=np.ones(int(numpts), dtype=np.float64),
        w_1r=np.ones(int(numpts), dtype=np.float64),
        T1=T1 * np.ones(int(numpts), dtype=np.float64),
        T2=T2 * np.ones(int(numpts), dtype=np.float64),
        mf_type=1,
        plt_tx=0,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=1,
        plt_echo=1,
    )
    pp = FIDPulseParameters(
        N=32,
        T_90=T_90,
        acqDelay=T_90 / 10,
        acqTpTime=T_90,
        psi=0.0,
        tacq=T_90,
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    return sp, pp


def set_params_tuned_orig(
    numpts: int = 10_000,
) -> tuple[TunedOrigParameters, TunedSystemParameters, TunedPulseParameters]:
    """Construct original/reference tuned-probe CPMG parameters.

    Mirrors MATLAB `Params/set_params_tuned_Orig.m`.
    """

    gamma = 2 * np.pi * 42.577e6
    f0 = 1e6
    fin = 1e6
    w0 = 2 * np.pi * fin
    L = 10e-6
    Q = 50.0
    R = 2 * np.pi * f0 * L / Q
    C = 1 / ((2 * np.pi * f0) ** 2 * L)
    maxoffs = 10.0
    T_90 = 25e-6
    T_180 = 2 * T_90
    pre_delay = 75e-6
    post_delay = 75e-6
    Vs = 1.0

    sens = ((np.pi / 2) / T_90) * (2 * w0 * L) / (gamma * Vs)
    sp = TunedSystemParameters(
        k=1.381e-23,
        T=300.0,
        gamma=gamma,
        f0=f0,
        fin=fin,
        w0=w0,
        L=L,
        Q=Q,
        R=R,
        C=C,
        Rs=1.0,
        Vs=Vs,
        Rin=1e6,
        Cin=5e-12,
        Rd=1e6,
        NF=1.0,
        vn=0.5e-9,
        in_=2e-15,
        m0=1.0,
        mth=1.0,
        numpts=int(numpts),
        maxoffs=maxoffs,
        del_w=np.linspace(-maxoffs, maxoffs, int(numpts)),
        mf_type=2,
        plt_tx=1,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=1,
        sens=sens,
    )
    pp = TunedPulseParameters(
        w=w0,
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=pre_delay,
        postDelay=post_delay,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tqs=5e-6,
        trd=5e-6,
        tref=np.array([pre_delay, T_180, post_delay], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        Rsref=np.array([2.0, 2.0, 20.0], dtype=np.float64),
        pcycle=1,
        tacq=np.array([3 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    params = TunedOrigParameters(
        texc=pp.texc,
        pexc=pp.pexc,
        aexc=pp.aexc,
        trd=pp.trd,
        tref=np.array([pp.tref[1]], dtype=np.float64),
        pref=np.array([pp.pref[1]], dtype=np.float64),
        aref=np.array([pp.aref[1]], dtype=np.float64),
        tfp=pp.preDelay,
        tqs=pp.tqs,
        tacq=pp.tacq,
        Rs=np.array([pp.Rsref[0], pp.Rsref[1], pp.Rsref[2]], dtype=np.float64),
        pcycle=1,
    )
    return params, sp, pp


def set_params_tuned_spa(
    numpts: int = 5_000,
) -> tuple[TunedOrigParameters, TunedSystemParameters, TunedPulseParameters]:
    """Construct tuned-probe SPA pulse-evaluation parameters.

    Mirrors MATLAB `Params/set_params_tuned_SPA.m`.
    """

    gamma = 2 * np.pi * 42.577e6
    f0 = 8e6
    fin = 8e6
    w0 = 2 * np.pi * fin
    L = 10e-6 * (1e6 / f0)
    Q = 50.0
    T_90 = 24e-6
    T_180 = 2 * T_90
    pre_delay = 144e-6
    post_delay = 144e-6
    Vs = 1.0
    sens = ((np.pi / 2) / T_90) * (2 * w0 * L) / (gamma * Vs)

    sp = TunedSystemParameters(
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
        Rs=1.0,
        Vs=Vs,
        Rin=1e6,
        Cin=5e-12,
        Rd=1e6,
        NF=1.0,
        vn=0.5e-9,
        in_=2e-15,
        m0=1.0,
        mth=1.0,
        numpts=int(numpts),
        maxoffs=10.0,
        del_w=np.linspace(-10.0, 10.0, int(numpts)),
        mf_type=2,
        plt_tx=1,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=1,
        sens=sens,
    )
    pp = TunedPulseParameters(
        w=w0,
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=pre_delay,
        postDelay=post_delay,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tqs=8e-6,
        trd=8e-6,
        tref=np.array([pre_delay, T_180, post_delay], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        Rsref=np.array([2.0, 2.0, 20.0], dtype=np.float64),
        pcycle=1,
        tacq=np.array([4 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    params = TunedOrigParameters(
        texc=pp.texc,
        pexc=pp.pexc,
        aexc=pp.aexc,
        trd=pp.trd,
        tref=np.array([pp.tref[1]], dtype=np.float64),
        pref=np.array([pp.pref[1]], dtype=np.float64),
        aref=np.array([pp.aref[1]], dtype=np.float64),
        tfp=pp.preDelay,
        tqs=pp.tqs,
        tacq=pp.tacq,
        Rs=np.array([pp.Rsref[0], pp.Rsref[1], pp.Rsref[2]], dtype=np.float64),
        pcycle=1,
    )
    return params, sp, pp


def set_params_tuned_jmr(
    numpts: int = 10_000,
) -> tuple[TunedSystemParameters, TunedPulseParameters]:
    """Construct JMR-paper tuned-probe parameters.

    Mirrors MATLAB `Params/set_params_tuned_JMR.m`.
    """

    gamma = 42.577e6 * 2 * np.pi
    f0 = 0.5e6
    fin = 0.5e6
    w0 = 2 * np.pi * fin
    L = 10e-6
    Q = 50.0
    T_90 = 25e-6
    T_180 = 2 * T_90
    pre_delay = 75e-6
    post_delay = 75e-6
    Vs = 1.0
    sens = ((np.pi / 2) / T_90) * (2 * w0 * L) / (gamma * Vs)

    sp = TunedSystemParameters(
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
        numpts=int(numpts),
        maxoffs=10.0,
        del_w=np.linspace(-10.0, 10.0, int(numpts)),
        mf_type=2,
        plt_tx=0,
        plt_rx=1,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=0,
        sens=sens,
    )
    pp = TunedPulseParameters(
        w=w0,
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=pre_delay,
        postDelay=post_delay,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tqs=1e-6,
        trd=2e-6,
        tref=np.array([pre_delay, T_180, post_delay], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        Rsref=np.array([2.0, 2.0, 20.0], dtype=np.float64),
        pcycle=1,
        tacq=np.array([3 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    return sp, pp


def set_params_untuned_orig(
    numpts: int = 10_000,
) -> tuple[UntunedOrigParameters, UntunedSystemParameters, UntunedPulseParameters]:
    """Construct original/reference untuned-probe CPMG parameters.

    Mirrors MATLAB `Params/set_params_untuned_Orig.m`.
    """

    gamma = 2 * np.pi * 42.577e6
    f0 = 0.5e6
    fin = 0.5e6
    w0 = 2 * np.pi * fin
    L = 10e-6
    Q = 50.0
    R = 2 * np.pi * f0 * L / Q
    C = 1 / ((2 * np.pi * 10 * f0) ** 2 * L)
    maxoffs = 10.0
    T_90 = 26e-6
    T_180 = 2 * T_90
    pre_delay = 78e-6
    post_delay = 78e-6
    Vs = 1.0
    L1 = 75e-6
    R1 = 0.26
    Nrx = 4.0

    sens = ((np.pi / 2) / T_90) * (2 * w0 * L) / (gamma * Vs)
    sp = UntunedSystemParameters(
        k=1.381e-23,
        T=300.0,
        gamma=gamma,
        f0=f0,
        fin=fin,
        w0=w0,
        L=L,
        Q=Q,
        R=R,
        C=C,
        Rs=2.0,
        Vs=Vs,
        Rin=1e6,
        Cin=5e-12,
        Rd=1e6,
        Rdup=0.2,
        Nrx=Nrx,
        krx=0.9996,
        L1=L1,
        R1=R1,
        L2=1250e-6,
        R2=0.91,
        NF=1.0,
        vn=0.5e-9,
        in_=2e-15,
        m0=1.0,
        mth=1.0,
        numpts=int(numpts),
        maxoffs=maxoffs,
        del_w=np.linspace(-maxoffs, maxoffs, int(numpts)),
        mf_type=2,
        plt_tx=0,
        plt_rx=0,
        plt_sequence=1,
        plt_axis=1,
        plt_mn=1,
        plt_echo=1,
        sens=sens,
    )
    pp = UntunedPulseParameters(
        w=w0,
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=pre_delay,
        postDelay=post_delay,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tqs=8e-6,
        trd=8e-6,
        tref=np.array([pre_delay, T_180, post_delay], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        Rsref=np.array([2.0, 2.0, 20.0], dtype=np.float64),
        tacq=np.array([3 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    params = UntunedOrigParameters(
        texc=pp.texc,
        pexc=pp.pexc,
        aexc=pp.aexc,
        trd=pp.trd,
        tref=np.array([pp.tref[1]], dtype=np.float64),
        pref=np.array([pp.pref[1]], dtype=np.float64),
        aref=np.array([pp.aref[1]], dtype=np.float64),
        tfp=pp.preDelay,
        tqs=pp.tqs,
        tacq=pp.tacq,
        Rs=np.array([pp.Rsref[0], pp.Rsref[1], pp.Rsref[2]], dtype=np.float64),
        pcycle=1,
    )
    return params, sp, pp


def set_params_untuned_spa(
    numpts: int = 5_000,
) -> tuple[UntunedOrigParameters, UntunedSystemParameters, UntunedPulseParameters]:
    """Construct untuned-probe SPA pulse-evaluation parameters.

    Mirrors MATLAB `Params/set_params_untuned_SPA.m`.
    """

    gamma = 2 * np.pi * 42.577e6
    f0 = 8e6
    fin = 8e6
    w0 = 2 * np.pi * fin
    L = 10e-6 * (1e6 / f0)
    Q = 50.0
    T_90 = 24e-6
    T_180 = 2 * T_90
    pre_delay = 144e-6
    post_delay = 144e-6
    Vs = 1.0
    L1 = 75e-6
    R1 = 0.26
    Nrx = 4.0
    sens = ((np.pi / 2) / T_90) * (2 * w0 * L) / (gamma * Vs)

    sp = UntunedSystemParameters(
        k=1.381e-23,
        T=300.0,
        gamma=gamma,
        f0=f0,
        fin=fin,
        w0=w0,
        L=L,
        Q=Q,
        R=2 * np.pi * f0 * L / Q,
        C=1 / ((2 * np.pi * 10 * f0) ** 2 * L),
        Rs=1.0,
        Vs=Vs,
        Rin=1e6,
        Cin=5e-12,
        Rd=1e6,
        Rdup=0.2,
        Nrx=Nrx,
        krx=0.9996,
        L1=L1,
        R1=R1,
        L2=1250e-6,
        R2=0.91,
        NF=1.0,
        vn=0.5e-9,
        in_=2e-15,
        m0=1.0,
        mth=1.0,
        numpts=int(numpts),
        maxoffs=10.0,
        del_w=np.linspace(-10.0, 10.0, int(numpts)),
        mf_type=2,
        plt_tx=1,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=1,
        sens=sens,
    )
    pp = UntunedPulseParameters(
        w=w0,
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=pre_delay,
        postDelay=post_delay,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tqs=8e-6,
        trd=8e-6,
        tref=np.array([pre_delay, T_180, post_delay], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        Rsref=np.array([2.0, 2.0, 20.0], dtype=np.float64),
        tacq=np.array([4 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    params = UntunedOrigParameters(
        texc=pp.texc,
        pexc=pp.pexc,
        aexc=pp.aexc,
        trd=pp.trd,
        tref=np.array([pp.tref[1]], dtype=np.float64),
        pref=np.array([pp.pref[1]], dtype=np.float64),
        aref=np.array([pp.aref[1]], dtype=np.float64),
        tfp=pp.preDelay,
        tqs=pp.tqs,
        tacq=pp.tacq,
        Rs=np.array([pp.Rsref[0], pp.Rsref[1], pp.Rsref[2]], dtype=np.float64),
        pcycle=1,
    )
    return params, sp, pp


def set_params_untuned_jmr(
    numpts: int = 2000,
) -> tuple[UntunedSystemParameters, UntunedPulseParameters]:
    """Construct JMR-paper untuned-probe parameters.

    Mirrors MATLAB `Params/set_params_untuned_JMR.m`.
    """

    gamma = 2 * np.pi * 42.577e6
    f0 = 0.5e6
    fin = 0.5e6
    w0 = 2 * np.pi * fin
    L = 10e-6
    Q = 50.0
    T_90 = 25e-6
    T_180 = 2 * T_90
    Vs = 1.0
    L1 = 75e-6
    R1 = 0.26
    Nrx = 4.0
    sens = ((np.pi / 2) / T_90) * (2 * w0 * L) / (gamma * Vs)

    sp = UntunedSystemParameters(
        k=1.381e-23,
        T=300.0,
        gamma=gamma,
        f0=f0,
        fin=fin,
        w0=w0,
        L=L,
        Q=Q,
        R=2 * np.pi * f0 * L / Q,
        C=1 / ((2 * np.pi * 10 * f0) ** 2 * L),
        Rs=2.0,
        Vs=Vs,
        Rin=1e6,
        Cin=5e-12,
        Rd=1e6,
        Rdup=0.2,
        Nrx=Nrx,
        krx=0.9996,
        L1=L1,
        R1=R1,
        L2=1250e-6,
        R2=0.91,
        NF=1.0,
        vn=0.45e-9,
        in_=2e-15,
        m0=1.0,
        mth=1.0,
        numpts=int(numpts),
        maxoffs=10.0,
        del_w=np.linspace(-10.0, 10.0, int(numpts)),
        mf_type=2,
        plt_tx=0,
        plt_rx=1,
        plt_sequence=1,
        plt_axis=1,
        plt_mn=1,
        plt_echo=1,
        sens=sens,
    )
    pp = UntunedPulseParameters(
        w=w0,
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=20e-6,
        postDelay=50e-6,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        tqs=8e-6,
        trd=8e-6,
        tref=np.array([20e-6, T_180, 50e-6], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        Rsref=np.array([2.0, 2.0, 20.0], dtype=np.float64),
        tacq=np.array([5 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    return sp, pp


def set_params_matched_orig(
    numpts: int = 10_000,
) -> tuple[MatchedSystemParameters, MatchedPulseParameters]:
    """Construct original/reference matched-probe CPMG parameters.

    Mirrors MATLAB `Params/set_params_matched_Orig.m`.
    """

    gamma = 2 * np.pi * 42.577e6
    f0 = 1e6
    L = 10e-6
    Q = 50.0
    T_90 = 25e-6
    T_180 = 2 * T_90
    pre_delay = 150e-6
    post_delay = 150e-6
    maxoffs = 10.0

    sp = MatchedSystemParameters(
        k=1.381e-23,
        T=300.0,
        gamma=gamma,
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
        numpts=int(numpts),
        maxoffs=maxoffs,
        del_w=np.linspace(-maxoffs, maxoffs, int(numpts)),
        mf_type=2,
        plt_tx=0,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=0,
    )
    pp = MatchedPulseParameters(
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=pre_delay,
        postDelay=post_delay,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        trd=3 * T_90,
        tref=np.array([pre_delay, T_180, post_delay], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        tacq=np.array([3 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    return sp, pp


def set_params_matched_jmr(
    numpts: int = 2000,
) -> tuple[MatchedSystemParameters, MatchedPulseParameters]:
    """Construct JMR-paper matched-probe parameters.

    Mirrors MATLAB `Params/set_params_matched_JMR.m`.
    """

    f0 = 0.5e6
    L = 10e-6
    Q = 50.0
    T_90 = 25e-6
    T_180 = 2 * T_90
    sp = MatchedSystemParameters(
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
        numpts=int(numpts),
        maxoffs=10.0,
        del_w=np.linspace(-10.0, 10.0, int(numpts)),
        mf_type=2,
        plt_tx=0,
        plt_rx=1,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=1,
    )
    pp = MatchedPulseParameters(
        N=32,
        T_90=T_90,
        T_180=T_180,
        psi=0.0,
        preDelay=20e-6,
        postDelay=70e-6,
        texc=np.array([T_90], dtype=np.float64),
        pexc=np.array([np.pi / 2], dtype=np.float64),
        aexc=np.array([1.0], dtype=np.float64),
        tcorr=-(2 / np.pi) * T_90,
        trd=3 * T_90,
        tref=np.array([20e-6, T_180, 70e-6], dtype=np.float64),
        pref=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        aref=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        tacq=np.array([5 * T_180], dtype=np.float64),
        tdw=0.5e-6,
        amp_zero=1e-4,
    )
    return sp, pp
