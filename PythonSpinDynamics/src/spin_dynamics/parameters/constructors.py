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
