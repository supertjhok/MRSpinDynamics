"""Inline magnet/coil spacing constraints from finite magnets and exact Zeeman spectra."""

from dataclasses import replace
from functools import lru_cache
import itertools
import numpy as np
from spin_dynamics.nqr.hamiltonians import batched_nqr_hamiltonians
from spin_dynamics.nqr.orientations import powder_average_grid
from spin_dynamics.nqr.polarization_enhancement import HalbachPrepolarizationMagnet
from reference import material_reference


@lru_cache(maxsize=64)
def spacing_constraint(
    radius=0.23,
    length=0.46,
    width=0.08,
    remanence=1.15,
    spacing=1.2,
    coil_length=0.46,
    pulse_s=60e-6,
    loaded_q=30.0,
    resolution=1,
):
    """Sample the entire 430 x 340 x 35 mm clear mail volume inside the coil.

    The tolerance is a provisional engineering allocation, not a material constant.
    Sampling convergence must be assessed before making a hardware decision.
    """
    eig, line, values = material_reference()
    z_extent = min(0.215, coil_length / 2)
    points = np.array(
        list(
            itertools.product(
                np.linspace(-0.17, 0.17, 3),
                np.linspace(-0.0175, 0.0175, 3),
                np.linspace(-z_extent, z_extent, 5),
            )
        )
    )
    points[:, 2] += spacing
    magnet = HalbachPrepolarizationMagnet(
        center_radius=radius,
        length=length,
        rod_width=width,
        remanence=remanence,
        n_cross=3 + 2 * (resolution - 1),
        n_length=21 + 10 * (resolution - 1),
    )
    fields = np.linalg.norm(magnet.b0_vector(points), axis=1)
    directions = np.array(
        [
            o.b1_direction_pas
            for o in powder_average_grid(12 * resolution, 24 * resolution)
        ]
        + list(np.eye(3))
        + list(-np.eye(3))
    )
    # Eigenvalues of H_Q + H_Z, not gamma*B interpreted as an NQR splitting.
    b = (fields[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    levels = np.linalg.eigvalsh(batched_nqr_hamiltonians(eig.site, b)) / (2 * np.pi)
    frequencies = levels[:, line.upper] - levels[:, line.lower]
    deviations = frequencies - line.frequency_hz
    native_linewidth = 1 / (np.pi * values["t2_star_s"])
    allowance = 0.1 * min(native_linewidth, 1 / pulse_s, line.frequency_hz / loaded_q)
    maximum = float(np.max(np.abs(deviations)))
    physical_separation = spacing > (length + coil_length) / 2
    return {
        "centre_spacing_m": spacing,
        "surface_gap_m": spacing - (length + coil_length) / 2,
        "maximum_fringe_field_t": float(fields.max()),
        "maximum_line_shift_hz": maximum,
        "powder_and_volume_frequency_span_hz": float(np.ptp(frequencies)),
        "allowed_zeeman_perturbation_hz": allowance,
        "passes": bool(maximum <= allowance and physical_separation),
        "spatial_points": len(points),
        "orientations": len(directions),
        "criterion": "max absolute line shift <= 10% of min(native linewidth, inverse pulse duration, loaded bandwidth)",
        "scope": "sampled exact spin-1 Zeeman spectrum; aniline x line; not a certified bound",
    }


def for_config(cfg, resolution=1):
    return spacing_constraint(
        cfg.magnet_radius_m,
        cfg.magnet_length_m,
        cfg.magnet_rod_width_m,
        cfg.magnet_remanence_t,
        cfg.magnet_coil_spacing_m,
        cfg.coil_length_m,
        cfg.pulse_s,
        cfg.loaded_q,
        resolution,
    )


def spacing_sweep(cfg):
    return [
        for_config(replace(cfg, magnet_coil_spacing_m=spacing))
        for spacing in (0.48, 0.55, 0.65, 0.8, 1.0, 1.2, 1.6)
    ]
