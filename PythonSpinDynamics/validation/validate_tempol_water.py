"""Offline published TEMPOL/water comparison; no fitting or network access.

Run from PythonSpinDynamics:
python validation/validate_tempol_water.py --output .tmp/tempol_benchmark.json
Optional --source-pdf verifies the author PDF and embedded-figure checksums.
The observations, coordinates, provenance and reduction choices live in
validation/experimental/tempol_water_published.json. Source-figure extraction
uses optional PyMuPDF only for --source-pdf; normal comparison uses NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.dnp import (  # noqa: E402
    NitroxideSaturationModel,
    NitroxideRelaxationMatrix,
    bloch_saturation,
    overhauser_from_factors,
)
from spin_dynamics.dnp.overhauser import ELECTRON_GAMMA_HZ_PER_T  # noqa: E402

DATA_PATH = ROOT / "validation/experimental/tempol_water_published.json"


def load_data(path=DATA_PATH):
    """Load the frozen transcription without fetching or altering observations."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digitized_points(record):
    """Reproduce axis coordinates from stored native-image pixel centers."""
    points = np.asarray(record["points_pixels"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("digitization points must be finite (n,2) pixel coordinates")
    converted = []
    for dimension, name in enumerate(("axis_x_pixels_values", "axis_y_pixels_values")):
        axis = np.asarray(record[name], dtype=float)
        if (
            axis.shape != (2, 2)
            or not np.isfinite(axis).all()
            or axis[0, 0] == axis[1, 0]
        ):
            raise ValueError("axis requires two distinct finite pixel/value anchors")
        converted.append(
            axis[0, 1]
            + (points[:, dimension] - axis[0, 0])
            * (axis[1, 1] - axis[0, 1])
            / (axis[1, 0] - axis[0, 0])
        )
    return np.column_stack(converted)


def published_relaxation(data):
    """Build the source's full nitrogen operator from tensors and MD spectra.

    Angular-velocity amplitudes in rad^2/ps^2 are converted to s^-2. The
    spin-rotation coupling uses T=g-g_free*I and Eq. (37), including both
    oscillatory Lorentzian terms. There is no enhancement/saturation fitting.
    """
    p = data["sezer_profile"]
    f = p["full_relaxation"]
    g = np.asarray(f["g_tensor"])
    free_g = 2.00231930436256
    t0 = g.mean() - free_g
    anisotropy_squared = np.sum((g - g.mean()) ** 2) / 3
    a1 = f["angular_velocity_a1_rad2_per_ps2"]
    a2 = f["angular_velocity_a1_plus_a2_rad2_per_ps2"] - a1
    tau1, tau2, tau3 = [f[f"angular_velocity_tau{i}_seconds"] for i in (1, 2, 3)]

    def spin_rotation(omega):
        return (
            (t0**2 + anisotropy_squared)
            * 1e24
            * (
                a1 * tau1 / (1 + (omega * tau1) ** 2)
                + a2 * tau2 / 2 / (1 + ((omega - 1 / tau3) * tau2) ** 2)
                + a2 * tau2 / 2 / (1 + ((omega + 1 / tau3) * tau2) ** 2)
            )
        )

    gamma = 2 * np.pi * ELECTRON_GAMMA_HZ_PER_T
    return NitroxideRelaxationMatrix.from_isotropic_tumbling(
        field_tesla=p["field_tesla"],
        correlation_time_seconds=f["rotational_correlation_time_seconds"],
        g_tensor=g,
        hyperfine_tensor_hz=np.asarray(p["hyperfine_tensor_gauss"])
        * ELECTRON_GAMMA_HZ_PER_T
        * 1e-4,
        spin_rotation_j0_per_second=spin_rotation(0),
        spin_rotation_jomega_per_second=spin_rotation(gamma * p["field_tesla"]),
        additional_r1_per_second=f["additional_r1_gauss"] * gamma * 1e-4,
        additional_r2_per_second=f["additional_r2_gauss"] * gamma * 1e-4,
    )


def profile_model(
    data, detuning_mhz, *, exchange_scale=1.0, b1_scale=1.0, relaxation="full"
):
    """Evaluate full, line-dependent or legacy reduced models without fitting.

    The source plots microwave minus resonance frequency; the API takes
    resonance minus microwave, hence the minus sign in the detuning conversion.
    B1 is the co-rotating amplitude in gauss; no additional factor of two.
    """
    p = data["sezer_profile"]
    if relaxation not in ("full", "line_dependent", "reduced"):
        raise ValueError("relaxation must be full, line_dependent or reduced")
    t1 = np.inf if relaxation == "full" else p["t1e_seconds"]
    t2 = (
        np.inf
        if relaxation == "full"
        else (
            p["t2e_seconds_by_nitrogen_m_ascending"]
            if relaxation == "line_dependent"
            else p["t2e_seconds_by_nitrogen_m_ascending"][1]
        )
    )
    model = NitroxideSaturationModel(
        hyperfine_hz=np.mean(p["hyperfine_tensor_gauss"])
        * ELECTRON_GAMMA_HZ_PER_T
        * 1e-4,
        electron_t1_seconds=t1,
        electron_t2_seconds=t2,
        relaxation_matrix=published_relaxation(data) if relaxation == "full" else None,
        nitrogen_transition_per_second=0,
        exchange_coefficient_per_molar_second=exchange_scale
        / (p["exchange_time_seconds"] * p["concentration_molar"]),
    )
    saturation = model.steady_state(
        field_tesla=p["field_tesla"],
        concentration_molar=p["concentration_molar"],
        temperature_kelvin=p["temperature_kelvin"],
        nutation_rad_per_s=2
        * np.pi
        * ELECTRON_GAMMA_HZ_PER_T
        * p["b1_corotating_gauss"]
        * 1e-4
        * b1_scale,
        detuning_rad_per_s=-2 * np.pi * np.asarray(detuning_mhz) * 1e6,
    ).saturation_factor
    cw = overhauser_from_factors(
        field_tesla=p["field_tesla"],
        coupling_factor=p["coupling_factor"],
        leakage_factor=p["leakage_factor"],
        saturation_factor=saturation,
        t1_seconds=1
        / (
            1 / p["water_t10_seconds"]
            + p["nuclear_relaxivity_per_molar_second"] * p["concentration_molar"]
        ),
        temperature_kelvin=p["temperature_kelvin"],
    )
    return saturation, cw.enhancement - 1


def error_metrics(observed, predicted):
    """Report descriptive errors, without treating digitization as measurement SD."""
    error = np.asarray(predicted) - np.asarray(observed)
    return {
        "n": int(error.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def compare(data=None):
    """Return descriptive comparisons and qualified empirical acceptance status."""
    data = load_data() if data is None else data
    p = data["sezer_profile"]
    observed_e = digitized_points(p["figure5"])
    observed_s = digitized_points(p["figure6c"])
    _, predicted_e = profile_model(data, observed_e[:, 0])
    predicted_s, _ = profile_model(data, observed_s[:, 0])
    e_metrics = error_metrics(observed_e[:, 1], predicted_e)
    s_metrics = error_metrics(observed_s[:, 1], predicted_s)
    ablations = {}
    for mode in ("reduced", "line_dependent", "full"):
        ablations[mode] = {
            "enhancement": error_metrics(
                observed_e[:, 1],
                profile_model(data, observed_e[:, 0], relaxation=mode)[1],
            ),
            "saturation_theory": error_metrics(
                observed_s[:, 1],
                profile_model(data, observed_s[:, 0], relaxation=mode)[0],
            ),
        }
    full_blocks = published_relaxation(data)
    r1, u, v, r0 = full_blocks.longitudinal_blocks()
    # Fixed engineering envelopes for this explicit reduction, NOT statistical
    # confidence intervals or a statement that all data agree within uncertainty.
    envelopes = {"enhancement_rmse_max": 0.8, "saturation_theory_rmse_max": 0.08}
    within_envelopes = (
        e_metrics["rmse"] <= envelopes["enhancement_rmse_max"]
        and s_metrics["rmse"] <= envelopes["saturation_theory_rmse_max"]
    )
    product = data["epr_saturation_product"]
    predicted_product = (
        product["comparison_t1e_seconds"] * product["comparison_t2e_seconds"]
    )
    anchors = []
    for point in data["hofer_anchors"]["points"]:
        xi, saturation = point["coupling_factor"], point["saturation_factor"]
        reference = data["hofer_anchors"]
        radical_r1 = reference["two_wi_per_second"] / (1 - 7 * xi / 5)
        leakage = radical_r1 / (radical_r1 + reference["r10_per_second"])
        result = overhauser_from_factors(
            field_tesla=point["field_tesla"],
            coupling_factor=xi,
            leakage_factor=leakage,
            saturation_factor=saturation,
            t1_seconds=1 / (radical_r1 + reference["r10_per_second"]),
        )
        anchors.append(
            {
                "field_tesla": point["field_tesla"],
                "leakage_from_nmrd": leakage,
                "epsilon_factor_closure": float(result.enhancement - 1),
                "epsilon_approximate_reported": point["approximate_epsilon_max"],
                "independent_validation": False,
            }
        )
    return {
        "data_sha256": hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "fit_parameters": [],
        "active_model": "full_nitrogen_relaxation_and_transverse_exchange",
        "model_comparison": ablations,
        "full_relaxation_diagnostics": {
            "t2_diagonal_seconds_ascending_m": (
                1 / full_blocks.transverse_per_second.diagonal()
            ).tolist(),
            "collective_r1_rayleigh_per_second": float(np.mean(r1.sum(axis=1))),
            "nuclear_feedback_u_frobenius_per_second": float(np.linalg.norm(u)),
            "nuclear_feedback_v_frobenius_per_second": float(np.linalg.norm(v)),
            "nuclear_population_r0_rank": int(np.linalg.matrix_rank(r0, tol=1e-6)),
        },
        "enhancement_profile": e_metrics,
        "saturation_published_theory": s_metrics,
        "reduction_envelopes": envelopes,
        "within_reduction_envelopes": bool(within_envelopes),
        "epr_product_comparison": {
            "reported_seconds_squared": product["t1e_t2e_seconds_squared"],
            "other_study_parameter_product_seconds_squared": predicted_product,
            "ratio": predicted_product / product["t1e_t2e_seconds_squared"],
            "status": "cross_study_parameter_difference_not_statistical_rejection",
        },
        "hofer_factor_closure": anchors,
        "interpretation": "Partial published-data agreement: full tensor-derived relaxation improves saturation-theory RMSE but slightly worsens measured DNP RMSE relative to the retained legacy reduction. No DNP refitting. Saturation theory and DNP-derived saturation are not independent experimental saturation validation; cross-study relaxation inputs are not a universal calibration.",
    }


def saturation_product_curves(data, b1_gauss):
    """Reconstruct homogeneous-line fit curves; neither is raw saturation data."""
    p = data["epr_saturation_product"]
    omega = 2 * np.pi * ELECTRON_GAMMA_HZ_PER_T * np.asarray(b1_gauss) * 1e-4
    measured_fit = bloch_saturation(
        omega,
        p["t1e_t2e_seconds_squared"] / p["t2e_linewidth_seconds"],
        p["t2e_linewidth_seconds"],
    )
    other_parameters = bloch_saturation(
        omega, p["comparison_t1e_seconds"], p["comparison_t2e_seconds"]
    )
    return measured_fit, other_parameters


def verify_source_pdf(path, data):
    """Verify the source bytes and embedded images; requires optional PyMuPDF."""
    import fitz

    source = data["sources"]["sezer2009"]
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != source["pdf_sha256"]:
        raise ValueError("source PDF SHA256 mismatch")
    with fitz.open(path) as document:
        for figure in ("figure5", "figure6"):
            page = document[source["pdf_page_indices_zero_based"][figure]]
            digests = {
                hashlib.sha256(document.extract_image(item[0])["image"]).hexdigest()
                for item in page.get_images()
            }
            if source[figure + "_jpeg_sha256"] not in digests:
                raise ValueError(f"{figure} embedded-image SHA256 mismatch")
            if figure == "figure5":
                image_bytes = next(
                    document.extract_image(item[0])["image"]
                    for item in page.get_images()
                    if hashlib.sha256(
                        document.extract_image(item[0])["image"]
                    ).hexdigest()
                    == source[figure + "_jpeg_sha256"]
                )
                np.testing.assert_allclose(
                    extract_marker_centers(image_bytes),
                    data["sezer_profile"]["figure5"]["points_pixels"],
                    atol=0.01,
                    rtol=0,
                )


def extract_marker_centers(image_bytes):
    """Reproduce the red-square digitization from the source JPEG, without fitting.

    Optional Pillow/SciPy dependencies. Threshold red outlines, fill enclosed
    marker interiors, and merge fragments whose centers are <=4 native pixels
    apart. Components use 4-connectivity; cluster centers are area weighted.
    The frozen subset was visually checked against the source squares. Some
    squares have open/compressed outlines and are intentionally not recovered.
    """
    from io import BytesIO
    from PIL import Image
    from scipy.ndimage import binary_fill_holes, center_of_mass, label

    pixels = np.asarray(Image.open(BytesIO(image_bytes))).astype(float)
    red = (pixels[:, :, 0] > pixels[:, :, 1] + 35) & (
        pixels[:, :, 0] > pixels[:, :, 2] + 25
    )
    holes = binary_fill_holes(red) & ~red
    labels, count = label(holes)
    centers = np.array([center_of_mass(holes, labels, i) for i in range(1, count + 1)])[
        :, ::-1
    ]
    areas = np.array([(labels == i).sum() for i in range(1, count + 1)])
    groups = [{i} for i in range(count)]
    for i in range(count):
        for j in range(i):
            if np.linalg.norm(centers[i] - centers[j]) <= 4:
                a = next(group for group in groups if i in group)
                b = next(group for group in groups if j in group)
                if a is not b:
                    a.update(b)
                    groups.remove(b)
    output = [
        np.average(centers[list(group)], axis=0, weights=areas[list(group)])
        for group in groups
    ]
    return np.round(sorted(output, key=lambda point: point[0]), 2).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-pdf", type=Path)
    args = parser.parse_args()
    data = load_data()
    if args.source_pdf:
        verify_source_pdf(args.source_pdf, data)
    report = compare(data)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    if not report["within_reduction_envelopes"]:
        raise SystemExit("reduced model is outside the documented benchmark envelopes")


if __name__ == "__main__":
    main()
