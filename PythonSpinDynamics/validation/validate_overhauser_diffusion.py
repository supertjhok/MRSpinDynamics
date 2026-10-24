"""Independently integrate Hwang-Freed Eq. A6 to audit CW ODNP rates.

Run from PythonSpinDynamics with the optional SciPy dependency installed:
python validation/validate_overhauser_diffusion.py [--write-reference]
The reference contains theoretical values, not digitized experimental data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.dnp import (  # noqa: E402
    TranslationalDiffusionModel,
    ffhs_spectral_density,
    translational_relaxation_rates,
)


def diffusion_integral(omega: float, tau: float) -> float:
    """Integrate the diffusion eigenmodes; normalized one-sided J in seconds."""
    u = abs(omega) * tau

    def integrand(x):
        frequency = 1 if u == 0 else x**4 / (x**4 + u**2)
        return frequency / (x**6 - 2 * x**4 + 9 * x**2 + 81)

    value, _ = quad(integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-11, limit=300)
    return 54 * tau * value / np.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    rows = []
    for tau in [30e-12, 54e-12, 100e-12, 300e-12]:
        for field in [0.1, 0.35, 1.0]:
            we, wn = 2 * np.pi * field * np.array([28.0249514242e9, 42.57747892e6])
            jm, ji, jp = [diffusion_integral(w, tau) for w in [we - wn, wn, we + wn]]
            xi = (6 * jm - jp) / (6 * jm + 3 * ji + jp)
            model = TranslationalDiffusionModel(3e-10, 9e-20 / tau)
            calculated = translational_relaxation_rates(field, model, 0.001)
            np.testing.assert_allclose(calculated.coupling_factor, xi, atol=2e-10)
            np.testing.assert_allclose(
                model.spectral_density([we - wn, wn, we + wn]), [jm, ji, jp], rtol=2e-10
            )
            rows.append(
                dict(field_tesla=field, tau_d_seconds=tau, coupling_factor=float(xi))
            )
    # Table I gives 1/2 J_two_sided(0)*d*D/n=4/27; here J is normalized
    # by C(0)=n/(3*d^3), so J_one_sided(0)/tau=4/9.
    np.testing.assert_allclose(ffhs_spectral_density(0, 1), 4 / 9, rtol=1e-15)
    payload = {
        "evidence_type": "published_theory_and_independent_quadrature",
        "doi": "10.1063/1.431841",
        "source_url": "https://www.acert.cornell.edu/PDFs/JChemPhys63_4017_1975.pdf",
        "source_location": "Appendix A Eq. A6; Table I analytic zero-frequency entry",
        "method": "SciPy adaptive quadrature of the diffusion eigenmode integral; not the production rational expression",
        "normalized_one_sided_j0_over_tau": 4 / 9,
        "table_i_half_two_sided_j0_dD_over_density": 4 / 27,
        "gamma_e_hz_per_t": 28.0249514242e9,
        "gamma_n_hz_per_t": 42.57747892e6,
        "transition_convention": "Positive magnitudes; H_Z=+omega_e*S_z-omega_n*I_z; dipolar DQ at difference, ZQ at sum",
        "diffusion_curve": rows,
        "limitations": [
            "Theory reference only; no independently measured enhancement-vs-power dataset is asserted.",
            "The original paper's density/radial and two-sided conventions are converted explicitly.",
            "Absolute rate prefactor is independently checked by spatial and Poisson integrals in tests/test_dnp_spectral.py.",
        ],
    }
    if args.write_reference:
        destination = ROOT / "validation/experimental/overhauser_ffhs_reference.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "validated_conditions": len(rows),
                "xi_at_54ps_035T": rows[4]["coupling_factor"],
            }
        )
    )


if __name__ == "__main__":
    main()
