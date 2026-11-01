"""L-06 / L-56: package JAX entry points must not flip the global x64 flag.

The check runs in a fresh interpreter so imports done by other tests in the
same session cannot mask a module that still sets ``jax_enable_x64`` globally.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HAS_JAX = importlib.util.find_spec("jax") is not None

_SCRIPT = textwrap.dedent(
    """
    import json
    import numpy as np
    import jax

    flags = {}
    import spin_dynamics.workflows  # noqa: F401
    import spin_dynamics.optimal_control  # noqa: F401
    import spin_dynamics.optimization  # noqa: F401
    import spin_dynamics.nqr  # noqa: F401
    flags["after_imports"] = bool(jax.config.jax_enable_x64)

    from spin_dynamics.coupling.systems import coupled_spin_system
    from spin_dynamics.optimal_control._jax_propagation import propagate_state_numpy
    from spin_dynamics.optimal_control.hamiltonians import coupled_spin_control_model
    from spin_dynamics.optimal_control.objectives import make_grape_objective

    model = coupled_spin_control_model(coupled_spin_system([30.0], [[0.0]]))
    psi0 = np.array([1.0, 0.0], dtype=complex)
    target = np.array([0.0, 1.0], dtype=complex)
    n_segments, dt = 6, 2.0e-5
    objective = make_grape_objective(
        model, n_segments=n_segments, dt=dt, target=target, psi0=psi0,
        fixed_amplitude=2.0 * np.pi * 5.0e3,
    )
    phases = np.linspace(0.1, 1.2, n_segments)
    value, grad = objective(phases)
    flags["after_grape"] = bool(jax.config.jax_enable_x64)

    def numpy_fidelity(x):
        amp = np.full(n_segments, 2.0 * np.pi * 5.0e3)
        psi = propagate_state_numpy(model.h_drift, model.h_x, model.h_y, amp, x, dt, psi0)
        return float(abs(np.vdot(target, psi)) ** 2)

    eps = 1e-6
    fd = np.array([
        (numpy_fidelity(phases + eps * e) - numpy_fidelity(phases - eps * e)) / (2 * eps)
        for e in np.eye(n_segments)
    ])

    from spin_dynamics.nqr._jax_eigh import batched_eigh
    from spin_dynamics.nqr.hamiltonians import batched_nqr_hamiltonians
    from spin_dynamics.nqr.systems import QuadrupolarSite

    site = QuadrupolarSite(
        spin=1.0, quadrupole_frequency_hz=900.0e3, eta=0.3, gamma_hz_per_t=3.0766e6
    )
    fields = np.array([[0.0, 0.0, 1e-3], [2e-4, 1e-4, 0.0], [0.0, 3e-4, 1e-4]])
    hams = batched_nqr_hamiltonians(site, fields)
    jax_values, _vectors = batched_eigh(hams)
    numpy_values = np.linalg.eigvalsh(hams)
    flags["after_nqr"] = bool(jax.config.jax_enable_x64)
    print(json.dumps({
        "flags": flags,
        "value": float(value),
        "numpy_value": numpy_fidelity(phases),
        "grad_dtype": str(np.asarray(grad).dtype),
        "grad_err": float(np.max(np.abs(np.asarray(grad) - fd))),
        "grad_scale": float(np.max(np.abs(fd))),
        "eigh_dtype": str(jax_values.dtype),
        "eigh_rel_err": float(
            np.max(np.abs(jax_values - numpy_values)) / np.max(np.abs(numpy_values))
        ),
    }))
    """
)


@unittest.skipUnless(HAS_JAX, "jax not installed")
class GlobalX64FlagTests(unittest.TestCase):
    def test_package_jax_paths_leave_global_x64_disabled(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("JAX_ENABLE_X64", None)
        proc = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
            timeout=600,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(
            result["flags"],
            {"after_imports": False, "after_grape": False, "after_nqr": False},
        )
        # Results still in float64 and matching NumPy references.
        self.assertEqual(result["grad_dtype"], "float64")
        self.assertAlmostEqual(result["value"], result["numpy_value"], delta=1e-12)
        self.assertLess(result["grad_err"], 1e-7 * max(1.0, result["grad_scale"]))
        self.assertEqual(result["eigh_dtype"], "float64")
        self.assertLess(result["eigh_rel_err"], 1e-12)


if __name__ == "__main__":
    unittest.main()
