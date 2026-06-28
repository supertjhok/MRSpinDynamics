"""Report whether the active Python environment is ready for development."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PackageCheck:
    module: str
    distribution: str
    purpose: str
    recommended: bool = True


CHECKS = (
    PackageCheck("spin_dynamics", "python-spin-dynamics", "editable package"),
    PackageCheck("numpy", "numpy", "core arrays"),
    PackageCheck("scipy", "scipy", "optimization and inverse Laplace"),
    PackageCheck("matplotlib", "matplotlib", "plotting examples"),
    PackageCheck("PIL", "pillow", "image examples"),
    PackageCheck("numba", "numba", "CPU JIT acceleration"),
    PackageCheck("jax", "jax", "JAX CPU/GPU acceleration"),
    PackageCheck("jaxlib", "jaxlib", "JAX compiled runtime"),
    PackageCheck("ruff", "ruff", "lint checks"),
    PackageCheck("pytest", "pytest", "benchmark/test tooling"),
    PackageCheck("pytest_benchmark", "pytest-benchmark", "benchmark tooling"),
)


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _find_module(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def _jax_devices() -> list[str]:
    try:
        jax = importlib.import_module("jax")
        return [str(device) for device in jax.devices()]
    except Exception as exc:
        return [f"unavailable ({exc})"]


def _has_jax_gpu(devices: list[str]) -> bool:
    return any("cuda" in device.lower() or "gpu" in device.lower() for device in devices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the PythonSpinDynamics development environment."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero when recommended development packages are missing",
    )
    parser.add_argument(
        "--require-jax-gpu",
        action="store_true",
        help="return nonzero unless JAX reports a CUDA/GPU device",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    print("PythonSpinDynamics development environment")
    print(f"  project root: {root}")
    print(f"  executable:   {sys.executable}")
    print(f"  python:       {platform.python_version()} ({platform.system()})")
    print()

    missing: list[str] = []
    for check in CHECKS:
        ok = _find_module(check.module)
        status = "ok" if ok else "missing"
        version = _version(check.distribution) if ok else "-"
        print(
            f"  {check.distribution:22s} {status:7s} "
            f"{version:12s} {check.purpose}"
        )
        if check.recommended and not ok:
            missing.append(check.distribution)

    devices = _jax_devices()
    print()
    print(f"  jax devices: {', '.join(devices)}")

    if args.strict and missing:
        print()
        print("Missing recommended packages:")
        for name in missing:
            print(f"  - {name}")
        print()
        print("Recreate or update the environment with one of:")
        print("  powershell -ExecutionPolicy Bypass -File scripts/setup_dev_env.ps1")
        print("  bash scripts/setup_dev_env_wsl.sh")
        return 1

    if args.require_jax_gpu and not _has_jax_gpu(devices):
        print()
        print("JAX did not report a CUDA/GPU device.")
        print("For WSL with a supported NVIDIA driver, recreate or update with:")
        print("  JAX_CUDA=13 bash scripts/setup_dev_env_wsl.sh")
        print("Use JAX_CUDA=12 only when the driver/toolkit constraints require it.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
