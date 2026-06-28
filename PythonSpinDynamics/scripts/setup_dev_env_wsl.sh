#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${VENV:-.venv-wsl}"
EXTRAS="${EXTRAS:-dev,opt,plot,perf,bench}"
NO_VERIFY="${NO_VERIFY:-0}"
JAX_CUDA="${JAX_CUDA:-0}"

if [[ "$VENV_PATH" != /* ]]; then
  VENV_PATH="$ROOT/$VENV_PATH"
fi

cd "$ROOT"

echo "Creating or updating PythonSpinDynamics environment:"
echo "  root:   $ROOT"
echo "  venv:   $VENV_PATH"
echo "  python: $PYTHON_BIN"
echo "  extras: $EXTRAS"
echo "  jax cuda: $JAX_CUDA"

if [[ ! -d "$VENV_PATH" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install -e ".[${EXTRAS}]"

case "$JAX_CUDA" in
  0|false|False|no|No|"")
    VERIFY_FLAGS=(--strict)
    ;;
  12)
    "$VENV_PATH/bin/python" -m pip install --upgrade "jax[cuda12]"
    VERIFY_FLAGS=(--strict --require-jax-gpu)
    ;;
  13)
    "$VENV_PATH/bin/python" -m pip install --upgrade "jax[cuda13]"
    VERIFY_FLAGS=(--strict --require-jax-gpu)
    ;;
  *)
    echo "JAX_CUDA must be 0, 12, or 13; got '$JAX_CUDA'" >&2
    exit 2
    ;;
esac

if [[ "$NO_VERIFY" != "1" ]]; then
  "$VENV_PATH/bin/python" scripts/verify_dev_env.py "${VERIFY_FLAGS[@]}"
fi

echo
echo "Activate with:"
echo "  source \"$VENV_PATH/bin/activate\""
echo
echo "Run smoke checks with:"
echo "  python -m unittest tests.smoke_tests"
echo "  python -m ruff check src tests examples"
