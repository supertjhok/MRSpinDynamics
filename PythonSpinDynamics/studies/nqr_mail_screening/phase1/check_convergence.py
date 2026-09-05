"""Run independent resolution refinements; does not certify scanner accuracy."""

import json
from dataclasses import replace
from pathlib import Path
from pulsed_study import Config, simulate
from geometry import for_config

c = Config(echoes=3)
a = simulate(c)[0]
checks = {}
for name, cfg in (
    ("powder", replace(c, powder_theta=8, powder_phi=16)),
    ("offsets", replace(c, offset_points=81)),
    ("time", replace(c, dt_s=10e-6)),
    ("rf_envelope", replace(c, pulse_steps=32)),
):
    b = simulate(cfg)[0]
    checks[name] = abs(b["single_shot_snr_1g"] / a["single_shot_snr_1g"] - 1)
    print(name, checks[name], flush=True)
g0, g1 = for_config(c), for_config(c, 2)
checks["zeeman_geometry"] = abs(
    g1["maximum_line_shift_hz"] / g0["maximum_line_shift_hz"] - 1
)
print(checks, flush=True)
Path(".tmp/nqr_pulsed/convergence.json").write_text(json.dumps(checks, indent=2) + "\n")
if max(checks.values()) > 0.03:
    raise SystemExit("Resolution check exceeds 3%; refine before interpreting")
