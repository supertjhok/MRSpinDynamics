"""Independent resolution and finite packet audit for Phase 3 records."""

from pathlib import Path
import json
import numpy as np
from acquisition import schedule, Receiver, lowpass


def waveform_error(a, b):
    mask = a["receive_mask"]
    reference = np.interp(a["time_s"], b["time_s"], b["signal_v"])
    return float(
        np.linalg.norm((a["signal_v"] - reference)[mask])
        / max(np.linalg.norm(reference[mask]), 1e-30)
    )


def packet(root, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    out = None
    for i, x in enumerate(nodes):
        for j, y in enumerate(nodes):
            for k, z in enumerate(nodes):
                r = schedule(root, position_m=(x * 0.004, y * 0.004, z * 0.01))
                if out is None:
                    out = dict(r)
                    out["signal_v"] = np.zeros_like(r["signal_v"])
                out["signal_v"] += (
                    r["signal_v"] * weights[i] * weights[j] * weights[k] / 8
                )
        print(f"packet order {order}: slab {i + 1}/{order} complete", flush=True)
    return out


def main():
    root = Path(".tmp/nqr_phase2_gate")
    output = Path(".tmp/nqr_phase3")
    output.mkdir(parents=True, exist_ok=True)
    coarse = schedule(root)
    fine = schedule(root, resolution=2)
    result = {
        "coupled_powder_disorder_pulse_and_acquisition_error": waveform_error(
            coarse, fine
        )
    }
    print(result, flush=True)
    p2 = packet(root, 2)
    print("8-voxel packet complete", flush=True)
    p3 = packet(root, 3)
    result["packet_8_vs_27_voxel_error"] = waveform_error(p2, p3)
    np.savez_compressed(
        output / "finite_packet.npz",
        time_s=p3["time_s"],
        signal_v=p3["signal_v"],
        receive_mask=p3["receive_mask"],
    )
    # Anti-alias timing test on the exact same waveform, separating receiver
    # discretization from spin-grid changes; compare causal continuous poles.
    cfg = Receiver()
    t = coarse["time_s"]
    high = schedule(root, fs_hz=2 * cfg.fs_hz)

    def filtered(r, fs):
        return lowpass(
            lowpass(
                lowpass(r["signal_v"], cfg.resonator_bandwidth_hz, fs),
                cfg.bandwidth_hz,
                fs,
            ),
            cfg.bandwidth_hz,
            fs,
        )

    a = filtered(coarse, cfg.fs_hz)
    b = np.interp(t, high["time_s"], filtered(high, 2 * cfg.fs_hz))
    mask = coarse["receive_mask"]
    result["receiver_200_vs_400_khz_relative_error"] = float(
        np.linalg.norm((a - b)[mask]) / np.linalg.norm(b[mask])
    )
    result["passed"] = bool(
        result["coupled_powder_disorder_pulse_and_acquisition_error"] < 0.05
        and result["packet_8_vs_27_voxel_error"] < 0.01
        and result["receiver_200_vs_400_khz_relative_error"] < 0.05
    )
    (output / "resolution_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
