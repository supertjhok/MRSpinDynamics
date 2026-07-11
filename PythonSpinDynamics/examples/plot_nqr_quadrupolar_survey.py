r"""A zero-field NQR line atlas across quadrupolar nuclei (spin 3/2 to 9/2).

With the isotope presets a site is one call away for any of the common quadrupolar
nuclei, so this example surveys a spread of them -- spin-3/2 (35Cl, 63Cu), spin-5/2
(27Al, 55Mn), spin-7/2 (51V, 59Co) and spin-9/2 (93Nb, 209Bi) -- and, for each,
uses ``diagonalize_site`` to lay out its zero-field NQR lines and their relative
intensities. A half-integer spin ``I`` has ``I - 1/2`` lines (at nu_Q, 2 nu_Q, ...
for an axial EFG), and the fundamental (lowest) line carries the largest transverse
matrix element -- which grows with spin -- so the higher-spin nuclei are not just
accessible but have strong fundamental lines.

Panels:

1. Line atlas: every nucleus's NQR lines on a log frequency axis, marker area set
   by the transverse matrix element (the detection strength). Shows where the lines
   sit and how many there are.
2. Fundamental-line matrix element vs. spin: the ``1/2 <-> 3/2`` transverse matrix
   element per nucleus.

The fundamental frequencies here are representative order-of-magnitude values for
well-known NQR compounds; pass your own with the presets
(``quadrupolar_site(isotope, nu_q_hz=... | cq_hz=..., eta=...)``). Run with
``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


# (isotope, representative fundamental line nu_1 in MHz, representative eta).
_SURVEY = [
    ("35Cl", 30.0, 0.0),
    ("63Cu", 26.0, 0.0),
    ("27Al", 0.36, 0.03),
    ("55Mn", 2.4, 0.05),
    ("51V", 0.66, 0.05),
    ("59Co", 4.0, 0.10),
    ("93Nb", 2.0, 0.10),
    ("209Bi", 6.5, 0.10),
]


# Keep CLI choices together so scientific defaults are easy to find and override.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Survey the zero-field NQR lines and intensities of common quadrupolar "
            "nuclei (spin 3/2 to 9/2) built from the isotope presets."
        )
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.nqr import diagonalize_site, quadrupolar_site

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True)

    spin_colors = {1.5: "C0", 2.5: "C2", 3.5: "C3", 4.5: "C4"}
    fundamental = []  # (label, spin, matrix element)
    seen_spins: set[float] = set()

    for row, (iso, nu1_mhz, eta) in enumerate(_SURVEY):
        site = quadrupolar_site(iso, nu_q_hz=nu1_mhz * 1e6, eta=eta)
        spin = site.spin
        color = spin_colors[spin]
        transitions = diagonalize_site(site).transitions
        # collapse degenerate partners: strongest per distinct line frequency.
        lines: dict[int, float] = {}
        for t in transitions:
            key = int(round(t.frequency_hz / (0.02 * nu1_mhz * 1e6)))
            lines[key] = max(lines.get(key, 0.0), t.strength)
        for key, strength in lines.items():
            freq_mhz = key * 0.02 * nu1_mhz
            label = f"I={spin:g}" if spin not in seen_spins else None
            ax[0].scatter(freq_mhz, row, s=25 + 90 * strength / 3.5, color=color,
                          edgecolor="k", lw=0.4, label=label, zorder=3)
            seen_spins.add(spin)
        # fundamental line = lowest-frequency line (smallest key); its strength.
        fundamental.append((iso, spin, lines[min(lines)]))

    ax[0].set_yticks(range(len(_SURVEY)))
    ax[0].set_yticklabels([f"{iso}" for iso, *_ in _SURVEY])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("NQR line frequency (MHz)")
    ax[0].set_title("Zero-field NQR line atlas (marker area ~ matrix element)")
    ax[0].grid(True, which="both", axis="x", ls=":", lw=0.5, alpha=0.6)
    ax[0].legend(fontsize=8, title="nuclear spin", loc="lower right")

    # (b) Fundamental-line transverse matrix element per nucleus (grows with spin).
    labels = [iso for iso, *_ in _SURVEY]
    strengths = [f[2] for f in fundamental]
    colors = [spin_colors[f[1]] for f in fundamental]
    ax[1].bar(range(len(labels)), strengths, color=colors, edgecolor="k", lw=0.5)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, rotation=45, ha="right")
    ax[1].set_ylabel("fundamental-line matrix element")
    ax[1].set_title("Fundamental (1/2<->3/2) detection strength")
    for i, (iso, spin, strength) in enumerate(fundamental):
        ax[1].text(i, strength + 0.05, f"{spin:g}", ha="center", fontsize=8)

    fig.suptitle("Quadrupolar-nucleus NQR survey (representative frequencies)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
