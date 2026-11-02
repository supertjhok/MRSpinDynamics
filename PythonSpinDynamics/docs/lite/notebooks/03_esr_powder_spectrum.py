# %% [markdown]
# # ESR powder spectrum of an anisotropic g tensor
#
# This notebook runs PythonSpinDynamics in your browser (Pyodide). Edit any
# cell and press Shift+Enter to rerun it.
#
# A spin-1/2 with principal g values `gx < gy < gz` resonates at a field that
# depends on its orientation, `B = h nu / (g_eff mu_B)`. A single crystal
# gives one line per orientation; a powder averages over all of them and
# shows edges at `gz` and `gx` and a peak at `gy`.

# %%
import matplotlib.pyplot as plt
import numpy as np

from spin_dynamics.esr import ESRSpinSystem, simulate_field_sweep, single_crystal_orientation

# %%
system = ESRSpinSystem(g_tensor=[2.00, 2.08, 2.24])
frequency = 9.5e9  # X band (Hz)

# %% [markdown]
# Single-crystal lines along the three principal axes (`beta` is the polar
# angle of B0 from the g tensor's z axis, `alpha` the azimuth).

# %%
axes_to_try = {"x": (0.0, np.pi / 2), "y": (np.pi / 2, np.pi / 2), "z": (0.0, 0.0)}
for name, (alpha, beta) in axes_to_try.items():
    crystal = simulate_field_sweep(
        system, frequency, orientations=single_crystal_orientation(alpha, beta)
    )
    print(f"B0 along {name}: resonance at {1e3 * crystal.lines[0].field_tesla:.2f} mT")

# %% [markdown]
# Powder pattern. `orientations="powder"` sizes the orientation grid from the
# g anisotropy and the linewidth; CW ESR is usually recorded as the first
# derivative.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for mode, axis in zip(("absorption", "derivative"), axes):
    powder = simulate_field_sweep(
        system,
        frequency,
        orientations="powder",
        broadening_tesla=0.4e-3,
        detection_mode=mode,
    )
    axis.plot(1e3 * powder.fields_tesla, powder.spectrum)
    axis.set_xlabel("B0 (mT)")
    axis.set_title(f"Powder {mode}")
print(f"{len(powder.lines)} orientation-resolved lines")
plt.show()

# %% [markdown]
# Try an axial tensor such as `[2.05, 2.05, 2.30]`, or a narrower linewidth.
