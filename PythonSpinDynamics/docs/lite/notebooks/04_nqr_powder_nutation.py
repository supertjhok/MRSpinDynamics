# %% [markdown]
# # Spin-1 NQR powder nutation
#
# This notebook runs PythonSpinDynamics in your browser (Pyodide). Edit any
# cell and press Shift+Enter to rerun it.
#
# In nuclear quadrupole resonance (NQR) there is no static field to align the
# crystallites, so the RF pulse drives each one with a different effective
# strength. For a spin-1 powder the echo after a pulse of nominal flip angle
# `theta` follows `sqrt(pi / (2 theta)) J_3/2(theta)`, with its maximum near
# 119 degrees rather than at 90.

# %%
import matplotlib.pyplot as plt
import numpy as np

from spin_dynamics.nqr import QuadrupolarSite, powder_average_grid, simulate_slse, slse_sequence

# %%
site = QuadrupolarSite(spin=1, isotope="14N", quadrupole_frequency_hz=900e3, eta=0.3)
orientations = powder_average_grid(12, 24)
nutation_hz = 10e3  # nominal RF nutation rate

angles = np.deg2rad(np.linspace(0.0, 720.0, 73))
signal = np.empty(angles.size, dtype=complex)
for index, angle in enumerate(angles):
    sequence = slse_sequence(
        "x",
        pulse_duration_seconds=angle / (2.0 * np.pi * nutation_hz),
        nutation_hz=nutation_hz,
        echo_spacing_seconds=1e-3,
        num_echoes=1,
    )
    signal[index] = simulate_slse(site, sequence, orientations=orientations).echo_amplitudes[0]

# %%
theta = np.where(angles == 0.0, 1e-12, angles)
reference = np.sin(theta) / theta**2 - np.cos(theta) / theta
magnitude = np.abs(signal)

plt.figure(figsize=(6, 4))
plt.plot(np.rad2deg(angles), magnitude / magnitude.max(), "o", label="simulated powder")
plt.plot(np.rad2deg(angles), np.abs(reference) / np.abs(reference).max(), label="|J3/2| envelope")
plt.axvline(119.0, color="0.5", linewidth=1)
plt.xlabel("Nominal flip angle (degrees)")
plt.ylabel("First-echo magnitude (normalized)")
plt.legend()
plt.show()
print(f"maximum at {np.rad2deg(angles[np.argmax(magnitude)]):.0f} degrees")

# %% [markdown]
# Try the `"y"` or `"z"` transition, or a different `eta`.
