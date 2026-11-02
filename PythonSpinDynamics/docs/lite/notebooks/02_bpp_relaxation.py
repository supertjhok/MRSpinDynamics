# %% [markdown]
# # BPP relaxation versus temperature
#
# This notebook runs PythonSpinDynamics in your browser (Pyodide). Edit any
# cell and press Shift+Enter to rerun it.
#
# In the Bloembergen-Purcell-Pound (BPP) model, molecular tumbling with
# correlation time `tau_c` sets the spectral density `J(w)` that drives
# relaxation. `T1` has a minimum where `w0 tau_c` is about 0.6; `T2` keeps
# falling as motion slows. With an Arrhenius `tau_c`, cooling the sample
# sweeps through that minimum.

# %%
import matplotlib.pyplot as plt
import numpy as np

from spin_dynamics.relaxation import BPPRelaxationModel

# %%
temperatures = np.linspace(220.0, 460.0, 300)  # K


def bpp_rates(larmor_hz):
    model = BPPRelaxationModel(
        angular_frequency_rad_per_s=2.0 * np.pi * larmor_hz,
        tau_ref_seconds=8e-9,  # tau_c at the reference temperature
        reference_temperature_kelvin=300.0,
        activation_energy_j_per_mol=16e3,
        coupling_scale_per_second2=5e8,
    )
    return model.rates(temperatures)


# %% [markdown]
# Compare two Larmor frequencies: the `T1` minimum moves to higher
# temperature (shorter `tau_c`) at the higher field.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for larmor_mhz in (20.0, 60.0):
    rates = bpp_rates(larmor_mhz * 1e6)
    axes[0].semilogy(temperatures, rates.t1_seconds, label=f"T1, {larmor_mhz:g} MHz")
    axes[0].semilogy(temperatures, rates.t2_seconds, "--", label=f"T2, {larmor_mhz:g} MHz")
    minimum = int(np.argmin(rates.t1_seconds))
    print(
        f"{larmor_mhz:g} MHz: T1 minimum {1e3 * rates.t1_seconds[minimum]:.1f} ms "
        f"at {temperatures[minimum]:.0f} K"
    )
axes[0].set_xlabel("Temperature (K)")
axes[0].set_ylabel("Relaxation time (s)")
axes[0].legend()

axes[1].semilogy(temperatures, 1e9 * rates.correlation_time_seconds)
axes[1].set_xlabel("Temperature (K)")
axes[1].set_ylabel("tau_c (ns)")
axes[1].set_title("Arrhenius correlation time")
plt.show()

# %% [markdown]
# Try a larger activation energy, or add `baseline_r1_per_second` and
# `baseline_r2_per_second` to the model for other relaxation pathways.
