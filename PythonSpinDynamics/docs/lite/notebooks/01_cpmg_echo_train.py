# %% [markdown]
# # CPMG echo train
#
# This notebook runs PythonSpinDynamics in your browser (Pyodide); nothing is
# installed on your computer. Edit any cell and press Shift+Enter to rerun it.
#
# A CPMG sequence is a 90-degree excitation followed by a train of refocusing
# pulses spaced by the echo time TE. Each echo is the sum of isochromats over a
# range of frequency offsets; with imperfect refocusing the first few echoes
# differ before the train settles to its asymptotic shape.

# %%
import matplotlib.pyplot as plt
import numpy as np

from spin_dynamics import conventions
from spin_dynamics.experiment import Acquisition, CPMGTrain, Experiment, Hardware, Sample

# %% [markdown]
# Describe the experiment declaratively. `plan()` resolves the workflow and
# checks the settings before anything runs.

# %%
T2 = 5e-3  # transverse relaxation time (s)

study = Experiment(
    sequence=CPMGTrain(num_echoes=32),
    sample=Sample(t1_seconds=T2, t2_seconds=T2),
    hardware=Hardware(probe="ideal"),
    acquisition=Acquisition(numpts=1801, maxoffs=10.0),
)
print(study.plan().report())

# %%
result = study.run().result
echo_times = result.sequence_time  # echo centres (s)
print(f"echo spacing TE = {1e6 * (echo_times[1] - echo_times[0]):.0f} us")

# %% [markdown]
# Echo shapes: the first echo differs from the later ones, which converge to
# the asymptotic CPMG echo.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for index in (0, 1, 2, 15):
    axes[0].plot(result.tvect, np.abs(result.echo[index]), label=f"echo {index + 1}")
axes[0].set_xlabel("Normalized time")
axes[0].set_ylabel("|echo|")
axes[0].set_title("Echo shapes")
axes[0].legend()

integrals = np.abs(result.echo_integrals)
axes[1].plot(1e3 * echo_times, integrals / integrals[0], "o", label="simulated")
axes[1].plot(1e3 * echo_times, np.exp(-(echo_times - echo_times[0]) / T2), label="exp(-t / T2)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Echo integral (normalized)")
axes[1].set_title("Echo-train decay")
axes[1].legend()
plt.show()

# %% [markdown]
# The simulated train decays a little more slowly than `exp(-t / T2)`. By
# default (MATLAB parity) spins do not relax during the refocusing pulses, so
# each echo period relaxes for `TE - T_180` only. The `"physics"` conventions
# profile relaxes during pulses as well:

# %%
with conventions.conventions("physics"):
    physical = study.run().result
physical_integrals = np.abs(physical.echo_integrals)

plt.figure(figsize=(5, 3.5))
plt.semilogy(1e3 * echo_times, integrals / integrals[0], "o", label="matlab (default)")
plt.semilogy(1e3 * echo_times, physical_integrals / physical_integrals[0], "s", label="physics")
plt.semilogy(1e3 * echo_times, np.exp(-(echo_times - echo_times[0]) / T2), label="exp(-t / T2)")
plt.xlabel("Time (ms)")
plt.ylabel("Echo integral (normalized)")
plt.legend()
plt.show()

# %% [markdown]
# Try `Hardware(probe="tuned")` to include a tuned receive probe, or change
# `T2` and `num_echoes`.
