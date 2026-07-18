"""Compatibility alias for the Mandal 2015 phase-step sweep example.

The original phase-class plot was misleading because a fixed refocusing-pulse
absolute phase creates a static pulse error rather than the echo-to-echo
modulation of interest.  Run this file to execute
``plot_mandal2015_phase_step_sweep.py``.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from plot_mandal2015_phase_step_sweep import main


if __name__ == "__main__":
    main()
