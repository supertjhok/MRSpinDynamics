% CALC_MASY_TUNED_PROBE
% Calculate tuned-probe CPMG asymptotic and received magnetization.
%
% Signature
%   [mrx,masy,SNR,vsig] = calc_masy_tuned_probe(sp,pp)
%
% Inputs
%   sp - Tuned-probe system/simulation structure.
%   pp - Pulse-sequence structure.
%
% Outputs
%   mrx - Complex received magnetization spectrum.
%   masy - Complex asymptotic magnetization spectrum before receiver filtering.
%   SNR - Signal-to-noise ratio estimate scaled to voltage units.
%   vsig - Matched-filter signal amplitude returned by tuned_probe_rx.
%
% Dependencies
%   calc_masy_tuned_probe_lp.
%
% Notes
%   This public wrapper replaces an old generated placeholder and delegates to
%   the active low-power tuned-probe implementation used by the examples.
% -------------------------------------------------------------------------

function [mrx,masy,SNR,vsig] = calc_masy_tuned_probe(sp,pp)

[mrx,masy,SNR,vsig] = calc_masy_tuned_probe_lp(sp,pp);
end
