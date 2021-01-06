% Calculate asymptotic SNR of a CPMG sequence assuming rectangular excitation and refocusing pulses
% Assume an untuned probe
% --------------------------------------------------------------

% function [masy,SNR] = plot_hardRef_tuned(vars)
% vars: len - normalized 90 pulse length, rat - ref/exc pulse ratio

function [masy,SNR] = plot_hardRef_untuned(vars)

[params,sp,pp] = set_params_untuned_Orig; % Define default system parameters

% Adjust excitation and refocusing pulse lengths 
params.texc = params.texc*vars.len;
params.tref = params.texc*vars.rat;

% Calculate asymptotic magnetization
[mrx,masy,SNR] = calc_masy_untuned_probe_lp(params,sp,pp);
