% Calculate asymptotic SNR of a CPMG sequence assuming rectangular excitation and refocusing pulses
% Assume a tuned probe
% --------------------------------------------------------------

% function [masy,SNR] = plot_hardRef_tuned(vars)
% vars: len - normalized 90 pulse length, rat - ref/exc pulse ratio

function [masy,SNR] = plot_hardRef_tuned(vars)

[params,sp,pp] = set_params_tuned_Orig; % Define default system parameters

% Adjust excitation and refocusing pulse lengths 
params.texc = params.texc*vars.len;
params.tref = params.texc*vars.rat;

% Calculate asymptotic magnetization
[mrx,masy,SNR] = calc_masy_tuned_probe_lp_Orig(params,sp,pp);
