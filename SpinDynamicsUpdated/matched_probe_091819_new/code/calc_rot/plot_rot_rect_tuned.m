% Calculate rotation axis and maximum SNR of a rectangular refocusing pulse
% Assume a tuned probe
% --------------------------------------------------------------

% function [masy,SNR] = plot_rot_rect_tuned(vars)
% vars: len - normalized refocusing pulse length

function [neff,SNR] = plot_rot_rect_tuned(vars)

[params,sp,pp] = set_params_tuned_OCT; % Define default system parameters

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Adjust refocusing pulse length
% (relative to T_90)
params.tref = pp.T_90*vars.len;
params.tfp = ((vars.t_E)*pp.T_90/(pi/2)-params.tref)/2;

% Calculate refocusing axis
[neff,SNR]=calc_rot_axis_tuned_probe_lp_Orig(params,sp,pp);
