% Calculate rotation axis and maximum SNR of a rectangular refocusing pulse
% Assume an untuned probe
% --------------------------------------------------------------

% function [masy,SNR] = plot_rot_rect_untuned(vars)
% vars: len = normalized refocusing pulse length, t_E = normalized echo period

function [neff,SNR] = plot_rot_rect_untuned(vars)

[params,sp,pp] = set_params_untuned_OCT; % Define default system parameters

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Adjust refocusing pulse length
% (relative to T_90)
params.tref = pp.T_90*vars.len;
params.tfp = ((vars.t_E)*pp.T_90/(pi/2)-params.tref)/2;

% Calculate refocusing axis
[neff,SNR]=calc_rot_axis_untuned_probe_lp(params,sp,pp);
