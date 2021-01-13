% Calculate rotation axis and maximum SNR of a rectangular refocusing pulse
% Assume a matched probe
% --------------------------------------------------------------

% function [masy,SNR] = plot_rot_rect_matched(vars)
% vars: len - normalized refocusing pulse length, t_E - normalized echo
% period

function [neff,SNR] = plot_rot_rect_matched(vars)

[sp,pp] = set_params_matched_OCT; % Define default system parameters

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Adjust refocusing pulse length
% (relative to T_90)
pp.tref(2) = pp.T_90*vars.len;

% Set free-precession interval
pp.tref(1) = ((vars.t_E)*pp.T_90/(pi/2)-pp.tref(2))/2; pp.tref(3) = pp.tref(1);

% Design matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Find receiver transfer functions
[tf1,tf2] = matched_receiver_tf(sp,pp);
sp.tf1=tf1; sp.tf2=tf2;

% Calculate refocusing axis
[neff,SNR]=calc_rot_axis_matched_probe_Orig(sp,pp);
