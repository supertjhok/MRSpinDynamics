% Calculate rotation axis of a refocusing cycle including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% params = [tref, pref, aref, tfp, tacq, Rs(off,on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [n,SNR] = plot_opt_ref_results_tuned_probe_lp(file,pulse_num)

filname = file;
% Load the refocusing pulse
% switch results_num
%     % No symmetry constraints
%     case 1 % fscale = 1, 1:8 tap on transmitter
%         filname=['results_ref_pulse_circsim_lp_1_16_run1']; % 16 segments, length = 1*T_80
% end

tmp=load(filname); results=tmp.results;
params=results{pulse_num,5};
sp=results{pulse_num,6}; pp=results{pulse_num,7};
% params.Rs = [5 10^6 20];
params.Rs = [5 10^6 5];
params.tref = params.tref;
% params.pref = params.pref*2+pi/4;
% params.pref = [0 0 0 0 0 0 0 0 pi/2 pi/2 pi/2 pi/2 pi/2 pi/2 pi/2 0 0 0 0 0 0 0 0];


sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Calculate refocusing axis
[n,SNR]=calc_rot_axis_tuned_probe_lp(params,sp,pp);