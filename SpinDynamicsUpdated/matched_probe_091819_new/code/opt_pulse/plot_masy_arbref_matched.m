% Calculate asymptotic SNR of a CPMG sequence assuming arbitrary 
% excitation and refocusing pulses
% Assume a matched probe
% --------------------------------------------------------------

% function [masy,SNR] = plot_masy_arbref_tuned(vars)
% params: 
% aexc, texc: normalized excitation pulse amplitude and length
% pref: refocusing pulse phase vector, delt: length of each segment (units of T_180)

function [mrx,masy,echo_rx,tvect,SNR] = plot_masy_arbref_matched(sp,pp)

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Create refocusing pulse
pp.aref = ones(1,length(pp.pref)); % Constant amplitude
pp.tref = pp.T_180*pp.delt*ones(1,length(pp.pref)); % Fixed segment length

% Calculate asymptotic magnetization
[mrx,masy,SNR] = calc_masy_matched_probe_Orig(sp,pp);

% Plot time-domain echo
[echo_rx,tvect]=calc_time_domain_echo(mrx,sp.del_w,1,1);
