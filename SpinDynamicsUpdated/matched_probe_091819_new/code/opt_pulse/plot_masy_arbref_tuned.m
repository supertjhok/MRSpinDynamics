% Calculate asymptotic SNR of a CPMG sequence assuming arbitrary 
% excitation and refocusing pulses
% Assume a tuned probe
% --------------------------------------------------------------

% function [masy,SNR] = plot_masy_arbref_tuned(vars)
% params: 
% aexc, texc: normalized excitation pulse amplitude and length
% pref: refocusing pulse phase vector, delt: length of each segment (units of T_180)

function [mrx,masy,echo_rx,tvect,SNR] = plot_masy_arbref_tuned(params,sp,pp)

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Create refocusing pulse
params.aref = ones(1,length(params.pref)); % Constant amplitude
params.tref = pp.T_180*params.delt*ones(1,length(params.pref)); % Fixed segment length

% Calculate asymptotic magnetization
[mrx,masy,SNR] = calc_masy_tuned_probe_lp_Orig(params,sp,pp);

% Plot time-domain echo
[echo_rx,tvect]=calc_time_domain_echo(mrx,sp.del_w,1,1);
