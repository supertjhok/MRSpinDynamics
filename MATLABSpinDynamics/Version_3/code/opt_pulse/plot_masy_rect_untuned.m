% Calculate asymptotic SNR of a CPMG sequence assuming rectangular excitation and refocusing pulses
% Assume an untuned probe
% --------------------------------------------------------------

% function [masy,SNR] = plot_masy_rect_untuned(vars)
% vars: len - normalized 90 pulse length, rat - ref/exc pulse ratio

function [mrx,masy,echo_rx,tvect,SNR] = plot_masy_rect_untuned(vars)

[params,sp,pp] = set_params_untuned_OCT; % Define default system parameters

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Adjust excitation and refocusing pulse lengths
% (relative to T_90)
params.texc = pp.T_90*vars.len;
params.tref = params.texc*vars.rat;

% Calculate asymptotic magnetization
[mrx,masy,SNR] = calc_masy_untuned_probe_lp(params,sp,pp);

% Plot time-domain echo
[echo_rx,tvect]=calc_time_domain_echo(mrx,sp.del_w,1,1);