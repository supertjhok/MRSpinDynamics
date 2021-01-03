% Calculate asymptotic magnetization of CPMG including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% params = [texc, pexc, aexc, tref, pref, aref, tfp, tacq, Rs(off,on), pcycle]
% (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [mrx,SNR,SNR_max] = plot_opt_exc_results_tuned_probe_lp(filname,pulse_num)

% Load the results
% switch results_num
%     case 1 % fscale = 1
%         filname=['results_circsim_lp_1_14_run1.mat']; % Normal, 80 segments
%         %filname=['results_circsim_lp_1_14_run1_32_inv1.mat']; % Inverted, 80 segments
%     case 2 % fscale = 2
%         filname=['results_circsim_lp_1_13_run1.mat']; % Normal, 80 segments
%         %filname=['results_circsim_lp_1_13_run1_35_inv1_reopt.mat']; % Inverted, 80 segments
%     case 3 % fscale = 1, 1:8 tap on transmitter
%         filname=['results_circsim_lp_1_8.mat']; % Normal, 80 segments
% end


tmp=load(filname); results=tmp.results;
params=results{pulse_num,8};
sp=results{pulse_num,9}; pp=results{pulse_num,10};
params.pcycle=0;
params.trd = pi;

% pp.NumPhases = 4096;
% [phiQ_ref]=quantize_phase(params.pexc,sp,pp);
% params.pexc = phiQ_ref;
% 
% sp.w0n=2*pi*0.257e6; 
% sp.w0=sp.w0n;
% pp.w=2*pi*0.257e6;
% sp.C =  1/(sp.L*(2*pi*0.25*1e6)^2);
% sp.cycle = 1/sp.w0;
% pp.T_90 = pp.T_90;
% pp.T_180 = pp.T_180;

% params.texc = pi/2; 
% params.pexc = pi/2;
% params.aexc = 1;

[mrx,SNR,SNR_max] = calc_masy_tuned_probe_lp(params,sp,pp);