% Plot results of refocusing pulse excitation for tuned probes
% --------------------------------------------------------------
% params = [texc, pexc, aexc, tref, pref, aref, tfp, tacq, Rs(off,on), pcycle]
% (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [mrx,masy,echo_rx,tvect,SNR] = plot_opt_exc_results_tuned(filname,pulse_num)

% Load the results
tmp=load(filname); results_all=tmp.results;
siz = size(results_all);

% Plot optimization result (axis_rms) of all pulses for comparison
axis_rms=zeros(1,siz(1));
for i=1:siz(1)
	tmp=results_all{i}; axis_rms(i)=tmp{7};
end
figure; plot(axis_rms,'LineWidth',1); hold on;
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
ylabel('Optimized SNR (rms)');

results=results_all{pulse_num};
plot(pulse_num,results{7},'rs','MarkerSize',10);
params=results{8};
sp=results{9}; pp=results{10};
params.pcycle=0;

% Quantize pulse phases
% pp.NumPhases = 4096;
% [phiQ_ref]=quantize_phase(params.pexc,sp,pp);
% params.pexc = phiQ_ref;

% Calculate asymptotic magnetization
[mrx,masy,SNR] = calc_masy_tuned_probe_lp_Orig(params,sp,pp);

% Plot time-domain echo
[echo_rx,tvect]=calc_time_domain_echo(mrx,sp.del_w,1,1);