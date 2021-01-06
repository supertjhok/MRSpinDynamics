% Plot results of refocusing pulse optimization for tuned probes
% --------------------------------------------------------------
% params = [tref, pref, aref, tfp, tacq, Rs(off,on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [neff,SNR] = plot_opt_ref_results_tuned(file,pulse_num)

% Load the results file
filname = file;

tmp=load(filname); results_all=tmp.results;
siz = size(results_all);

% Plot optimization result (axis_rms) of all pulses for comparison
axis_rms=zeros(1,siz(1));
for i=1:siz(1)
tmp=results_all{i}; axis_rms(i)=tmp{4};
end
figure; plot(axis_rms,'LineWidth',1); hold on;
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
ylabel('Optimized SNR (rms)');
    
results=results_all{pulse_num};
plot(pulse_num,results{4},'rs','MarkerSize',10);
params=results{5};
sp=results{6}; pp=results{7};

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Calculate refocusing axis
[neff,SNR]=calc_rot_axis_tuned_probe_lp_Orig(params,sp,pp);
