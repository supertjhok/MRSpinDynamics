% Plot results of refocusing pulse optimization for matched probes
% --------------------------------------------------------------
% params = [tref, pref, aref, tfp, tacq, Rs(off,on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [neff,SNR] = plot_opt_ref_results_matched(file,pulse_num)

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
    
% Read parameters	
results=results_all{pulse_num};
plot(pulse_num,results{4},'rs','MarkerSize',10);
sp=results{5}; pp=results{6};

% Create refocusing cycle
pp.tref=results{1}; pp.pref=results{2}; pp.aref=results{3};

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

% Calculate refocusing axis
[neff,SNR]=calc_rot_axis_matched_probe_Orig(sp,pp);
