% Summarize OCT simulation results and generate plots
% Written by: Soumyajit Mandal, 01/07/21

numiter = 24; % Number of optimization runs

% -------------------------------------------------------------------
% Refocusing pulse results (untuned, tuned, and matched probes)
% -------------------------------------------------------------------

% Set rectangular refocusing pulse parameters
rvars.len = 1.6; % Units of T_90
rvars.t_E = 7*pi; % Units of T_90

% Compute maximum SNR of rectangular refocusing pulses as a reference
[neff_ru,SNR_ru] = plot_rot_rect_untuned(rvars); % Untuned probe
[neff_rt,SNR_rt] = plot_rot_rect_tuned(rvars); % Tuned probe
[neff_rm,SNR_rm] = plot_rot_rect_matched(rvars); % Matched probe
close all; % Don't neeed these plots

% List of refocusing pulse result files
rlen = {'1','1p5','2'}; nres = length(rlen);
rlen_val = [1,1.5,2]; % Refocusing pulse lengths (normalized to T_180)

% Read refocusing pulse data
SNR_u = zeros(nres,numiter); SNR_t = SNR_u; SNR_m = SNR_u;
SNR_umax = zeros(1,nres); SNR_tmax = SNR_umax; SNR_mmax = SNR_umax;
for i = 1:nres
    rfil_u = ['ref_untuned_500k_' rlen{i}]; % Untuned probe
    rfil_t = ['ref_tuned_500k_' rlen{i}]; % Tuned probe
    rfil_m = ['ref_matched_500k_' rlen{i}]; % Matched probe
    
    pulse_nums = round(numiter*rand(1,3)); % Random pulse for initial plotting
    [neff_u,SNR_u(i,:)] = plot_opt_ref_results_untuned(rfil_u,pulse_nums(1));
    [neff_t,SNR_t(i,:)] = plot_opt_ref_results_tuned(rfil_t,pulse_nums(2));
    [neff_m,SNR_m(i,:)] = plot_opt_ref_results_matched(rfil_m,pulse_nums(3));
    
    % Store best SNR for convenience
    SNR_umax(i) = max(SNR_u(i,:)); 
    SNR_tmax(i) = max(SNR_t(i,:));
    SNR_mmax(i) = max(SNR_m(i,:));
end

% -------------------------------------------------------------------
% Excitation pulse results (tuned probe)
% -------------------------------------------------------------------

% Set rectangular excitation and refocusing pulse parameters
vars.len = 1; % Units of T_90
vars.rat = 1.6;

% Compute SNR of rectangular excitation and refocusing pulses as a reference
[mrx_ru,~,~,~,SNRasy_ru] = plot_masy_rect_untuned(vars); % Untuned probe
[mrx_rt,~,~,~,SNRasy_rt] = plot_masy_rect_tuned(vars); % Tuned probe
[mrx_rm,~,~,~,SNRasy_rm] = plot_masy_rect_matched(vars); % Matched probe

% List of excitation pulse result files
elen = {'1','1p5','2'}; nres = length(elen);
elen_val = [1,1.5,2]; % Refocusing pulse lengths (normalized to T_180)

SNRasy_t = zeros(nres,numiter);
SNRasy_tmax = zeros(1,nres);
for i = 1:nres
    efil_t = ['exc_tuned_500k_' elen{i}]; % Tuned probe
    
    pulse_nums = round(numiter*rand(1)); % Random pulse for initial plotting
    [mrx_t,masy_t,echo_rx_t,tvect_t,SNRasy_t(i,:)] = plot_opt_exc_results_tuned(efil_t,pulse_nums);
    
    % Store best SNR for convenience
    SNRasy_tmax(i) = max(SNRasy_t(i,:));
end

% -------------------------------------------------------------------
% Plot results
% -------------------------------------------------------------------

close all;

% Plot un-normalized (actual) SNR (refocusing)
figure;
plot(rlen_val,SNR_umax,'LineWidth',1); hold on;
plot(rlen_val,SNR_tmax,'LineWidth',1);
plot(rlen_val,SNR_mmax,'LineWidth',1);
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
xlabel('Refocusing pulse length, t_p/T_{180}');
ylabel('Normalized SNR (rms units)');
legend({'Untuned probe','Tuned probe','Matched probe'})

% Plot normalized SNR (refocusing)
figure;
plot(rlen_val,SNR_umax/SNR_ru,'LineWidth',1); hold on;
plot(rlen_val,SNR_tmax/SNR_rt,'LineWidth',1);
plot(rlen_val,SNR_mmax/SNR_rm,'LineWidth',1);
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
xlabel('Refocusing pulse length, t_p/T_{180}');
ylabel('SNR ratio (rms units)');
legend({'Untuned probe','Tuned probe','Matched probe'})

% Plot normalized SNR (excitation)
figure;
plot(elen_val,SNRasy_tmax/SNRasy_rt,'LineWidth',1);
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
xlabel('Refocusing pulse length, t_p/T_{180}');
ylabel('SNR ratio (rms units)');
legend({'Tuned probe'})