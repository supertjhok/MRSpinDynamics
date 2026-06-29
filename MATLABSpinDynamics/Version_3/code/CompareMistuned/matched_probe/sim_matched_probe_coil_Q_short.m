% SIM_MATCHED_PROBE_COIL_Q_SHORT
% Run a short matched-probe coil-Q sweep for mistuning comparisons.
%
% Purpose
%   Sweeps a small set of matched-probe coil-Q values, computes received
%   spectra and echoes, and plots signal/SNR trends for quick comparison work.
%
% Inputs
%   This script takes no function arguments. Qvec is defined directly in the
%   script, and set_params_matched supplies the base probe and pulse settings.
%
% Outputs
%   Creates figures for abs(mrx), abs(echo_rx), and SNR. Leaves Qvec, SNR,
%   echo_rx, tvect, mrx, sp, and pp in the workspace.
%
% Key functions
%   set_params_matched, calc_masy_matched_probe.
%
% Notes
%   This short 11-point sweep is a lightweight companion to the denser
%   CompareQ/sim_matched_probe_coil_Q workflow. The distinct filename avoids a
%   MATLAB path-order collision when Version_3/code is added with genpath.
%
% Written by: Soumyajit Mandal, 03/28/19
% -------------------------------------------------------------------------

[sp, pp] = set_params_matched; % Define system parameters

Qvec = linspace(10,100,11); % Vary coil Q

SNR = zeros(1, length(Qvec)); % Storage for output variables
echo_rx = zeros(4*length(sp.del_w),length(Qvec));
mrx = zeros(length(Qvec),length(sp.del_w));

% Run simulations
for i=1:length(Qvec)
    sp.plt_mn=0; sp.plt_tx=0; sp.plt_rx=0; sp.plt_axis=0; sp.plt_echo=0; % Turn plotting off to reduce the number of plots
    sp.Q = Qvec(i); % Change coil Q
    [mrx(i,:),echo_rx(:,i),tvect,SNR(i)]=calc_masy_matched_probe(sp,pp); % Simulate narrowband system
end

% Plot results
%-----------------------
figure;
imagesc(sp.del_w,Qvec,abs(mrx)); 
xlim([-5 5]); colorbar;
ylabel('Coil Q'); xlabel('\Delta\omega_{0}');
set(gca,'FontSize',14); title('M_{rx} (magnitude)');

%-----------------------
figure;
imagesc(tvect/pi,Qvec,abs(echo_rx)');
xlim([-4 4]); colorbar;
ylabel('Coil Q'); xlabel('Time (t/T_{180})');
set(gca,'FontSize',14); title('Echo (magnitude)');

%-----------------------
figure;
plot(Qvec,SNR); xlabel('Coil Q'); ylabel('SNR of asymptotic echo');
set(gca,'FontSize',14);
