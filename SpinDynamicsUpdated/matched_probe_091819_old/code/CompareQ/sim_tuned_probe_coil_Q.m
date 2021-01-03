% Simulate a tuned probe with varying coil Q
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
close all
[sp, pp] = set_params_tuned_JMR; % Define system parameters

Qvec = linspace(10,100,11); % Vary coil Q

SNR = zeros(1, length(Qvec)); % Storage for output variables
echo_rx = zeros(4*length(sp.del_w),length(Qvec));


% Run simulations
for i=1:length(Qvec)
    sp.plt_mn=0; sp.plt_tx=0; sp.plt_rx=0; sp.plt_axis=0; sp.plt_echo=0; % Turn plotting off to reduce the number of plots
    sp.Q = Qvec(i); % Change coil Q
    sp.R = 2*pi*sp.f0*sp.L/sp.Q;
    [mrx(i,:),echo_rx(:,i),tvect,SNR(i)]=calc_masy_tuned_probe_Qsweep(sp,pp); % Simulate narrowband system
   
end

% Plot results
%-----------------------
figure;
imagesc(sp.del_w,Qvec,abs(mrx)); 
xlim([-5 5]); colorbar;
ylabel('Coil Q'); xlabel('\Delta\omega_{0}');
set(gca,'FontSize',14); title('M_{rx} (magnitude)');
export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\TunedQ\absMRXTuned.pdf

%-----------------------
figure;
imagesc(tvect/pi,Qvec,abs(echo_rx'));
xlim([-4 4]); colorbar;
ylabel('Coil Q'); xlabel('Time (t/T_{180})');
set(gca,'FontSize',14); title('Echo (magnitude)');
export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\TunedQ\absEchoTuned.pdf

%-----------------------
figure;
plot(Qvec,SNR); xlabel('Coil Q'); ylabel('SNR of asymptotic echo');
set(gca,'FontSize',14);
export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\TunedQ\SNRtuned.pdf
