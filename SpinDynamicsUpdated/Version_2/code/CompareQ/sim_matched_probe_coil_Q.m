% Simulate a tuned-and-matched probe with varying coil Q
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19

[sp, pp] = set_params_matched; % Define system parameters

Qvec = linspace(10,100,100); % Vary coil Q

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
xlim([-5 5]); 
h = colorbar;
set(get(h,'title'),'string','Signal Amp');
ylabel('Coil Q'); 
xlabel('\Delta\omega_{0}');
set(gca,'FontSize',14); 
title('M_{rx} (magnitude)');
whiteBg
font
setSize
export_fig('F:\Dropbox\Apps\Overleaf\Portable and Autonomous Magnetic Resonance\Figures\SpecQMatched.pdf')

%-----------------------
figure;
imagesc(tvect/pi,Qvec,abs(echo_rx)');
xlim([-2 2]); 
h = colorbar;
set(get(h,'title'),'string','Signal Amp');
ylabel('Coil Q'); 
xlabel('Time (t/T_{180})');
set(gca,'FontSize',14);
title('Echo (magnitude)');
whiteBg
font
setSize
export_fig('F:\Dropbox\Apps\Overleaf\Portable and Autonomous Magnetic Resonance\Figures\EchoQMatched.pdf')

%-----------------------
figure;
plot(Qvec,SNR,'LineWidth',2); 
xlabel('Coil Q');
ylabel('SNR of asymptotic echo');
set(gca,'FontSize',14);
whiteBg
font
setSize
% export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\TunedQ\SNRtuned.pdf
export_fig('F:\Dropbox\Apps\Overleaf\Portable and Autonomous Magnetic Resonance\Figures\SNRQMatched.pdf')
