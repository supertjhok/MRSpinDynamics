close all
[sp, pp] = set_params_matched; % Define system parameters
 
Qvec = linspace(10,100,11); % Vary coil Q
 
SNR = zeros(1, length(Qvec)); % Storage for output variables
% echo_rx = zeros(4*length(sp.del_w),length(Qvec));
% mrx = zeros(length(Qvec),length(sp.del_w));
 
% Run simulations
for i=1:length(Qvec)
    % Turn plotting off to reduce the number of plots
    sp.plt_mn=0; sp.plt_tx=0; sp.plt_rx=0; sp.plt_axis=0; sp.plt_echo=0; 
    sp.Q = Qvec(i); % Change coil Q

    % Simulate narrowband system
    [mrx(i,:),tvect,SNR(i)]=calc_masy_matched_probe(sp,pp);
     [echo_rx(:,i),tvect2]=calc_time_domain_echo(mrx(i,:),sp.del_w,0,0);
end
 
% Plot results
figure; 
imagesc(sp.del_w,Qvec,abs(mrx)); % Asymptotic magnetization
xlim([-5 5]);
colorbar
whiteBg
setSize
font
ylabel('Coil Q')
xlabel('\Delta\omega_o');
title('Magnitude of M_{rx}');

figure;
imagesc(tvect2/pi,Qvec,abs(echo_rx)'); % Time-domain echo magnetization
xlim([-4 4]);
colorbar
whiteBg
setSize
font
ylabel('Coil Q');
xlabel('Time(t/T_{180}');
title('Echo(magnitude)');


figure;
plot(Qvec,SNR); % SNR
whiteBg
setSize
font