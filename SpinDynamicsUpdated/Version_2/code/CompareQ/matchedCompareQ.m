close all
[sp, pp] = set_params_matched_Orig; % Define system parameters

Qvec = linspace(10,100,46); % Vary coil Q

SNR = zeros(1, length(Qvec)); % Storage for output variables
echo_rx = zeros(4*length(sp.del_w),length(Qvec)); tvect2 = echo_rx;
mrx = zeros(length(Qvec),length(sp.del_w));

% Turn plotting off to reduce the number of plots
sp.plt_mn=0; sp.plt_tx=0; sp.plt_rx=0; sp.plt_axis=0; sp.plt_echo=0;

% Run simulations
parfor i=1:length(Qvec)
    sp_curr=sp;
    sp_curr.Q = Qvec(i); % Change coil Q
    sp_curr.R = 2*pi*sp_curr.f0*sp_curr.L/sp_curr.Q; % Change coil resistance
    
    % Simulate narrowband system
    [mrx(i,:),~,SNR(i)]=calc_masy_matched_probe_Orig(sp_curr,pp);
    [echo_rx(:,i),tvect2(:,i)]=calc_time_domain_echo(mrx(i,:),sp_curr.del_w,0,0);
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
imagesc(tvect2(:,1)/pi,Qvec,abs(echo_rx)'); % Time-domain echo magnetization
xlim([-4 4]);
colorbar
whiteBg
setSize
font
ylabel('Coil Q');
xlabel('Time (t/T_{180})');
title('Echo (magnitude)');


figure;
plot(Qvec,SNR); % SNR
xlabel('Coil Q');
ylabel('SNR of asymptotic echo');
whiteBg
setSize
font