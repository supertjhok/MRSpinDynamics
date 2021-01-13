% Simulate a tuned-and-matched probe with varying frequency offset
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% Last modified: 01/03/21

[sp, pp] = set_params_matched_Orig; % Define system parameters

% Vary matching frequency (in units of probe BW = fin/Q)
fin = sp.fin; Q = sp.Q;
f0_vec = fin+(fin/Q)*linspace(-5,5,51); 

SNR = zeros(1, length(f0_vec)); % Storage for output variables
echo_rx = zeros(4*length(sp.del_w),length(f0_vec)); tvect2=echo_rx;
mrx = zeros(length(f0_vec),length(sp.del_w));

% Turn plotting off to reduce the number of plots
sp.plt_mn=0; sp.plt_tx=0; sp.plt_rx=0; sp.plt_axis=0; sp.plt_echo=0;

% Run simulations
parfor i=1:length(f0_vec)
    sp_curr = sp;
    sp_curr.f0 = f0_vec(i); % Change probe matching frequency
    
    [mrx(i,:),~,SNR(i)]=calc_masy_matched_probe_Orig(sp_curr,pp); % Simulate narrowband system
    [echo_rx(:,i),tvect2(:,i)]=calc_time_domain_echo(mrx(i,:),sp_curr.del_w,0,0);
end

% Plot results
%-----------------------
figure;
imagesc(sp.del_w,(f0_vec-sp.f0)/(fin/Q),abs(mrx)); 
xlim([-5 5]); colorbar;
ylabel('Frequency error (units of f_{in}/Q)'); xlabel('\Delta\omega_{0}');
set(gca,'FontSize',15);  set(gca,'FontWeight','bold');
title('M_{rx} (magnitude)');

%-----------------------
figure;
imagesc(tvect2(:,1)/pi,(f0_vec-sp.f0)/(fin/Q),abs(echo_rx)');
xlim([-4 4]); colorbar;
ylabel('Frequency error (units of f_{in}/Q)'); xlabel('Time (t/T_{180})');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
title('Echo (magnitude)');

%-----------------------
figure;
plot((f0_vec-sp.f0)/(fin/Q),SNR,'LineWidth',1); 
xlabel('Frequency error (units of f_{in}/Q)'); 
ylabel('SNR of asymptotic echo');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
