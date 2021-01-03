% Simulate a tuned-and-matched probe with varying frequency offset
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19

[sp, pp] = set_params_matched; % Define system parameters

% Vary matching frequency (in units of probe BW = fin/Q)
fin = sp.fin; Q = sp.Q;
f0_vec = fin+(fin/Q)*linspace(-5,5,21); 

SNR = zeros(1, length(f0_vec)); % Storage for output variables
echo_rx = zeros(4*length(sp.del_w),length(f0_vec));
mrx = zeros(length(f0_vec),length(sp.del_w));

% Run simulations
for i=1:length(f0_vec)
    sp.plt_mn=0; sp.plt_tx=0; sp.plt_rx=0; sp.plt_axis=0; sp.plt_echo=0; % Turn plotting off to reduce the number of plots
    sp.f0 = f0_vec(i); % Change coil Q
    [mrx(i,:),echo_rx(:,i),tvect,SNR(i)]=calc_masy_matched_probe(sp,pp); % Simulate narrowband system
end

% Plot results
%-----------------------
figure;
imagesc(sp.del_w,(f0_vec-f0)/(fin/Q),abs(mrx)); 
xlim([-5 5]); colorbar;
ylabel('Frequency error (units of f_{in}/Q)'); xlabel('\Delta\omega_{0}');
set(gca,'FontSize',14); title('M_{rx} (magnitude)');

%-----------------------
figure;
imagesc(tvect/pi,(f0_vec-f0)/(fin/Q),abs(echo_rx)');
xlim([-4 4]); colorbar;
ylabel('Frequency error (units of f_{in}/Q)'); xlabel('Time (t/T_{180})');
set(gca,'FontSize',14); title('Echo (magnitude)');

%-----------------------
figure;
plot((f0_vec-f0)/(fin/Q),SNR); xlabel('Frequency error (units of f_{in}/Q)'); ylabel('SNR of asymptotic echo');
set(gca,'FontSize',14);
