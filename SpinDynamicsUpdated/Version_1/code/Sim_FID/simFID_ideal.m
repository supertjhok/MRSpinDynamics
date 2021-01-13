function [mrx] = simFID_ideal(sp,pp)
% Simulate a basic FID sequence with rectangular pulses and finite
% transmit and receive bandwidth
% -------------------------------------------------------------------
T_90 = (pi/2)*pp.T_90/pp.T_90;

% Set acquisition parameters
tacq=(pi/2)*pp.tacq/pp.T_90; % Normalized acquisition window length
tdw=(pi/2)*pp.tdw/pp.T_90; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points
acqDelay = (pi/2)*pp.acqDelay/pp.T_90;
% Set plotting parameters
sp.plt_tx = 0;
sp.plt_rx = 0;
sp.plt_sequence = 1; % Plots /off
sp.plt_axis = 0; 
sp.plt_mn = 0; 
sp.plt_echo = 0;

% Create pulse sequence (in real time)
% --------------------------------------------

% Create structure
params.tp=[T_90 acqDelay tacq acqDelay]; %segment times
params.phi=[0 0 0 0]; %segment phase
params.amp=[1 0 0 0]; %segment amp
params.acq=[0 0 1 0]; %segment acq
params.grad=[0 0 0 0];
params.len_acq=tacq; %acq length
params.del_w=sp.del_w;  %isocromats
params.w_1=sp.w_1;      %omega 1
params.m0=sp.m0;     %initial magnetitization
params.T1n = sp.T1;
params.T2n = sp.T2;
params.mth = sp.mth;



% Calculate spin dynamics
% [macq]=sim_spin_dynamics_arb6(params);
[macq,tacq]=calc_macq_fid(sp,pp,params)

figure
plot(sp.del_w,fftshift(real(macq)))
hold on
plot(sp.del_w,fftshift(imag(macq)))

figure
plot(abs(fft(macq)));
mrx = macq;
% Run PAP phase cycle
% [~,mrx1,pnoise,f,~,~]=calc_macq_matched_probe(sp,pp);

% pp.phi=[-pexc pref]; % Change phase of excitation pulse
% [~,mrx2,~,~,~,~]=calc_macq_matched_probe(sp,pp);

% mrx=(mrx1-mrx2); pnoise=pnoise*sqrt(2); % Apply phase cycle

% figure; % Figure to show outputs
% % Calculate time-domain echoes
% echo_rx=zeros(NE,nacq);
% SNR=zeros(1,NE);
% for i=1:NE
%     [echo_rx(i,:),tvect]=calc_time_domain_echo_arb(mrx(i,:),sp.del_w,tacq,tdw,sp.plt_echo);
%     plot((tvect*T_90/(pi/2)+i*TE)*1e3,real(echo_rx(i,:)),'b-'); hold on;
%     plot((tvect*T_90/(pi/2)+i*TE)*1e3,imag(echo_rx(i,:)),'r-');
%     
%     SNR(i)=matched_probe_rx_snr(sp,mrx(i,:),pnoise,f);
% end

set(gca,'FontSize',14);
xlabel('Time (ms)'); ylabel('Received echoes');
end

