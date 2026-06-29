% Simulate x-y phase encoding pair

NE=10; % Number of echoes
TE=0.5e-3; % Echo period
T_90=25e-6; % Nominal 90 degree pulse length
T1=10e-3; T2=10e-3; % Relaxation times
grad=[0.2 1e-3]; % Gradient (strength, duration)

% Run simulations
[echo_rx1,tvect,SNR]=sim_cpmg_matched_probe_gradx(NE,TE,T1,T2,grad);
[echo_rx2,~,~]=sim_cpmg_matched_probe_grady(NE,TE,T1,T2,grad);

% Plot fesults
figure;
for i=1:NE
plot((tvect*T_90/(pi/2)+i*TE)*1e3,real(echo_rx2(i,:)),'b-'); hold on;
plot((tvect*T_90/(pi/2)+i*TE)*1e3,imag(echo_rx1(i,:)),'r-');
end