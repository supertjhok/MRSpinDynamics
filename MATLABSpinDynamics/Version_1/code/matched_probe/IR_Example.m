% Imaging example
% ----------------------------------------------
%close all

% ----------------------------------------------
% Define parameters
% ----------------------------------------------

% Pulse sequence
% ----------------------------------------------
params.NE=5; % Number of echoes
params.TE=384e-6; % Echo period (sec)
params.Tgrad=0.5e-3; % Gradient length (sec)
params.T1 = 0.5;
params.T2 = 0.5;
numTau = 32;
% Tau = logspace(log10(1e-8),log10(10),5);
 Tau = linspace(1e-6,10,numTau);  
% Sample parameters: change as needed to get interesting images
% ----------------------------------------------
params.rho=ones(16,16); % Spin density map (kind of boring right now)
params.T1map=5e-3*ones(16,16); % T1 map (also boring)
params.T2map=5e-3*ones(16,16); % T2 map (also boring)

% Image parameters
% ----------------------------------------------
params.pxz=[16,16]; % Image size in pixels (x,z)
params.FOV=[20,20]; % FOV in pixel units (x,z)

% ----------------------------------------------
% Run simulation
% ----------------------------------------------
[echo_int_all echo_asymp_all]=sim_cpmg_ir_matched_probe_relaxComp(params.NE,params.TE,Tau,params.T1,params.T2)
% [echo_int_all]=sim_cpmg_matched_probe_img(params);
figure 
i = 1
% Phase echos
% for i = 1:numTau
%     phi = angle(sum(squeeze((echo_asymp_all(i,3,:)))));
%     echo_asymp_all(i,3,:) = echo_asymp_all(i,3,:)*exp(-1i*phi);    
% end

for i = 1:numTau
    plot(squeeze(real(echo_asymp_all(i,3,:))))
    hold on
    legend('Real')
end

figure 
for i = 1:numTau
    plot(squeeze(imag(echo_asymp_all(i,3,:))))
    hold on
    legend('imag')
end
% plot(squeeze(real(echo_asymp_all(10,:,1))))
% legend('imag','real')
