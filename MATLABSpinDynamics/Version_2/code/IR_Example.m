% Imaging example
% ----------------------------------------------
close all
clear all
% ----------------------------------------------
% Define parameters
% ----------------------------------------------

% Pulse sequence
% ----------------------------------------------
params.NE=15; % Number of echoes
params.TE=500e-6; % Echo period (sec)
params.Tgrad=0.5e-3; % Gradient length (sec)
params.T1 = 0.5;
params.T2 = 0.5;
%Tau = logspace(log10(3e-6),log10(5),8);
Tau = linspace(3e-6,5,8);
nTau = size(Tau,2);
%Tau = linspace(1e-6,10,5);  
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
[echo_int_all echo_asymp_all]=sim_cpmg_ir_matched_probe_relax(params.NE,params.TE,Tau,params.T1,params.T2)
% [echo_int_all]=sim_cpmg_matched_probe_img(params);
figure 
i = 1
% for i = 1:nTau
%     figure
%     plot(squeeze(real(echo_asymp_all(i,ceil(params.NE/2),:))))
%     hold on
%     plot(squeeze(imag(echo_asymp_all(i,ceil(params.NE/2),:))))
%     legend('Real','Imag')
% end

% figure 
% for i = 1:nTau
%     plot(squeeze(imag(echo_asymp_all(i,ceil(params.NE/2),:))))
%     hold on
%     legend('imag')
% end
% plot(squeeze(real(echo_asymp_all(10,:,1))))
% legend('imag','real')
