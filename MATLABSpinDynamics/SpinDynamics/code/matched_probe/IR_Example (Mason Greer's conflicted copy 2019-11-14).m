% Imaging example
% ----------------------------------------------
close all

% ----------------------------------------------
% Define parameters
% ----------------------------------------------

% Pulse sequence
% ----------------------------------------------
params.NE=30; % Number of echoes
params.TE=10e-3; % Echo period (sec)
params.Tgrad=0.5e-3; % Gradient length (sec)
params.T1 = 0.05;
params.T2 = 0.05;

% Sample parameters: change as needed to get interesting images
% ----------------------------------------------
params.rho=ones(16,16); % Spin density map (kind of boring right now)
params.T1map=5e-3*ones(16,16); % T1 map (also boring)
params.T2map=5e-3*ones(16,16); % T2 map (also boring)

% Image parameters
% ----------------------------------------------
params.pxz=[16,16]; % Image size in pixels (x,z)
params.FOV=[20,20]; % FOV in pixel units (x,z)
Tau = logspace(log10(1e-6),log10(10),10);
% ----------------------------------------------
% Run simulation
% ----------------------------------------------
[echo_int_all]=sim_cpmg_ir_matched_probe_relax(params.NE,params.TE,Tau,params.T1,params.T2)
% [echo_int_all]=sim_cpmg_matched_probe_img(params);
