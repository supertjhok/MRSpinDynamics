% Imaging example
% ----------------------------------------------

% ----------------------------------------------
% Define parameters
% ----------------------------------------------

% Pulse sequence
% ----------------------------------------------
params.NE=6; % Number of echoes
params.TE=0.2e-3; % Echo period (sec)
params.Tgrad=0.5e-3; % Gradient length (sec)

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
[echo_int_all]=sim_cpmg_matched_probe_img(params);
