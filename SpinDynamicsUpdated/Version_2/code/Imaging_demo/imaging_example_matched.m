% IMAGING_EXAMPLE_MATCHED
% Simulate a simple CPMG image with matched-probe dynamics.
%
% Purpose
%   Builds a small image phantom from flower.png, assigns uniform T1/T2 maps,
%   and simulates CPMG image acquisition with matched-probe effects.
%
% Inputs
%   This script takes no function arguments. It requires flower.png to be
%   available on the MATLAB path or in the current working directory.
%
% Outputs
%   Leaves echo_int_all and params in the workspace.
%
% Key functions
%   imread, imresize, rgb2gray, sim_cpmg_matched_probe_img.
%
% Notes
%   sim_cpmg_matched_probe_img uses parfor internally, so the Parallel
%   Computing Toolbox is required as written. The commented parpool line can be
%   enabled to choose the worker pool explicitly.
% -------------------------------------------------------------------------
%parpool('local',32)
% ----------------------------------------------
% Define parameters
% ----------------------------------------------

% Pulse sequence
% ----------------------------------------------
params.NE=6; % Number of echoes
params.TE=0.2e-3; % Echo period (sec)
params.Tgrad=0.5e-3; % Gradient length (sec)
pixels = 16;
FOV = 40;
flower = imread('flower.png');
flowerResize = imresize(flower,[pixels pixels]);
IM = rgb2gray(flowerResize);

% Sample parameters: change as needed to get interesting images
% ----------------------------------------------
%params.rho=ones(16,16); % Spin density map (kind of boring right now)
params.rho = IM;
params.T1map=5e-3*ones(pixels,pixels); % T1 map (also boring)
params.T2map=5e-3*ones(pixels,pixels); % T2 map (also boring)

%params.T2map = IM;
% Image parameters
% ----------------------------------------------
params.pxz=[pixels,pixels]; % Image size in pixels (x,z)
params.FOV=[FOV,FOV]; % FOV in pixel units (x,z)

% ----------------------------------------------
% Run simulation
% ----------------------------------------------
[echo_int_all]=sim_cpmg_matched_probe_img(params);
