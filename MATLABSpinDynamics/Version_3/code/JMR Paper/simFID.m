% SIMFID
% Run a low-level FID simulation and plot time/frequency-domain signals.
%
% Purpose
%   Builds a small arbitrary-pulse FID parameter structure, runs the legacy
%   arbitrary-pulse simulator, and plots the simulated acquisition.
%
% Inputs
%   This script takes no function arguments. All pulse and relaxation settings
%   are defined directly in the script.
%
% Outputs
%   Creates figures for fft(macq) and macq. Leaves sp, pp, params, and macq in
%   the workspace.
%
% Key functions
%   set_params_FID, sim_spin_dynamics_arb7.
%
% Notes
%   This is a historical JMR-paper workflow and uses an older arbitrary-pulse
%   simulator rather than the current arb10 kernel.
% -------------------------------------------------------------------------

[sp, pp] = set_params_FID

params.tp = [25e-6 (25e-6*3) 50e-6 25e-6*6];
params.phi = [0 0 pi/2 0];
params.amp = [1 0 1 0];
params.acq = [0 0 0 1];
params.grad = [0 0 0 0];
params.len_acq = (25e-6*6);
numpts=2000;
maxoffs=1;
params.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);
params.del_wg = zeros(size(params.del_w));
params.w_1 = ones(size(params.del_w));
params.T1n = 200000;
params.T2n = 200000;

params.m0 = 1; % Initial magnetization vector amplitude
params.mth = 1; % Thermal magnetization vector amplitude

[macq]=sim_spin_dynamics_arb7(params)

figure
plot((real(fft(macq))));
hold on
plot((imag(fft(macq))));
title('fft')

figure
plot(real(macq))
hold on
plot(imag(macq))
title('macq')
