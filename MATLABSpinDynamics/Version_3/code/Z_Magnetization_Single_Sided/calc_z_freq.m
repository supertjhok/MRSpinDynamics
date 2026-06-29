% CALC_Z_FREQ
% Plot single-sided matched-probe z magnetization versus real frequency.
%
% Purpose
%   Builds single-sided matched-probe parameters, computes nutation
%   magnetization, converts normalized offsets to real frequency, and plots the
%   z-magnetization spectrum.
%
% Inputs
%   This script takes no function arguments. Parameters are created by
%   set_params_matched_SS.
%
% Outputs
%   Creates a z-magnetization plot and leaves sp, pp, mz, tvect, nutFreq, and
%   del_w_real in the workspace.
%
% Key functions
%   set_params_matched_SS, calc_masy_matched_nut.
%
% Notes
%   This is a single-sided z-magnetization exploration script.
% -------------------------------------------------------------------------

close all
[sp, pp] = set_params_matched_SS; % Define system parameters
[mz,tvect]=calc_masy_matched_nut(sp,pp); % Simulate narrowband system

nutFreq = 1/(pp.T_90*4);
% *19230.77
figure;

del_w_real = (sp.del_w/(2*pi)*nutFreq);

plot(del_w_real/10^6,mz);
