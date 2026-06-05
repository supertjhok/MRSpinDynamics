% DIFF_ECHO_Q
% Compare diffusion-affected matched-probe CPMG echoes across coil Q values.
%
% Purpose
%   Runs a short parameter sweep over Q for a matched-probe CPMG diffusion
%   simulation. This is a compact example for checking how probe bandwidth
%   affects the diffusion-aware echo response.
%
% Inputs
%   This script takes no function arguments. Parameters are set directly in the
%   script: NE, TE, T1, T2, dz, Delta, T_90, and Q.
%
% Outputs
%   Leaves the final echo_rx and tvect arrays in the workspace. Each loop
%   iteration overwrites those variables.
%
% Key functions
%   sim_dif_matched_CPMG_noRx.
%
% Notes
%   This script currently does not save or plot the full Q sweep. Add storage
%   inside the loop if all Q-dependent echoes are needed.
% -------------------------------------------------------------------------
close all

NE = 5;
TE = 1000e-6;
T1 = 100e-3;
T2 = 100e-3;
dz = 0.001;
Delta = 1000e-6;
T_90 = [100e-6];
Q = [50 500 5000 50000 500000];

for i = 1:5
    [echo_rx,tvect]=sim_dif_matched_CPMG_noRx(NE,TE,T1,T2,dz,Delta,T_90,Q(i));
end
