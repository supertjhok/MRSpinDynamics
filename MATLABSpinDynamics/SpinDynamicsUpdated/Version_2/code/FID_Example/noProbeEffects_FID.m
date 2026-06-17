% NOPROBEEFFECTS_FID
% Simulate an ideal-probe free induction decay.
%
% Purpose
%   Demonstrates the simplest FID workflow with no probe-circuit effects. The
%   script builds ideal FID parameters and runs the ideal FID simulator.
%
% Inputs
%   This script takes no function arguments. It uses set_params_ideal_FID to
%   construct the simulation and pulse-sequence parameter structures.
%
% Outputs
%   Leaves mrx, sp, and pp in the workspace.
%
% Key functions
%   set_params_ideal_FID, simFID_ideal.
%
% Notes
%   Run from a MATLAB path that includes the Version_2 code folders.
% -------------------------------------------------------------------------
close all
[sp,pp] = set_params_ideal_FID;

[mrx] = simFID_ideal(sp,pp);
