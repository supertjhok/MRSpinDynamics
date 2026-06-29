% SIM_CPMG_IR_MATCHED_PROBE_COMPARE
% Compare matched-probe CPMG-IR implementation runtimes.
%
% Purpose
%   Runs four CPMG inversion-recovery matched-probe implementations with the
%   same inputs and reports elapsed times for each optimization level.
%
% Inputs
%   This script takes no function arguments. Echo count, echo spacing, tau
%   vector, T1, and T2 are defined directly in the script.
%
% Outputs
%   Prints timings via tic/toc and leaves echo_int_all1 through echo_int_all4
%   in the workspace.
%
% Key functions
%   sim_cpmg_ir_matched_probe_relax, sim_cpmg_ir_matched_probe_relax2,
%   sim_cpmg_ir_matched_probe_relax3, sim_cpmg_ir_matched_probe_relax4.
%
% Notes
%   Start a parallel pool first for a fair comparison of parfor-based versions.
%   Historical timings: 5.3/3.1/1.7/1.2 s on a 32-core host and
%   149/70/26/4.4 s on a 2-core laptop.
% -------------------------------------------------------------------------

% Input parameters
NE=10; TE=0.5e-3;
tauvect=linspace(0.5,10,20)*1e-3;
T1=5e-3; T2=5e-3;

% Unoptimized - pulse shapes and matrices recalculated each time
tic;
echo_int_all1=sim_cpmg_ir_matched_probe_relax(NE,TE,tauvect,T1,T2);
toc

% Optimization 1 - precompute pulse shapes
tic;
echo_int_all2=sim_cpmg_ir_matched_probe_relax2(NE,TE,tauvect,T1,T2);
toc

% Optimization 2 - precompute pulse shapes and rotation matrices
tic;
echo_int_all3=sim_cpmg_ir_matched_probe_relax3(NE,TE,tauvect,T1,T2);
toc

% Optimization 3 - precompute pulse shapes, rotation matrices, and
% isochromats; do not convolve acquired spectra with acquisition window
tic;
echo_int_all4=sim_cpmg_ir_matched_probe_relax4(NE,TE,tauvect,T1,T2);
toc
