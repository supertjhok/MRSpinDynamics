% Compare code running speeds
% Make sure parallel pool is running first to get a fair comparison
% ----------------------------------------------------------------------
% Example: CPMG-IR sequence
% Results (on tonks, 32 cores): 5.3, 3.1, 1.7, 1.2 (sec)
% Results (on laptop, 2 cores): 149, 70, 26, 4.4 (sec)

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
