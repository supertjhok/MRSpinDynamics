
% Pulse sequence
% ----------------------------------------------
params.NE=20; % Number of echoes
params.TE=2e-3; % Echo period (sec)
tauvect = logspace(3e-3,10,7)
T1 = 2;
T2 = 2;

% ----------------------------------------------
% Run simulation
% ----------------------------------------------
 [echo_int_all]=sim_cpmg_ir_matched_probe_relax4(params.NE,params.TE,tauvect,T1,T2)
% 
% figure
% plot(real(echo_int_all))

%[macq,mrx]=calc_macq_matched_probe_relax4(sp,pp)(params.NE,params.TE,T1,T2)
