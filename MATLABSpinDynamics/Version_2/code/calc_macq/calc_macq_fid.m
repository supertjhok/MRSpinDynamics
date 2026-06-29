% Calculate acquired magnetization of arbitrary sequence with no transmitter and
% receiver bandwidth effects
% Don't calculate time-domain echoes to save time
% --------------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% 04/09/19: Modified to include absolute RF phase parameter (psi)
% 09/09/19: Modified to allow arbitrary pulse sequences & (w0,w1) maps

function [macq,tacq]=calc_macq_fid(sp,pp,params)

T_90=pp.T_90; % Rectangular T_90 time
params.T1n=(pi/2)*sp.T1/T_90; 
params.T2n=(pi/2)*sp.T2/T_90; % Relaxation time constants (normalized)

% Convert to normalized time
tacq=(pi/2)*pp.tacq/T_90; % Acquisition window length
params.mth = sp.mth;
T_90 = (pi/2)*pp.T_90/pp.T_90;
% Plot pulse sequence
if sp.plt_sequence
    figure;
    tplt=[0 cumsum(params.tp)]*T_90/(pi/2);
    subplot(3,1,1);
    stairs(tplt*1e3,[params.amp params.amp(end)],'LineWidth',1); % RF amplitude
    set(gca,'FontSize',14); ylabel('RF pulses');
    title('Transmitted pulse sequence');
    subplot(3,1,2);
    stairs(tplt*1e3,[params.grad params.grad(end)],'LineWidth',1);
    set(gca,'FontSize',14); ylabel('Gradient');
    subplot(3,1,3);
    stairs(tplt*1e3,[params.acq params.acq(end)],'LineWidth',1);
    set(gca,'FontSize',14); ylabel('Acquisition');
    xlabel('Time (ms)');
end


% Calculate spin dynamics
[macq]=sim_spin_dynamics_arb7(params);