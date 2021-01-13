function [macq,mrx]=calc_macq_tuned_probe_relax4(sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
T1=(pi/2)*sp.T1/T_90; 
T2=(pi/2)*sp.T2/T_90; % Relaxation time constants (normalized)

% Convert to normalized time
tacq=(pi/2)*pp.tacq/T_90; % Acquisition window length

% Create structure
params.tp=pp.tp;
params.pul=pp.pul; 
params.amp=pp.amp;
params.acq=pp.acq;
params.grad=pp.grad;
params.Rtot=pp.Rtot;
params.del_w=sp.del_w; 
params.del_wg=sp.del_wg;  
params.w_1=sp.w_1;
params.T1n=T1;
params.T2n=T2;
params.m0=sp.m0;
params.mth=sp.mth;

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
[macq]=sim_spin_dynamics_arb9(params);


% Received signals
mrx=macq.*sp.tf.*sp.w_1r; % Filtering by the receiver, coil sensitivity map