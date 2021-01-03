% Calculate acquired magnetization of arbitrary sequence including transmitter and
% receiver bandwidth effects for a tuned-and-matched probe
% --------------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% 04/09/19: Modified to include absolute RF phase parameter (psi)
% 09/09/19: Modified to allow arbitrary pulse sequences & (w0,w1) maps

function [macq,mrx,pnoise,f,echo_rx,tvect]=calc_macq_matched_probe_relax2(sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
T1=(pi/2)*sp.T1/T_90; T2=(pi/2)*sp.T2/T_90; % Relaxation time constants (normalized)

% Convert to normalized time
tacq=(pi/2)*pp.tacq/T_90; % Acquisition window length
tdw=(pi/2)*pp.tdw/T_90; % Receiver dwell time

% Create structure
params.tp=pp.tp; params.phi=pp.phi; params.amp=pp.amp; params.acq=pp.acq;
params.len_acq=tacq; params.del_w=sp.del_w; params.w_1=sp.w_1;
params.T1n=T1; params.T2n=T2;
params.m0=sp.m0; params.mth=sp.mth;

% Plot pulse sequence
if sp.plt_sequence
    figure;
    tplt=[0 cumsum(params.tp)]*T_90/(pi/2);
    subplot(2,1,1);
    rf_wfm=[params.amp params.amp(end)].*exp(1i*[params.phi params.phi(end)]); % Complex RF pulses
    stairs(tplt*1e3,real(rf_wfm),'b-','LineWidth',1); hold on;
    stairs(tplt*1e3,imag(rf_wfm),'r-','LineWidth',1);
    set(gca,'FontSize',14); ylabel('RF pulses');
    legend({'Real','Imag'});
    title('Transmitted pulse sequence');
    subplot(2,1,2);
    stairs(tplt*1e3,[params.acq params.acq(end)],'LineWidth',1);
    set(gca,'FontSize',14); ylabel('Acquisition');
    xlabel('Time (ms)');
end

% Calculate spin dynamics
[macq]=sim_spin_dynamics_arb6_relax(params);

% Allocate space for received signals
siz_acq=size(macq); nacq=siz_acq(1);
mrx=zeros(nacq,siz_acq(2)); pnoise=zeros(1,siz_acq(2));

nacq_t=round(tacq/tdw)+1; % Number of acquired time-domain points
echo_rx=zeros(nacq,nacq_t);
for i=1:nacq
    sp.plt_rx=0;
    
    % Filtering by the receiver
    [mrx(i,:),pnoise,f]=matched_probe_rx_no_mf(sp,pp,macq(i,:),sp.tf1,sp.tf2);
    
    % Calculate time-domain echo
    [echo_rx(i,:),tvect]=calc_time_domain_echo_arb(mrx(i,:),sp.del_w,tacq,tdw,sp.plt_echo);
end