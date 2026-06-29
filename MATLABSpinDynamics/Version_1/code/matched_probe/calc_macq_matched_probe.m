% Calculate acquired magnetization of arbitrary sequence including transmitter and
% receiver bandwidth effects for a tuned-and-matched probe
% --------------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% 04/09/19: Modified to include absolute RF phase parameter (psi)
% 09/09/19: Modified to allow arbitrary pulse sequences & (w0,w1) maps

function [macq,mrx,pnoise,f,echo_rx,tvect]=calc_macq_matched_probe(sp,pp)

del_w=sp.del_w; % Offset frequency vector
w_1=sp.w_1; % Transmit/receive coil sensitivity factor vector

T_90=pp.T_90; % Rectangular T_90 time
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Time delay used to account for ring down
tdel=2*T_90; % Added delay
tdeln=(pi/2)*tdel/T_90; % Normalized delay

% Convert to normalized time
tacq=(pi/2)*pp.tacq/T_90; % Acquisition window length
tdw=(pi/2)*pp.tdw/T_90; % Receiver dwell time

% Initialize variables
tp=pp.tp; phi=pp.phi; amp=pp.amp; acq=pp.acq; % Real time
tp_curr=[]; phi_curr=[]; amp_curr=[]; acq_curr=[]; % Normalized time
state_rf=0; state_rf_curr=0;

for i=1:length(tp)

    if state_rf==0 && amp(i)>0 % Beginning of RF pulse
        state_rf_curr=1; % Set new flag
        ind=i;
    end
    
    if state_rf==0 && amp(i)==0 % Time delay, copy to new structure
        tp_curr = [tp_curr (pi/2)*tp(i)/T_90]; phi_curr = [phi_curr phi(i)];
        amp_curr = [amp_curr amp(i)];
        acq_curr = [acq_curr acq(i)];
    end
    
    if state_rf==1 && amp(i)==0 % End of RF pulse
        state_rf_curr=0; % Set new flag
        
        % Add delay to RF pulse to account for ring down, create structure
        pp_curr=pp;
        pp_curr.tp = [tp(ind:i-1) tdel];
        pp_curr.phi = [phi(ind:i-1) 0];
        pp_curr.amp = [amp(ind:i-1) 0];
        
        % Calculate RF pulse
        sp.plt_rx=0; % Turn off plotting
        [tvect, Icr, tf1, tf2] = find_coil_current(sp,pp_curr);
        
        delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
        texc=delt*ones(1,length(tvect));
        pexc=atan2(imag(Icr),real(Icr));
        aexc=abs(Icr);
        aexc(aexc<amp_zero)=0; % Threshold amplitude
        
        % Remove added delay from RF pulse
        texc=[texc -tdeln]; pexc=[pexc 0]; aexc=[aexc 0];
        acq_exc=zeros(1,length(texc));
        
        % Add to pulse sequence structure
        tp_curr = [tp_curr texc (pi/2)*tp(i)/T_90];
        phi_curr = [phi_curr pexc phi(i)];
        amp_curr = [amp_curr aexc amp(i)];
        acq_curr = [acq_curr acq_exc acq(i)];
    end
    
    state_rf=state_rf_curr; % Update flag
end

% Plot pulse sequence
if sp.plt_sequence
    figure;
    tplt=[0 cumsum(tp_curr)]*T_90/(pi/2);
    subplot(2,1,1);
    rf_wfm=[amp_curr amp_curr(end)].*exp(1i*[phi_curr phi_curr(end)]); % Complex RF pulses
    stairs(tplt*1e3,real(rf_wfm),'b-','LineWidth',1); hold on;
    stairs(tplt*1e3,imag(rf_wfm),'r-','LineWidth',1);
    set(gca,'FontSize',14); ylabel('RF pulses');
    legend({'Real','Imag'});
    title('Transmitted pulse sequence');
    subplot(2,1,2);
    stairs(tplt*1e3,[acq_curr acq_curr(end)],'LineWidth',1);
    set(gca,'FontSize',14); ylabel('Acquisition');
    xlabel('Time (ms)');
end
% Create structure
params.tp=tp_curr; params.phi=phi_curr; params.amp=amp_curr; params.acq=acq_curr;
params.len_acq=tacq; params.del_w=del_w; params.w_1=w_1;
params.m0=sp.m0;

% Calculate spin dynamics
[macq]=sim_spin_dynamics_arb6(params);

% Allocate space for received signals
siz_acq=size(macq); nacq=siz_acq(1);
mrx=zeros(nacq,siz_acq(2)); pnoise=zeros(1,siz_acq(2));

nacq_t=round(tacq/tdw)+1; % Number of acquired time-domain points
echo_rx=zeros(nacq,nacq_t);
for i=1:nacq
    sp.plt_rx=0;
    
    % Filtering by the receiver
    [mrx(i,:),pnoise,f]=matched_probe_rx_no_mf(sp,pp,macq(i,:),tf1,tf2);
    
    % Calculate time-domain echo
    [echo_rx(i,:),tvect]=calc_time_domain_echo_arb(mrx(i,:),del_w,tacq,tdw,sp.plt_echo);
end