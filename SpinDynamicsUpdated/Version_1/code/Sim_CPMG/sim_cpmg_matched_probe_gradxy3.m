% Simulate a pair of CPMG sequences (x,y) with phase gradient,
% rectangular pulses, and finite transmit and receive bandwidth
% -----------------------------------------------------------------------
% Precompute excitation and refocusing pulses for speed

% NE -> Number of echoes to simulate
% TE -> Echo spacing (real value)
% T1, T2 -> Relaxation time constants (s)
% grad -> (normalized strength, length)
% -----------------------------------------------------------------------
% Sequence: (pi/2)x - (pi)x
function [echo_rx_xy,tvect]=sim_cpmg_matched_probe_gradxy3(NE,TE,T1,T2,grad)

% Simulate each echo of a CPMG sequence
[sp, pp]=set_params_matched; % Define system parameters
T_90=pp.T_90; % Nominal T_90 pulse length

% Set plotting parameters
sp.plt_tx = 0; sp.plt_rx = 0; sp.plt_sequence = 0; % Plots /off
sp.plt_axis = 0; sp.plt_mn = 0; sp.plt_echo = 0;
sp.plt_output = 1;

% Design matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Create (w0, w1) and coil sensitivity maps
numpts=sp.numpts; maxoffs=10;
sp.del_w=linspace(-maxoffs,maxoffs,numpts); % Linear gradient
sp.del_wg=grad(1)*ones(1,numpts); % Uniform offset
sp.w_1=ones(1,numpts); % Uniform transmit w_1
sp.w_1r=ones(1,numpts); % Uniform receiver sensitivity

% Set sample parameters
sp.m0=ones(1,numpts); sp.mth=ones(1,numpts); % Spin density
sp.T1=T1*ones(1,numpts); sp.T2=T2*ones(1,numpts); % Relaxation constants

% Set acquisition parameters
tacq=(pi/2)*pp.tacq/T_90; % Normalized acquisition window length
tdw=(pi/2)*pp.tdw/T_90; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points

% Create pulse sequence (in normalized time)
% --------------------------------------------
% Pre-calculate all pulses for speed
Rtot={};
pp_in.tp=pp.T_90; pp_in.tdel=2*pp.T_90; pp_in.phi=pi/2; pp_in.amp=1;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Exc pulse y (phase = pi/2)
sp.tf1=pp_out.tf1; sp.tf2=pp_out.tf2; % Save Rx transfer functions
Rtot{1}=calc_rotation_matrix(sp,pp_out);

pp_in.phi=3*pi/2;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Exc pulse -y (phase = 3*pi/2)
Rtot{2}=calc_rotation_matrix(sp,pp_out);

pp_in.tp=pp.T_180; pp_in.phi=0;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Ref pulse x (phase = 0)
Rtot{3}=calc_rotation_matrix(sp,pp_out);

pp_in.tp=pp.T_180; pp_in.phi=pi/2;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Ref pulse y (phase = pi/2)
Rtot{4}=calc_rotation_matrix(sp,pp_out);

% Excitation pulses 1 and 2, including timing correction
texc=[pi/2 -1]; aexc=[1 0];
pexc1=[1 0]; pexc2=[2 0];  % pexc is pulse type
acq_exc=[0 0]; gexc=[0 0]; % gexc is the gradient

% Encoding periods 1 and 2
tenc=[(pi/2)*grad(2)/T_90 pi (pi/2)*grad(2)/T_90]; aenc=[0 1 0];
penc1=[0 3 0]; penc2=[0 4 0];
acq_enc=[0 0 1]; genc=[grad(1) 0 0];

% Refocusing cycles
nref=3; % Segments in refocusing cycle
tref=zeros(1,nref*NE); pref1=tref; pref2=tref;
aref=tref; acq_ref=tref; gref=tref;
tfp=(pi/2)*(TE-pp.T_180)/(2*T_90); % Free precession period (normalized)
for i=1:NE
    tref((i-1)*nref+1:i*nref)=[tfp pi tfp];
    pref1((i-1)*nref+1:i*nref)=[0 3 0]; % Pulse type
    pref2((i-1)*nref+1:i*nref)=[0 4 0]; % Pulse type
    aref((i-1)*nref+1:i*nref)=[0 1 0];
    acq_ref((i-1)*nref+1:i*nref)=[0 0 1];
    gref((i-1)*nref+1:i*nref)=[0 0 0]; % Gradient
end

% Create complete pulse sequences
% Common terms
pp.tp=[texc tenc tref]; pp.amp=[aexc aenc aref];
pp.acq=[acq_exc acq_enc acq_ref]; pp.grad=[gexc genc gref];
pp.Rtot=Rtot;

% Variable terms
pul1=[pexc1 penc1 pref1]; % Sequence 1: (y,x)
pul2=[pexc2 penc1 pref1]; % Sequence 2: (-y,x)
pul3=[pexc1 penc2 pref2]; % Sequence 3: (y,y)
pul4=[pexc2 penc2 pref2]; % Sequence 4: (-y,y)

% Run PAP phase cycle: (1-2)
pp.pul=pul1; [~,mrx1,~,~,~,~]=calc_macq_matched_probe_relax3(sp,pp);
pp.pul=pul2; [~,mrx2,~,~,~,~]=calc_macq_matched_probe_relax3(sp,pp);
mrx_x=(mrx1-mrx2); % Apply phase cycle

% Run PAP phase cycle: (3-4)
pp.pul=pul3; [~,mrx3,~,~,~,~]=calc_macq_matched_probe_relax3(sp,pp);
pp.pul=pul4; [~,mrx4,~,~,~,~]=calc_macq_matched_probe_relax3(sp,pp);
mrx_y=(mrx3-mrx4); % Apply phase cycle

if sp.plt_output
    figure; % Figure to show outputs
end
% Calculate time-domain echoes
echo_rx_xy=zeros(NE,nacq);
for i=1:NE
    [echo_rx_x,tvect]=calc_time_domain_echo_arb(mrx_x(i,:),sp.del_w,tacq,tdw,sp.plt_echo);
    [echo_rx_y,~]=calc_time_domain_echo_arb(mrx_y(i,:),sp.del_w,tacq,tdw,sp.plt_echo);
    echo_rx_xy(i,:)=imag(echo_rx_x)+1i*real(echo_rx_y);
    
    if sp.plt_output
        subplot(3,1,1);
        plot((tvect*T_90/(pi/2)+i*TE)*1e3,real(echo_rx_x),'b-'); hold on;
        plot((tvect*T_90/(pi/2)+i*TE)*1e3,imag(echo_rx_x),'r-');
        
        subplot(3,1,2);
        plot((tvect*T_90/(pi/2)+i*TE)*1e3,real(echo_rx_y),'b-'); hold on;
        plot((tvect*T_90/(pi/2)+i*TE)*1e3,imag(echo_rx_y),'r-');
        
        subplot(3,1,3);
        plot((tvect*T_90/(pi/2)+i*TE)*1e3,real(echo_rx_xy(i,:)),'b-'); hold on;
        plot((tvect*T_90/(pi/2)+i*TE)*1e3,imag(echo_rx_xy(i,:)),'r-');
    end
end

if sp.plt_output
    set(gca,'FontSize',14);
    xlabel('Time (ms)'); ylabel('Received echoes');
end

function [pp_out]=calc_pulse_shape(sp,pp,pp_in)

T_90=pp.T_90;
tdeln=(pi/2)*pp_in.tdel/T_90; % Normalized delay
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Add delay to RF pulse to account for ring down, create structure
pp_curr=pp;
pp_curr.tp = [pp_in.tp pp_in.tdel];
pp_curr.phi = [pp_in.phi 0];
pp_curr.amp = [pp_in.amp 0];

% Calculate RF pulse
sp.plt_rx=0; % Turn off plotting
[tvect, Icr, tf1, tf2] = find_coil_current(sp,pp_curr);
pp_out.tf1=tf1; pp_out.tf2=tf2;

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr);
aexc(aexc<amp_zero)=0; % Threshold amplitude

% Remove added delay from RF pulse
pp_out.tp=[texc -tdeln]; pp_out.phi=[pexc 0]; pp_out.amp=[aexc 0];
pp_out.acq=zeros(1,length(texc)+1);