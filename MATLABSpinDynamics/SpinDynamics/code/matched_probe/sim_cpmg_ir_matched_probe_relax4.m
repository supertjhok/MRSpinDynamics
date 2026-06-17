% Simulate a CPMG-IR sequence with rectangular pulses and finite
% transmit and receive bandwidth
% -------------------------------------------------------------------
% Precompute RF pulses and isochromats for speed

% NE -> Number of echoes to simulate
% TE -> Echo spacing (real value)
% tauvect -> Initial encoding vector (s)
% T1, T2 -> Relaxation time constants (s)
function [echo_int_all]=sim_cpmg_ir_matched_probe_relax4(NE,TE,tauvect,T1,T2)

% Simulate each echo of a CPMG sequence
[sp, pp]=set_params_matched; % Define system parameters
T_90=pp.T_90; % Nominal T_90 pulse length

% Set plotting parameters
sp.plt_tx = 0; sp.plt_rx = 0; sp.plt_sequence = 0; % Plots /off
sp.plt_axis = 0; sp.plt_mn = 0; sp.plt_echo = 0;

% Design matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Create (w0, w1) and coil sensitivity maps
numpts=sp.numpts; maxoffs=10;
sp.del_w=linspace(-maxoffs,maxoffs,numpts); % Linear gradient
sp.del_wg=zeros(1,numpts); % No additional gradient
sp.w_1=ones(1,numpts); % Uniform transmit w_1
sp.w_1r=ones(1,numpts); % Uniform receiver sensitivity

% Set sample parameters
sp.m0=ones(1,numpts); sp.mth=ones(1,numpts); % Spin density
sp.T1=T1*ones(1,numpts); sp.T2=T2*ones(1,numpts); % Relaxation constants

% Set acquisition parameters
tacq=(pi/2)*pp.tacq/T_90; % Normalized acquisition window length
tdw=(pi/2)*pp.tdw/T_90; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points
tvect=linspace(-tacq/2,tacq/2,nacq)'; % Acquisition vector, size: [nacq,1]
isoc=exp(1i*tvect*sp.del_w); % Isochromats (without additional gradients)

% Create pulse sequence (in normalized time)
% --------------------------------------------
% Pre-calculate all pulses for speed
Rtot={};
pp_in.tp=pp.T_90; pp_in.tdel=2*pp.T_90; pp_in.phi=pi/2; pp_in.amp=1;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Exc pulse 1 (phase = pi/2)
sp.tf1=pp_out.tf1; sp.tf2=pp_out.tf2; % Save Rx transfer functions
Rtot{1}=calc_rotation_matrix(sp,pp_out);

pp_in.phi=3*pi/2;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Exc pulse 2 (phase = 3*pi/2)
Rtot{2}=calc_rotation_matrix(sp,pp_out);

pp_in.tp=pp.T_180; pp_in.phi=0;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Ref pulse (phase = 0)
Rtot{3}=calc_rotation_matrix(sp,pp_out);

% Excitation pulse 1, including timing correction
texc=[pi/2 -1]; aexc=[1 0];
pexc1=[1 0]; % pexc is pulse type
acq_exc=[0 0]; gexc=[0 0]; % gexc is the gradient

%  Excitation pulse 2, including timing correction
pexc2=[2 0]; % pexc is pulse type

% Refocusing cycles
nref=3; % Segments in refocusing cycle
tref=zeros(1,nref*NE); pref=tref; aref=tref; acq_ref=tref; gref=tref;
tfp=(pi/2)*(TE-pp.T_180)/(2*T_90); % Free precession period (normalized)
for i=1:NE
    tref((i-1)*nref+1:i*nref)=[tfp pi tfp];
    pref((i-1)*nref+1:i*nref)=[0 3 0]; % Pulse type
    aref((i-1)*nref+1:i*nref)=[0 1 0];
    acq_ref((i-1)*nref+1:i*nref)=[0 0 1];
    gref((i-1)*nref+1:i*nref)=[0 0 0]; % Gradient
end

% Encoding period (pi pulse and delay)
tenc=[pi (pi/2)*tauvect(1)/T_90];
penc=[3 0]; aenc=[1 0];
genc=[0 0]; acq_enc=[0 0];
ind_enc=length(tenc); % Index of encoding period

% Assume PAP phase cycle
% Create complete pulse sequence 1
pp1=pp;
pp1.tp=[tenc texc tref]; pp1.amp=[aenc aexc aref];
pp1.pul=[penc pexc1 pref];
pp1.acq=[acq_enc acq_exc acq_ref]; pp1.grad=[genc gexc gref];
pp1.Rtot=Rtot;

% Create complete pulse sequence 2
pp2=pp1;
pp2.pul=[penc pexc2 pref];

% Create pulse sequence (in normalized time)
% --------------------------------------------
ntau=length(tauvect); % Number of encoding steps
echo_int_all=zeros(ntau,NE); % Output echo integrals
parfor j=1:ntau % Parallelize for speed
    ppc1=pp1; ppc2=pp2; % Local variables
    
    % Set encoding period
    tp=ppc1.tp; tp(ind_enc)=(pi/2)*tauvect(j)/T_90;
    ppc1.tp=tp; ppc2.tp=tp;
    
    % Run PAP phase cycle
    [~,mrx1]=calc_macq_matched_probe_relax4(sp,ppc1);
    [~,mrx2]=calc_macq_matched_probe_relax4(sp,ppc2);
    mrx=(mrx2-mrx1); % Apply phase cycle
    
    % Calculate time-domain echoes
    % Can't plot inside a parfor loop
    
    echo_rx=isoc*mrx'; % Size: [nacq,NE]
    echo_int_all(j,:)=trapz(tvect,echo_rx); % Estimate echo integrals, size: [1,NE]
end

% Plot results
figure;
subplot(1,2,1); imagesc(TE*linspace(1,NE,NE)*1e3,tauvect*1e3,real(echo_int_all)); colorbar;
set(gca,'FontSize',14); xlabel('Relaxation time (ms)'); ylabel('Encoding time (ms)');
title('Real');
subplot(1,2,2); imagesc(TE*linspace(1,NE,NE)*1e3,tauvect*1e3,imag(echo_int_all)); colorbar;
set(gca,'FontSize',14); xlabel('Relaxation time (ms)'); ylabel('Encoding time (ms)');
title('Imag');
set(gca,'FontSize',14); xlabel('Relaxation time (ms)'); ylabel('Encoding time (ms)');

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