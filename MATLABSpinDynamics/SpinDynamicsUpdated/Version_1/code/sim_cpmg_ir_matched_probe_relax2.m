% Simulate a CPMG-IR sequence with rectangular pulses and finite
% transmit and receive bandwidth
% -------------------------------------------------------------------
% Precompute excitation and refocusing pulses for speed

% NE -> Number of echoes to simulate
% TE -> Echo spacing (real value)
% tauvect -> Initial encoding vector (s)
% T1, T2 -> Relaxation time constants (s)
function [echo_int_all]=sim_cpmg_ir_matched_probe_relax2(NE,TE,tauvect,T1,T2)

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
pp_in.tp=pp.T_90; pp_in.tdel=2*pp.T_90; pp_in.phi=pi/2; pp_in.amp=1;
pp_out1=calc_pulse_shape(sp,pp,pp_in); % Exc pulse 1 (phase = pi/2)
sp.tf1=pp_out1.tf1; sp.tf2=pp_out1.tf2;

pp_in.phi=3*pi/2;
pp_out2=calc_pulse_shape(sp,pp,pp_in); % Exc pulse 2 (phase = 3*pi/2)

pp_in.tp=pp.T_180; pp_in.phi=0;
pp_out3=calc_pulse_shape(sp,pp,pp_in); % Ref pulse (phase = 0)

% Excitation pulse 1, including timing correction
texc1=[pp_out1.tp -1];
pexc1=[pp_out1.phi 0]; aexc1=[pp_out1.amp 0]; acq_exc=[pp_out1.acq 0];

%  Excitation pulse 2, including timing correction
texc2=[pp_out2.tp -1];
pexc2=[pp_out2.phi 0]; aexc2=[pp_out2.amp 0];

% Refocusing cycles
nref=2+length(pp_out3.tp); % Segments in refocusing cycle
tref=zeros(1,nref*NE); pref=tref; aref=tref; acq_ref=tref;
tfp=(pi/2)*(TE-pp.T_180)/(2*T_90); % Free precession period (normalized)
for i=1:NE
    tref((i-1)*nref+1:i*nref)=[tfp pp_out3.tp tfp];
    pref((i-1)*nref+1:i*nref)=[0 pp_out3.phi 0];
    aref((i-1)*nref+1:i*nref)=[0 pp_out3.amp 0];
    acq_ref((i-1)*nref+1:i*nref)=[0 pp_out3.acq 1];
end

% Encoding period (pi pulse and delay)
tenc=[pp_out3.tp (pi/2)*tauvect(1)/T_90];
penc=[pp_out3.phi 0]; aenc=[pp_out3.amp 0]; acq_enc=[pp_out3.acq 0];
ind_enc=length(tenc); % Index of encoding period

% Assume PAP phase cycle
% Create complete pulse sequence 1
pp1=pp;
pp1.tp=[tenc texc1 tref]; pp1.phi=[penc pexc1 pref];
pp1.amp=[aenc aexc1 aref]; pp1.acq=[acq_enc acq_exc acq_ref];

% Create complete pulse sequence 2
pp2=pp;
pp2.tp=[tenc texc2 tref]; pp2.phi=[penc pexc2 pref];
pp2.amp=[aenc aexc2 aref]; pp2.acq=[acq_enc acq_exc acq_ref];

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
    [~,mrx1,~,~,~,~]=calc_macq_matched_probe_relax2(sp,ppc1);
    [~,mrx2,~,~,~,~]=calc_macq_matched_probe_relax2(sp,ppc2);
    mrx=(mrx1-mrx2); % Apply phase cycle
    
    % Calculate time-domain echoes
    % Can't plot inside a parfor loop
    echo_rx=zeros(NE,nacq); echo_int=zeros(1,NE);
    for i=1:NE
        [echo_rx(i,:),tvect]=calc_time_domain_echo_arb(mrx(i,:),sp.del_w,tacq,tdw,sp.plt_echo);
        
        % Calculate echo integral
        echo_int(i)=trapz(tvect,echo_rx(i,:));
    end
    
    echo_int_all(j,:)=echo_int;
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