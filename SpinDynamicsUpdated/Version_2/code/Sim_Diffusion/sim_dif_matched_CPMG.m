% Simulate a basic CPMG sequence with rectangular pulses and finite
% transmit and receive bandwidth
% -------------------------------------------------------------------

% NE -> Number of echoes to simulate
% TE -> Echo spacing (real value)
% T1, T2 -> Relaxation time constants (s)
function [echo_rx,tvect]=sim_dif_matched_CPMG(NE,TE,T1,T2,dz,Delta,T_90,Q)

% Simulate each echo of a CPMG sequence
[sp, pp]=set_params_matched; % Define system parameters
pp.T_90 = T_90;
T_180 = T_90*2;
pp.T_180 = T_180;
pp.tacq = TE/2;
sp.Q =Q;

% Design matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Create (w0, w1) and coil sensitivity maps
numpts=sp.numpts; 
sp.w_1=ones(1,numpts); % Uniform transmit w_1
w1=pi/(2*T_90);
maxoffs=sp.gamma*sp.grad*dz./w1;

sp.del_w=linspace(-maxoffs,maxoffs,numpts); % Linear gradient

sp.w_1r=ones(1,numpts); % Uniform receiver sensitivity

% Set sample parameters
sp.m0=ones(1,numpts); 
sp.mth=ones(1,numpts); % Spin density
sp.T1=T1*ones(1,numpts);
sp.T2=T2*ones(1,numpts); % Relaxation constants
sp.Delta = Delta;

% Set acquisition parameters
tacq=(pi/2)*pp.tacq/T_90; % Normalized acquisition window length
tdw=(pi/2)*pp.tdw/T_90; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points

% Set plotting parameters
sp.plt_tx = 0; sp.plt_rx = 0; sp.plt_sequence = 1; % Plots /off
sp.plt_axis = 0; sp.plt_mn = 0; sp.plt_echo = 0;

% Create pulse sequence (in real time)
% --------------------------------------------
% Excitation pulse, including timing correction
texc=[1 -2/pi]*pp.T_90; 
pexc=[pi/2 0];
aexc=[1 0]; 
acq_exc=[0 0];

%Ecoding pulse/period
%tenc = [(Delta-0.5*T_90-0.5*T_180) T_180 (Delta-0.5*T_90-0.5*T_180) 0.5*(TE-pp.T_180)];
tenc = [(Delta-0.5*T_90-0.5*T_180) T_180 (Delta-0.5*T_90-0.5*T_180)];
penc = [0 0 0];
aenc = [0 1 0];
acq_enc = [0 0 0];
%penc = [0 0 0 0];
%aenc = [0 1 0 0];
%acq_enc = [0 0 0 0];


% Refocusing cycles
numSegments = 3;
tref=zeros(1,numSegments*NE); 
pref=tref;
aref=tref;
acq_ref=tref;


acqDelay = (TE-pp.T_180-pp.tacq)/2;

for i=1:NE
    tref((i-1)*3+1:i*3)=[(TE-pp.T_180)/2 pp.T_180 (TE-pp.T_180)/2];
    %tref((i-1)*numSegments+1:i*numSegments)=[pp.T_180 acqDelay pp.tacq acqDelay];
    %pref((i-1)*numSegments+1:i*numSegments)=[0 0 0 0];
    %aref((i-1)*numSegments+1:i*numSegments)=[1 0 0 0];
    %acq_ref((i-1)*numSegments+1:i*numSegments)=[0 0 1 0];
    pref((i-1)*numSegments+1:i*numSegments)=[0 0 0];
    aref((i-1)*numSegments+1:i*numSegments)=[0 1 0];
    acq_ref((i-1)*numSegments+1:i*numSegments)=[0 0 1];
end

% Create complete pulse sequence
pp.tp=[texc tenc tref]; 
pp.phi=[pexc penc pref];
pp.amp=[aexc aenc aref];
pp.acq=[acq_exc acq_enc acq_ref];
pp.pul = pp.tp;
% Run PAP phase cycle
[~,mrx1,~,~,~,~]=calc_macq_matched_probe_relax_diff(sp,pp);

pp.phi=[pexc+pi penc pref]; % Change phase of excitation pulse
[~,mrx2,~,~,~,~]=calc_macq_matched_probe_relax_diff(sp,pp);

mrx=(mrx1-mrx2); % Apply phase cycle
%mrx = mrx1;
figure; % Figure to show outputs
% Calculate time-domain echoes
echo_rx=zeros(NE,nacq); SNR=zeros(1,NE);
for i=1:NE
    [echo_rx(i,:),tvect]=calc_time_domain_echo_arb(mrx(i,:),sp.del_w,tacq,tdw,sp.plt_echo);
    plot((tvect*T_90/(pi/2)+i*TE)*1e3,real(echo_rx(i,:)),'b-'); hold on;
    plot((tvect*T_90/(pi/2)+i*TE)*1e3,imag(echo_rx(i,:)),'r-');
    
end

set(gca,'FontSize',14);
xlabel('Time (ms)'); ylabel('Received echoes');