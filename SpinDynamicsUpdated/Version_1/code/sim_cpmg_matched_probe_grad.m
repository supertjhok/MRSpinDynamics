% Simulate a CPMG sequence with phase gradient, rectangular pulses, and finite
% transmit and receive bandwidth
% -----------------------------------------------------------------------

% NE -> Number of echoes to simulate
% TE -> Echo spacing (real value)
% T1, T2 -> Relaxation time constants (s)
% grad -> (value, length)
function [echo_rx,tvect,SNR]=sim_cpmg_matched_probe_grad(NE,TE,T1,T2,grad)

% Simulate each echo of a CPMG sequence
[sp, pp]=set_params_matched; % Define system parfclosmeters
T_90=pp.T_90; % Nominal T_90 pulse length

% Set plotting parameters
sp.plt_tx = 0; sp.plt_rx = 0; sp.plt_sequence = 0; % Plots /off
sp.plt_axis = 0; sp.plt_mn = 0; sp.plt_echo = 0;
sp.plt_output = 0;

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

% Create pulse sequence (in real time)
% --------------------------------------------
% Excitation pulse, including timing correction
texc=[1 -2/pi]*pp.T_90;
pexc=[pi/2 0]; aexc=[1 0]; acq_exc=[0 0]; grad_exc=[0 0];

% Encoding period
tenc=[grad(2) pp.T_180 grad(2)];
penc=[0 0 0];
aenc=[0 1 0];
acq_enc=[0 0 1];
grad_enc=[grad(1) 0 0];

% Refocusing cycles
tref=zeros(1,3*NE); pref=tref; aref=tref; acq_ref=tref; grad_ref=tref;
for i=1:NE
    tref((i-1)*3+1:i*3)=[(TE-pp.T_180)/2 pp.T_180 (TE-pp.T_180)/2];
    pref((i-1)*3+1:i*3)=[0 0 0];
    aref((i-1)*3+1:i*3)=[0 1 0];
    acq_ref((i-1)*3+1:i*3)=[0 0 1];
    grad_ref((i-1)*3+1:i*3)=[0 0 0];
end

% Create complete pulse sequence
pp.tp=[texc tenc tref]; pp.phi=[pexc penc pref];
pp.amp=[aexc aenc aref]; pp.acq=[acq_exc acq_enc acq_ref];
pp.grad=[grad_exc grad_enc grad_ref];

% Run PAP phase cycle
[~,mrx1,pnoise,f,~,~]=calc_macq_matched_probe_grad(sp,pp);

pp.phi=[pexc+pi penc pref]; % Change phase of excitation pulse
[~,mrx2,~,~,~,~]=calc_macq_matched_probe_grad(sp,pp);

mrx=(mrx1-mrx2); pnoise=pnoise*sqrt(2); % Apply phase cycle

if sp.plt_output
    figure; % Figure to show outputs
end
% Calculate time-domain echoes
echo_rx=zeros(NE,nacq); SNR=zeros(1,NE);
for i=1:NE
    [echo_rx(i,:),tvect]=calc_time_domain_echo_arb(mrx(i,:),sp.del_w,tacq,tdw,sp.plt_echo);
    
    if sp.plt_output
        plot((tvect*T_90/(pi/2)+i*TE)*1e3,real(echo_rx(i,:)),'b-'); hold on;
        plot((tvect*T_90/(pi/2)+i*TE)*1e3,imag(echo_rx(i,:)),'r-');
    end
    
    SNR(i)=matched_probe_rx_snr(sp,mrx(i,:),pnoise,f);
end

if sp.plt_output
    set(gca,'FontSize',14);
    xlabel('Time (ms)'); ylabel('Received echoes');
end