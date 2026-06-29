% Simulate a CPMG sequence with time-varying B0 field.
% Only compute the final echo.
% Supports arbitrary excitation and refocusing pulses.
% ----------------------------------------------------------------------
% Input parameters must include:
% -----------------------------------------------------------------------
% NE -> Number of echoes to simulate
% B_0t -> Time-varying field offset vector (NE points, in T)
% rho, T1map, T2map -> 2D spin density and relaxation time maps (of the sample)
% -----------------------------------------------------------------------
% Sequence: (t_exc)x - [(t_ref)y]^(N_E)
% ----------------------------------------------------------------------
function [mrx,echo_rx,echo_int]=sim_cpmg_ideal_tv_final(sp,pp)

% Read in parameters
NE=pp.NE;
tE=sum(pp.tref); tseq=tE*linspace(0.5,NE-0.5,NE); % Actual time

del_w=sp.del_w; % Original offset frequency vector
Npts=length(del_w); % Total number of simulation points
B_0t=sp.B_0t; % Field offset waveform

% Check size of (x,z) domain: should match sample property matrices
siz=size(sp.rho);
if sp.nx ~= siz(1) || sp.nz ~= siz(2)
    disp('Error: Sample property matrix size does not match!');
    return;
end

% Define time scaling parameter
T_90=pp.T_90; % Nominal T_90 pulse length
w_1n=(pi/2)/T_90; % Nominal nutation frequency

w_0t=(sp.gamma*B_0t/w_1n); % Normalized field offset vector

% Predicted maximum number of echoes before simulation instability occurs
% (in the absence of a gradient - gradients will make matters worse)
dw=2*sp.maxoffs_y/sp.ny; % Normalized to w1
NE_max=4*T_90/(dw*tE);
if NE>NE_max
    disp(['NE_max = ' num2str(NE_max)]);
    disp('Warning: Attempt to simulate beyond NE_max!');
    disp('Accuracy of results is not guaranteed');
end

% Create receiver sensitivity map
sp.w_1r=sp.w_1; % Assuming single transmit/receive coil

% Set acquisition parameters
tacq=w_1n*pp.tacq; % Normalized acquisition window length
tdw=w_1n*pp.tdw; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points
tvect=linspace(-tacq/2,tacq/2,nacq)'; % Acquisition vector, size: [nacq,1]
tvect_rep=repmat(tvect,1,Npts); % Repeated acquisition vector, size: [nacq,npts]
isoc=exp(1i*tvect*del_w); % Isochromats (without field offsets)

% Create pulse sequence (in normalized time)
% --------------------------------------------
% Pre-calculate all pulses for speed
Rtot={}; pp_in=pp;

% Exc pulse (phase = pi/2)
nexc = length(pp.texc); % Segments in refocusing cycle
pp_in.tp=w_1n*pp.texc; pp_in.phi=pp.pexc; pp_in.amp=pp.aexc;
Rtot{1}=calc_rotation_matrix(sp,pp_in);

% Exc pulse (phase = 3*pi/2)
pp_in.tp=w_1n*pp.texc; pp_in.phi=(pp.pexc+pi); pp_in.amp=pp.aexc;
Rtot{2}=calc_rotation_matrix(sp,pp_in);

% Ref pulses (phase = 0)
nref = length(pp.tref); % Segments in refocusing cycle
pp_in.tp=w_1n*pp.tref(2:nref-1); pp_in.phi=pp.pref(2:nref-1);
pp_in.amp=pp.aref(2:nref-1);
for i=1:NE
    sp_in=sp;
    sp_in.del_w=del_w + w_0t(i); % Change field offset vector
    Rtot{i+2}=calc_rotation_matrix(sp_in,pp_in);
end

% Excitation pulses 1 and 2, including timing correction for rectangular
% pulses
if nexc==1 % Rectangular
    texc=[pi/2 w_1n*pp.tcorr]; aexc=[1 0];
    pexc1=[1 0]; pexc2=[2 0];  % pexc is pulse type
    acq_exc=[0 0]; gexc=[0 0]; % gexc is the field offset
else
    texc=[pi/2]; aexc=[1];
    pexc1=[1]; pexc2=[2];  % pexc is pulse type
    acq_exc=[0]; gexc=[0]; % gexc is the field offset
end

% Refocusing cycles
tref=zeros(1,3*NE); pref=tref;
aref=tref; acq_ref=tref; gref=tref;
for i=1:NE
    tref((i-1)*3+1:i*3)=w_1n*[pp.tref(1) pi pp.tref(nref)];
    pref((i-1)*3+1:i*3)=[0 (i+2) 0]; % Pulse type
    aref((i-1)*3+1:i*3)=[0 1 0];
    acq_ref((i-1)*3+1:i*3)=[0 0 0]; % Don't acquire echoes
    gref((i-1)*3+1:i*3)=w_0t(i)*[1 1 1]; % Constant field offset
end
acq_ref((NE-1)*3+1:NE*3)=[0 0 1]; % Only acquire final echo

% Create complete pulse sequences
% Fixed terms
pp.tp=[texc tref]; pp.amp=[aexc aref];
pp.acq=[acq_exc acq_ref]; pp.grad=[gexc gref];
pp.Rtot=Rtot;

% Variable terms
pul1=[pexc1 pref]; % Sequence 1: (y,x)
pul2=[pexc2 pref]; % Sequence 2: (-y,x)

% Estimate spin dynamics: run PAP phase cycle
ppc=pp; % Create local variables
ppc.pul=pul1; mrx1=calc_macq_ideal_probe_relax4(sp,ppc);
ppc.pul=pul2; mrx2=calc_macq_ideal_probe_relax4(sp,ppc);
mrx=(mrx1-mrx2)/2; % Apply phase cycle

% Calculate time-domain echo
% Add field offsets to isochromats before calculating echoes
echo_rx=(exp(1i*w_0t(NE)*tvect_rep).*isoc)*mrx';
echo_int=trapz(tvect,echo_rx); % Estimate echo integrals

if sp.plt_output % Plot selected outputs
    figure(5); % Magnetization
    plot(sp.del_w,real(mrx),'LineWidth',1); hold on;
    plot(sp.del_w,imag(mrx),'LineWidth',1);
    set(gca,'FontSize',14); set(gca,'FontWeight','Bold');
    xlabel('\Delta\omega_0/\omega_{1n}'); ylabel('M_{rx}(\omega)');
    
    figure(6); % Echo
    plot(tvect,real(echo_rx),'LineWidth',1); hold on;
    plot(tvect,imag(echo_rx),'LineWidth',1);
    set(gca,'FontSize',14); set(gca,'FontWeight','Bold');
    xlabel('t_{acq} (norm)'); ylabel('s_{rx}(t)');
    
    figure(7); % Field fluctuation
    stairs(tseq*1e3,w_0t,'LineWidth',1);
    set(gca,'FontSize',14); set(gca,'FontWeight','Bold');
    ylabel('\Delta\omega_{0}(t)/\omega_{1n}');
end