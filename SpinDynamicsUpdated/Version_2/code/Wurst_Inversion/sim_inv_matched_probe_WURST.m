function [ sp ] = sim_inv_matched_probe_WURST( params )
%SIM_INV_MATCHED_PROBE_WURST Summary of this function goes here
%   Detailed explanation goes here
% Read in parameters
% Read in parameters
% NE=params.NE;
% TE=params.TE;
% Tgrad=params.Tgrad;
% rho=params.rho;
% T1map=params.T1map;
% T2map=params.T2map;
% pxz=params.pxz;
% FOV=params.FOV;

% Define pulse system parameters
[sp, pp]=set_params_matched;
T_90=pp.T_90; % Nominal T_90 pulse length
T_180=2*T_90;


% Set size of simulation domain
sp.numptsy=400; 
sp.maxoffs=5; % y-axis

% Predicted maximum number of echoes before simulation instability occurs
% (in the absence of a gradient - gradients will make matters worse)
dw=2*sp.maxoffs/sp.numptsy; % Normalized to w1

% Set plotting parameters
sp.plt_tx = 0;
sp.plt_rx = 0;
sp.plt_sequence = 1; % Plots on/off
sp.plt_axis = 0;
sp.plt_mn = 0;
sp.plt_echo = 0;
sp.plt_output = 1;
sp.plt_fields = 0;

% Set acquisition parameters
tacq=(pi/2)*pp.tacq/T_90; % Normalized acquisition window length
tdw=(pi/2)*pp.tdw/T_90; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points
tvect=linspace(-tacq/2,tacq/2,nacq)'; % Acquisition vector, size: [nacq,1]
isoc=exp(1i*tvect*sp.del_w); % Isochromats (without additional gradients)

% Design impedance matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; 
sp.C2=C2; % Save matching capacitor values

% Create pulse sequence (in normalized time)
% --------------------------------------------
% Pre-calculate all pulses for speed
Rtot={};
pp_in.tp=pp.T_90;
pp_in.tdel=2*pp.T_90;
pp_in.phi=pi/2;
pp_in.amp=1;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Exc pulse y (phase = pi/2)
sp.tf1=pp_out.tf1; 
sp.tf2=pp_out.tf2; % Save Rx transfer functions

sp.w_1 = ones(1,size(sp.del_w,2));
sp.w_1r=sp.w_1; % , assuming single transmit/receive coil

sp.T1 = ones(1,size(sp.del_w,2)).*1000;
sp.T2 = ones(1,size(sp.del_w,2)).*1000;
Rtot{1}=calc_rotation_matrix(sp,pp_out);

% Excitation pulses 1 and 2, including timing correction
texc=[pp.T_90*20 1e-6 pp.T_90*10]; 
aexc=[1 0 0];
pexc1=[1 0 0];
pexc2=[2 0 0];  % pexc is pulse type
acq_exc=[0 0 1];
gexc=[0 0 0]; % gexc is the gradient

% Create complete pulse sequences
% Fixed terms
pp.tp=texc ;
pp.amp=aexc ;
pp.acq=acq_exc ;
pp.grad=gexc;
pp.Rtot=Rtot;

% Variable terms
pul1=pexc1; % Sequence 1: (y,x)

% Calculate gradient strength vectors
% Tgradn=(pi/2)*Tgrad/T_90; % Normalized gradient length
% % Estimate maximum gradient frequency offsets using specified FOV
% wxmax=pi*px^2/(2*FOV(1)*Tgradn); wzmax=pi*pz^2/(2*FOV(2)*Tgradn);
% gradx=wxmax*linspace(-1,1,px); % x-gradient steps
% gradz=wzmax*linspace(-1,1,pz); % z-gradient steps



% Create gradient field
%  spc.del_wg=gradxc(i)*spc.del_wx+gradzc(j)*spc.del_wz;
 sp.del_wg = zeros(1,size(sp.del_w,2));

% Run PAP phase cycle: (1-2)
pp.pul=pul1; 
[~,mrx1]=calc_macq_matched_probe_relax4(sp,pp);
% pp.pul=pul2; 
% [~,mrx2]=calc_macq_matched_probe_relax4(sp,pp);
mrx_x=mrx1; % Apply phase cycle   

figure
plot(abs(mrx_x));
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
[tvect, Icr, tf1, tf2] = find_coil_current_WURST(sp,pp_curr);
% [tvect, Icr, tf1, tf2] = find_coil_current(sp,pp_curr);
pp_out.tf1=tf1; pp_out.tf2=tf2;

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr);
aexc(aexc<amp_zero)=0; % Threshold amplitude

% Remove added delay from RF pulse
pp_out.tp=[texc -tdeln]; pp_out.phi=[pexc 0]; pp_out.amp=[aexc 0];
pp_out.acq=zeros(1,length(texc)+1);

end

