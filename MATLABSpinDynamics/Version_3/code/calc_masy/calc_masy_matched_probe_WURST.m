% Calculate asymptotic magnetization of CPMG including transmitter and
% receiver bandwidth effects for a tuned-and-matched probe
% --------------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% 04/09/19: Modified to include absolute RF phase parameter (psi)

function [mrx,echo,tvect,SNR]=calc_masy_matched_probe_WURST(sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Convert acquisition time to normalized time
tacq=(pi/2)*pp.tacq/T_90;

% Design matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Calculate refocusing axis
[neff]=calc_rot_axis_matched_probe_WURST(sp,pp);

% Add delay to excitation pulse to account for ring down, create structure
pp_curr=pp; 
tdel=2*T_90; % Added delay
pp_curr.tp=[pp.texc tdel];
pp_curr.phi=[pp.pexc 0];
pp_curr.amp=[pp.aexc 0];

% Calculate excitation pulse
sp.plt_rx=0; 
[tvect, Icr, tf1, tf2] = find_coil_current_WURST(sp,pp_curr); 

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect)); 
tdeln=(pi/2)*tdel/T_90; 
tcorrn=(pi/2)*pp.tcorr/T_90;
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr);
aexc(aexc<amp_zero)=0; % Threshold amplitude

% Remove added delay from excitation pulse and add timing correction
texc=[texc -tdeln tcorrn]; 
pexc=[pexc 0 0]; 
aexc=[aexc 0 0];

% Calculate spin dynamics with PAP phase cycle
[masy1]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,sp.del_w,tacq);
[masy2]=sim_spin_dynamics_asymp_mag3(texc,pexc+pi,aexc,neff,sp.del_w,tacq);
masy=(masy1-masy2)/2;

% Filtering by the receiver
sp.plt_rx=0; 

[mrx,echo,tvect,SNR] = matched_probe_rx(sp,pp,masy,tf1,tf2);






