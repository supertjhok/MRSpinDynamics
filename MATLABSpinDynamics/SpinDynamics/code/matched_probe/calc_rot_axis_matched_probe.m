% Calculate rotation axis of a refocusing cycle including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% 04/09/19: Modified to include absolute RF phase parameter (psi)

function [n]=calc_rot_axis_matched_probe(sp,pp)

% Simulation parameters
T_90=pp.T_90; % Rectangular T_90 time
win=2*pi*sp.fin; % Input frequency

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Create structure
pp_curr=pp;
pp_curr.tp=pp.tref; pp_curr.phi=pp.pref; pp_curr.amp=pp.aref;
texc_tot=sum(pp.texc)+pp.tcorr; % Total length of excitation period
pp_curr.psi=pp.psi+mod(win*texc_tot,2*pi); % Calculate absolute phase at start of refocusing cycle

% Find coil current
[tvect, Icr, ~, ~] = find_coil_current(sp,pp_curr);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
trefc=delt*ones(1,length(tvect));
prefc=atan2(imag(Icr),real(Icr));
arefc=abs(Icr);
arefc(arefc<amp_zero)=0; % Threshold amplitude
    
[n]=calc_rot_axis_arba3(trefc,prefc,arefc,sp.del_w,sp.plt_axis);