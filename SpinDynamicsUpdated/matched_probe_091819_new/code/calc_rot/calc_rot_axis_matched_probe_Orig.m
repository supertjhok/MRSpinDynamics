% Calculate rotation axis of a refocusing cycle including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% 04/09/19: Modified to include absolute RF phase parameter (psi)

function [neff,SNR]=calc_rot_axis_matched_probe_Orig(sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
win=2*pi*sp.fin; % Input frequency

% Convert acquisition time to normalized time
tacq=(pi/2)*pp.tacq/T_90;

% Simulation parameters
del_w=sp.del_w;
window = sinc(del_w*tacq/(2*pi)); % window function for acquisition
window=window./sum(window);

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Create structure
pp_curr=pp;
pp_curr.tp=pp.tref;
pp_curr.phi=pp.pref;
pp_curr.amp=pp.aref;
texc_tot=sum(pp.texc)+pp.tcorr; % Total length of excitation period
pp_curr.psi=pp.psi+mod(win*texc_tot,2*pi); % Calculate absolute phase at start of refocusing cycle

% Find coil current
[tvect, Icr, ~, ~] = find_coil_current(sp,pp_curr);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
trefc=delt*ones(1,length(tvect));
prefc=atan2(imag(Icr),real(Icr));
arefc=abs(Icr);
arefc(arefc<amp_zero)=0; % Threshold amplitude
    
[neff]=calc_rot_axis_arba3(trefc,prefc,arefc,sp.del_w,sp.plt_axis);
masy_matched = conv(neff(1,:)+1i*neff(2,:),window,'same'); % Maximum possible asymptotic magnetization

[~,~,~,SNR]=matched_probe_rx(sp,pp,masy_matched,sp.tf1,sp.tf2); % Filtering by matched receiver