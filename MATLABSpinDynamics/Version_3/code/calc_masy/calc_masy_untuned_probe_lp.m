% CALC_MASY_UNTUNED_PROBE_LP
% Calculate CPMG asymptotic and received magnetization for an untuned probe.
%
% Signature
%   [mrx,masy,SNR] = calc_masy_untuned_probe_lp(params,sp,pp)
%
% Inputs
%   params - Compact pulse/circuit parameter structure. Required fields include
%     texc, pexc, aexc, tref, pref, aref, tfp, tqs, trd, tacq, Rs, and pcycle.
%   sp - Untuned-probe system/simulation structure. Required fields include
%     gamma, sens, del_w, plt_tx, plt_rx, and untuned-probe circuit fields used
%     by untuned_probe_lp and untuned_probe_rx.
%   pp - Pulse-sequence structure. Required fields include T_90, tcorr, and
%     amp_zero.
%
% Outputs
%   mrx - Complex received magnetization spectrum after untuned receiver
%     filtering.
%   masy - Complex asymptotic magnetization spectrum before receiver filtering.
%   SNR - Signal-to-noise ratio estimate scaled to voltage units.
%
% Dependencies
%   calc_rot_axis_untuned_probe_lp, untuned_probe_lp,
%   sim_spin_dynamics_asymp_mag3, untuned_probe_rx.
%
% Notes
%   Pulse times are converted from seconds to normalized units through T_90.
%   The excitation pulse is generated through the untuned-probe circuit model
%   and scaled by coil sensitivity.
% --------------------------------------------------------------

function [mrx,masy,SNR] = calc_masy_untuned_probe_lp(params,sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma);
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Convert acquisition time to normalized time
tacq=(pi/2)*pp.tacq/T_90;

sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off/on plots
%sp.mf_type=2;

% Calculate refocusing axis
%sp.plt_tx=1;
[neff]=calc_rot_axis_untuned_probe_lp(params,sp,pp);

% Create excitation pulse, including delays to allow pulse to ring down *T_90/(pi/2)
pp.tref=[params.texc params.tqs params.trd];
pp.pref=[params.pexc 0 0]; pp.aref=[params.aexc 0 0];
pp.Rsref=[params.Rs(2)*ones(1,length(params.texc)) params.Rs(3) params.Rs(1)];

%sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off/on plots

[tvect,Icr,~,~] = untuned_probe_lp(sp,pp); % Calculate excitation pulse

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude
ind=find(aexc==0); pexc(ind)=0; 

if sp.plt_tx
    figure(99);
    plot(tvect/T_90,aexc);
    xlabel('Normalized time, t/T_{90}')
    ylabel('Normalized current amplitude in coil')
end

% Remove added delays from excitation pulse
if length(params.texc)==1 % Rectangular excitation pulse, use timing correction
	texc=[texc -(params.tqs+params.trd-pp.tcorr)*(pi/2)/T_90];
else % Longer excitation pulse
	texc=[texc -(params.tqs+params.trd)*(pi/2)/T_90];
end
pexc=[pexc 0]; aexc=[aexc 0];

% Calculate spin dynamics
switch params.pcycle
    case 0 % No phase cycle
		[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,sp.del_w,tacq);
    case 1  % Complete PAP phase cycle
		[masy1]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,sp.del_w,tacq);
		[masy2]=sim_spin_dynamics_asymp_mag3(texc,pexc+pi,aexc,neff,sp.del_w,tacq);
        masy=(masy1-masy2)/2;
    case 2  % Complete PI phase cycle
		[masy1]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,sp.del_w,tacq);
		[masy2]=sim_spin_dynamics_asymp_mag3(texc,-pexc,aexc,neff,sp.del_w,tacq);
        masy=(masy1-masy2)/2;
end

[mrx,SNR,~]=untuned_probe_rx(sp,pp,masy); % Filtering by untuned receiver
SNR=SNR/1e8; % SNR in voltage units

if sp.plt_rx
    figure(4);
    plot(sp.del_w,real(masy),'b--'); hold on; plot(sp.del_w,imag(masy),'r--');
    plot(sp.del_w,real(mrx),'b-'); hold on; plot(sp.del_w,imag(mrx),'r-');
    title('Asymptotic magnetization')
    xlabel('\Delta\omega_{0}/\omega_{1,max}')
    ylabel('M_{asy}, M_{rx}')
end
