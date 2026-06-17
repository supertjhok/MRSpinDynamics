% Calculate asymptotic magnetization of CPMG including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% params = [texc, pexc, aexc, tref, pref, aref, tfp, tacq, Rs(Qsw_on,Qsw_off,Tx_on), pcycle]
% (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [mrx,SNR,SNR_max] = calc_masy_tuned_probe_lp_orig(params,sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma);
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

%sp.plt_axis=0;  sp.plt_tx=1; sp.plt_rx=1; % Turn off/on plots
%sp.mf_type=2;

% Calculate refocusing axis
%sp.plt_tx=99;
[neff,SNR_max]=calc_rot_axis_tuned_probe_lp_Orig(params,sp,pp);
params.neff=neff;

% Create excitation pulse, including delays to allow pulse to ring down *T_90/(pi/2)
pp.tref=[params.texc params.tqs params.trd];
pp.pref=[params.pexc 0 0]; pp.aref=[params.aexc 0 0];
pp.Rsref=[params.Rs(3)*ones(1,length(params.texc)) params.Rs(2) params.Rs(1)];

sp.plt_axis=0;  sp.plt_tx=2; sp.plt_rx=1; % Turn off/on plots

[tvect,Icr] = tuned_probe_lp(sp,pp); % Calculate excitation pulse

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude

if sp.plt_tx
    figure(12);
    plot(tvect*1e6,aexc);
    xlabel('Time (\mus)')
    ylabel('Normalized current amplitude in coil')
end

% Remove added delays from excitation pulse
if length(params.texc)==1 % Rectangular excitation pulse, use Martin's timing correction
	texc=[texc -(params.tqs+params.trd+2*params.texc/pi)];
else % Longer excitation pulse
	texc=[texc -(params.tqs+params.trd)];
end
pexc=[pexc 0]; aexc=[aexc 0];

% Calculate spin dynamics
switch params.pcycle
    case 0 % No phase cycle
		[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,params.tacq);
    case 1  % Complete PAP phase cycle
		[masy1]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,params.tacq);
		[masy2]=sim_spin_dynamics_asymp_mag3(texc,pexc+pi,aexc,params.neff,sp.del_w,params.tacq);
        masy=(masy1-masy2)/2;
    case 2  % Complete PI phase cycle
		[masy1]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,params.tacq);
		[masy2]=sim_spin_dynamics_asymp_mag3(texc,-pexc,aexc,params.neff,sp.del_w,params.tacq);
        masy=(masy1-masy2)/2;
end

[mrx,SNR]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver
SNR=SNR/1e8; % SNR in voltage units

if sp.plt_rx
    figure(4);
    plot(sp.del_w,real(masy),'b--'); hold on; plot(sp.del_w,imag(masy),'r--');
    plot(sp.del_w,real(mrx),'b-'); hold on; plot(sp.del_w,imag(mrx),'r-');
    title('Asymptotic magnetization')
    xlabel('\Delta\omega_{0}/\omega_{1,max}')
    ylabel('M_{asy}, M_{rx}')
end