% Calculate asymptotic magnetization of CPMG including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% params = [texc, pexc, aexc, tref, pref, aref, tfp, tacq, Rs(Qsw_on,Qsw_off,Tx_on), pcycle]
% (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [mrx,masy,SNR,vsig] = calc_masy_tuned_probe_lp(sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Convert acquisition time to normalized time
tacq=(pi/2)*pp.tacq/T_90;
% Calculate refocusing axis
[neff]=calc_rot_axis_tuned_probe_lp(sp,pp);

%pp_exc
pp_exc = pp;
tdel=2*T_90; % Added delay
pp_exc.tref=[pp.texc tdel]; 
pp_exc.pref=[pp.pexc 0]; 
pp_exc.aref=[pp.aexc 0];
% pp_exc.Rsref=sp.R;

[tvect,Icr] = tuned_probe_lp(sp,pp_exc); % Calculate excitation pulse

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
tdeln=(pi/2)*tdel/T_90; 
tcorrn=(pi/2)*pp.tcorr/T_90;
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr);

if sp.plt_tx
    figure(12);
    plot(tvect*1e6,aexc);
    xlabel('Time (\mus)')
    ylabel('Normalized current amplitude in coil')
end

aexc = (aexc - 0) / ( max(aexc(50:100)) - 0);
% aexc(aexc>1)=1;
aexc(aexc<amp_zero)=0; % Threshold amplitude

% Remove added delay from excitation pulse and add timing correction
texc=[texc -tdeln tcorrn]; 
pexc=[pexc 0 0]; 
aexc=[aexc 0 0];

% Calculate spin dynamics
switch pp.pcycle
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

[mrx,SNR,vsig]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver
SNR=SNR/1e8; % SNR in voltage units

if sp.plt_rx
    figure(4);
%   plot(sp.del_w,real(masy),'b--'); hold on; plot(sp.del_w,imag(masy),'r--');
    plot(sp.del_w,real(mrx)); hold on; 
    plot(sp.del_w,imag(mrx));
    title('Asymptotic magnetization')
    xlabel('\Delta\omega_{0}/\omega_{1,max}')
    ylabel('M_{asy}, M_{rx}')
    
end

% [echo_rx,tvect]=calc_time_domain_echo(mrx,sp.del_w,1);