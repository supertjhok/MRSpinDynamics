% Calculate rotation axis of a refocusing cycle including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% params = [tref, pref, aref, tfp, tacq, Rs(Qsw_on,Qsw_off,Tx_on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [neff,SNR]=calc_rot_axis_tuned_probe_lp_Orig(params,sp,pp)

% Simulation parameters
del_w=sp.del_w;
window = sinc(del_w*params.tacq/(2*pi)); % window function for acquisition
window=window./sum(window);

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma);
sens=sp.sens; % Coil sensitivity, T/A

% Create refocusing cycle
% pp.tref=[params.tfp params.tref params.tqs (params.tfp-params.tqs)]*T_90/(pi/2);
pp.tref=[params.tfp params.tref params.tqs (params.tfp-params.tqs)];
pp.pref=[0 params.pref 0 0]; pp.aref=[0 params.aref 0 0]; 
pp.Rsref=[params.Rs(1) params.Rs(2)*ones(1,length(params.tref)) params.Rs(3) params.Rs(1)];

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

%sp.plt_tx=2;
[tvect,Icr,~,~] = tuned_probe_lp_Orig(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
trefc=delt*ones(1,length(tvect));
prefc=atan2(imag(Icr),real(Icr));
arefc=abs(Icr)*sens/B1max;
arefc(arefc<amp_zero)=0; % Threshold amplitude
ind=find(arefc==0); prefc(ind)=0; 

if sp.plt_tx
    figure(98);
    plot(tvect/T_90,arefc);
    xlabel('Normalized time, t/T_{90}')
    ylabel('Normalized current amplitude in coil')
end
   
[neff]=calc_rot_axis_arba3(trefc,prefc,arefc,sp.del_w,sp.plt_axis);
masy_matched = conv(abs(neff(1,:)+1i*neff(2,:)),window,'same'); % Maximum possible asymptotic magnetization

[~,SNR]=tuned_probe_rx(sp,pp,masy_matched); % Filtering by tuned receiver
SNR=SNR/1e8;