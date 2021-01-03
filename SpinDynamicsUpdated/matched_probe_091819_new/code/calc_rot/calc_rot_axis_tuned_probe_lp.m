% Calculate rotation axis of a refocusing cycle including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% params = [tref, pref, aref, tfp, tacq, Rs(Qsw_on,Qsw_off,Tx_on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [n]=calc_rot_axis_tuned_probe_lp(sp,pp)

% Simulation parameters
%del_w=sp.del_w;
%window = sinc(del_w*pp.tacq/(2*pi)); % window function for acquisition
%window=window./sum(window);

T_90=pp.T_90; % Rectangular T_90 time

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

[tvect,Icr] = tuned_probe_lp(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
trefc=delt*ones(1,length(tvect));
prefc=atan2(imag(Icr),real(Icr));
arefc=abs(Icr);

arefc=(arefc-min(arefc))/(max(arefc)-min(arefc)); % Normalize amplitude
arefc(arefc<amp_zero)=0; % Threshold amplitude
if sp.plt_tx
    figure(98);
    plot(tvect*1e6,arefc);
    xlabel('Time (\mus)')
    ylabel('Normalized current amplitude in coil')
end
    
[n]=calc_rot_axis_arba3(trefc,prefc,arefc,sp.del_w,sp.plt_axis);
% masy_matched = conv(abs(n(1,:)+1i*n(2,:)),window,'same'); % Maximum possible asymptotic magnetization
 
% [~,SNR]=tuned_probe_rx(sp,pp,masy_matched); % Filtering by tuned receiver
% SNR=SNR/1e8;