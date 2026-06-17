% texc,pexc,aexc = excitation pulse times, phases, amplitudes
% TE = echo spacing
% T_FP = free precession time
% len_acq = acquisition time
% tcorr = timing correction for rectangular pulses (normalized value)
% Delays follow pulses

function [masy]=cpmg_van_spin_dynamics_asymp_mag4(texc,pexc,aexc,neff,del_w,len_acq,tcorr)

if tcorr > 0 % Rectangular excitation pulse, use Martin's timing correction
    nexc=length(texc)+1;
    
    tp=zeros(1,nexc); phi=tp; amp=tp;
    tp(1:nexc-1)=texc; tp(nexc)=-tcorr;
    phi(1:nexc-1)=pexc;
    amp(1:nexc-1)=aexc;
    [masy]=sim_spin_dynamics_asymp_mag3(tp,phi,amp,neff,del_w,len_acq);
else % No timing correction
    [masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,del_w,len_acq);
end

