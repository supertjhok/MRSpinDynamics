% texc,pexc,aexc = excitation pulse times, phases, amplitudes
% TE = echo spacing
% T_FP = free precession time
% len_acq = acquisition time
% Delays follow pulses

function [minit,trans,masy]=cpmg_van_spin_dynamics_asymp_mag3a(texc,pexc,aexc,neff,del_w,len_acq)

nexc=length(texc)+1;

tp=zeros(1,nexc); phi=tp; amp=tp;
tp(1:nexc-1)=texc;
phi(1:nexc-1)=pexc;
amp(1:nexc-1)=aexc;

if nexc==2 % Rectangular pulse, include Martin's timing correction
    tp(nexc)=-1/aexc(1);
else
    tp(nexc)=0;
end

[minit,trans,masy]=sim_spin_dynamics_asymp_mag3a(tp,phi,amp,neff,del_w,len_acq);