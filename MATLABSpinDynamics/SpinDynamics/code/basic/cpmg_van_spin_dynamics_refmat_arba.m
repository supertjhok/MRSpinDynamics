% texc,pexc,aexc = excitation pulse times, phases, amplitudes
% TE = echo spacing
% T_FP = free precession time
% delt = integration time
% Delays follow pulses

function [echo,tvect,echo_pk,echo_rms]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,delt,len_acq)

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

[echo,tvect]=sim_spin_dynamics_refmat_arba(tp,phi,amp,refmat,del_w,len_acq);

ind=find(abs(tvect)<delt/2);

echo_pk=max(abs(echo(ind)));
echo_rms=sqrt(trapz(tvect(ind),abs(echo(ind)).^2)); %/length(ind);