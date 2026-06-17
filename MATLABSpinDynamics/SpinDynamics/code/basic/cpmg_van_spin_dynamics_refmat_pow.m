% texc,pexc = excitation pulse times, phases
% TE = echo spacing
% T_FP = free precession time
% delt = integration time
% Delays follow pulses

function [echo_pk,echo_rms]=cpmg_van_spin_dynamics_refmat_pow(texc,pexc,T_90,T_180,T_FP,refmat,delt)

delt=delt/1e6;
nexc=length(texc);

tf=zeros(1,nexc);
if nexc==1 % Rectangular pulse, include Martin's timing correction
    tf(nexc)=T_FP/2-2*T_90/pi;
else
    tf(nexc)=T_FP/2;
end

[echo,tvect]=sim_spin_dynamics_refmat_pow(T_90,T_180,texc,pexc,tf,refmat);

ind=find(abs(tvect)<delt/2);

echo_pk=max(abs(echo(ind)));
echo_rms=trapz(tvect(ind),abs(echo(ind)).^2)/length(ind);