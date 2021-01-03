% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% TE = echo spacing
% Delays follow pulses

function [echo_pk,echo_rms]=cpmg_van_spin_dynamics_norelax(texc,tref,pexc,pref,T_90,NE,T_FP,delt)

delt=delt/1e6;
nexc=length(texc);
nref=length(tref);

tp=zeros(1,nexc+NE*nref);
phi=tp;
tf=tp;

tp(1:nexc)=texc;
phi(1:nexc)=pexc;
tf(1:nexc-1)=zeros(1,nexc-1);

if nexc==1 % Rectangular pulse, include Martin's timing correction
    tf(nexc)=T_FP/2-2*T_90/pi;
else
    tf(nexc)=T_FP/2;
end

for i=1:NE
    tp(nexc+(i-1)*nref+1:nexc+i*nref)=tref;
    phi(nexc+(i-1)*nref+1:nexc+i*nref)=pref;
    tf(nexc+(i-1)*nref+1:nexc+i*nref-1)=zeros(1,nref-1);
    tf(nexc+i*nref)=T_FP;
end
tf(nexc+NE*nref)=T_FP/2;

[echo,tvect]=sim_spin_dynamics_allpw_norelax(T_90,tp,phi,tf);

ind=find(abs(tvect)<delt/2);

echo_pk=max(abs(echo(ind)));
echo_rms=trapz(tvect(ind),abs(echo(ind)).^2)/length(ind);