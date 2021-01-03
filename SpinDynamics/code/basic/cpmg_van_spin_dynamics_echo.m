% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% TE = echo spacing
% Delays follow pulses

function [echo,tvect]=cpmg_van_spin_dynamics_echo(texc,tref,pexc,pref,T_90,NE,T_FP,T1,T2)

nexc=length(texc);
nref=length(tref);

tp=zeros(1,nexc+NE*nref);
phi=tp;
tf=tp;

tp(1:nexc)=texc;
phi(1:nexc)=pexc;

if nexc==1
    tf(nexc)=0.5*T_FP-2*T_90/pi; % Correction (for rectangular-rectangular only)
else
    tf(1:nexc-1)=zeros(1,nexc-1);
    tf(nexc)=0.5*T_FP;
end

for i=1:NE
    tp(nexc+(i-1)*nref+1:nexc+i*nref)=tref;
    phi(nexc+(i-1)*nref+1:nexc+i*nref)=pref;
    tf(nexc+(i-1)*nref+1:nexc+i*nref-1)=zeros(1,nref-1);
    tf(nexc+i*nref)=T_FP;
end
tf(nexc+NE*nref)=T_FP/2;

[echo,tvect]=sim_spin_dynamics_allpw(T_90,tp,phi,tf,T1,T2);