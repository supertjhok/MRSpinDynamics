% Echo shape with optimal excitation pulse (all initial magnetization
% vectors aligned with effective rotation axes of the refocusing pulse)
% ------------------------------------------
% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% TE = echo spacing
% Delays follow pulses

function [echo,tvect]=cpmg_van_spin_dynamics_opt(tref,pref,T_90,NE,T_FP,T1,T2)

nref=length(tref);

tp=zeros(1,1+NE*nref);
phi=tp;
tf=tp;

tp(1)=0; % Fake excitation pulse to keep simulator happy

tf(1)=T_FP/2;
for i=1:NE
    tp((i-1)*nref+2:i*nref+1)=tref;
    phi((i-1)*nref+2:i*nref+1)=pref;
    tf((i-1)*nref+2:i*nref)=zeros(1,nref-1);
    tf(i*nref+1)=T_FP;
end
tf(NE*nref+1)=T_FP/2;

if nref>1
    minit=calc_rot_axis(T_90,tref,pref,[T_FP/2,zeros(1,nref-1),T_FP/2]);
else
    minit=calc_rot_axis(T_90,tref,pref,[T_FP/2,T_FP/2]);
end
[echo,tvect]=sim_spin_dynamics_allpw_arbi(minit,T_90,tp,phi,tf,T1,T2);