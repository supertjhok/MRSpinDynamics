% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% TE = echo spacing
% Delays follow pulses

function [mat]=calc_refocusing_mat_delw1(tref,pref,T_90,del_w1,NE,T_FP)

nref=length(tref);

tp=zeros(1,NE*nref);
phi=tp;
tf=tp;

for i=1:NE
    tp((i-1)*nref+1:i*nref)=tref;
    phi((i-1)*nref+1:i*nref)=pref;
    tf((i-1)*nref+1:i*nref-1)=zeros(1,nref-1);
    tf(i*nref)=T_FP;
end
tf(NE*nref)=T_FP/2;

[mat]=calc_spin_mat_delw1(T_90,del_w1,tp,phi,tf);