% tref,pref,aref = refocusing pulse times, phases, amplitudes
% NE = number of echoes
% TE = echo spacing
% Delays follow pulses

function [mat,del_w]=calc_refocusing_mat_arba(tref,pref,aref,NE,T_FP)

% Single refocusing cycle = (FP/2 - Pulse - FP/2)
nref=length(tref)+2;

tp=zeros(1,nref);
phi=tp; amp=tp;

tp(1)=T_FP/2; tp(nref)=tp(1);
tp(2:nref-1)=tref;
phi(2:nref-1)=pref;
amp(2:nref-1)=aref;

% Single refocusing cycle
[mat_single,del_w]=calc_spin_mat_arba(tp,phi,amp);

% NE refocusing cycles
numpts=length(del_w);

mat=mat_single;
for i=2:NE
    for k=1:numpts
        mat(:,:,k)=mat_single(:,:,k)*mat(:,:,k);
    end
end