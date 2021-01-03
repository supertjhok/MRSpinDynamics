% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% TE = echo spacing
% Delays follow pulses

function [echo,tvect,mxyz]=cpmg_van_spin_dynamics_trajectory(exc_name,ref_name,T_90,T_180,NE,T_FP,T1,T2,offs)

close all;
tmp=load('exc_timings.mat');
results=tmp.results;

tmp=size(results);
numres=tmp(1);
for i=1:numres
    if strcmpi(exc_name,results{i,1})
        texc=T_90*results{i,2};
        pexc=results{i,3};
    end
end

tmp=load('ref_timings.mat');
results=tmp.results;

tmp=size(results);
numres=tmp(1);
for i=1:numres
    if strcmpi(ref_name,results{i,1})
        tref=T_180*results{i,2};
        pref=results{i,3};
    end
end

nexc=length(texc);
nref=length(tref);

tp=zeros(1,nexc+NE*nref);
phi=tp;
powdb=tp;
tf=tp;

tp(1:nexc)=texc;
phi(1:nexc)=pexc;
powdb(1:nexc)=20*log10(T_180/(2*T_90));
tf(1:nexc-1)=zeros(1,nexc-1);

if nexc==1 % Assume rectangular pulse
    tf(nexc)=0.5*T_FP-2*T_90/pi; % Martin's timing correction
else % Not a rectangular pulse
    tf(nexc)=0.5*T_FP;
end

for i=1:NE
    tp(nexc+(i-1)*nref+1:nexc+i*nref)=tref;
    phi(nexc+(i-1)*nref+1:nexc+i*nref)=pref;
    powdb(nexc+(i-1)*nref+1:nexc+i*nref)=zeros(1,nref);
    tf(nexc+(i-1)*nref+1:nexc+i*nref-1)=zeros(1,nref-1);
    tf(nexc+i*nref)=T_FP;
end
tf(nexc+NE*nref)=T_FP/2;

[echo,tvect,mxyz]=sim_spin_dynamics_allpw_trajectory(T_180/2,tp,phi,powdb,tf,T1,T2,offs);