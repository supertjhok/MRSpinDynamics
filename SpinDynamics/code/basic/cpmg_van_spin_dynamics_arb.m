% General CPMG sequence
% ------------------------------------------------------------------------
% texc,pexc,aexc,oexc = excitation pulse times, phases, amplitudes, offsets
% tref,pref,aref,oref = refocusing pulse times, phases, amplitudes, offsets
% NE = number of echoes
% TE = echo spacing
% ------------------------------------------------------------------------

function [macq,del_w0]=...
    cpmg_van_spin_dynamics_arb(texc,pexc,aexc,oexc,tref,pref,aref,oref,NE,len_acq,T1,T2)

% Create pulse sequence
% ----------------------------------------------------------------------
nexc=length(texc)+1;
nref=length(tref);

tp=zeros(1,nexc+NE*nref); phi=tp;
amp=tp; offs=tp; acq=tp;

tp(1:nexc-1)=texc;
phi(1:nexc-1)=pexc;
amp(1:nexc-1)=aexc;
offs(1:nexc-1)=oexc;

if nexc<4 % Rectangular excitation pulse, include Martin's timing correction
    tp(nexc)=-1/aexc(1);
else
    tp(nexc)=0;
end

for i=1:NE
    tmp=nexc+(i-1)*nref+1:nexc+i*nref;
    tp(tmp)=tref;
    phi(tmp)=pref;
    amp(tmp)=aref;
    offs(tmp)=oref;
    %acq(nexc+i*nref)=1; % Acquire every echo
end
acq(nexc+NE*nref)=1; % Acquire last echo only

% Current code - specify arbitrary (w0,w1) map
% ----------------------------------------------------------------------
numpts=1e4+1; % Number of spin vectors to simulate

maxoffs=20;  % Maximum static offset
del_w0=linspace(-maxoffs,maxoffs,numpts); % Uniform static field gradient

max_w1=0; % Maximum RF inhomogeneity
w_1=linspace(1-max_w1,1+max_w1,numpts); % Uniform RF field gradient

[macq]=sim_spin_dynamics_arb5(tp,phi,amp,offs,acq,len_acq,T1,T2,del_w0,w_1);

% Old code - linear w0, constant w1
% ----------------------------------------------------------------------
% Exact relaxation during pulses with matrix exponential -> very slow 
% [macq,del_w]=sim_spin_dynamics_arb(tp,phi,amp,offs,acq,len_acq,T1,T2);

% Approximate relaxation during pulses -> much faster
%[macq,del_w]=sim_spin_dynamics_arb3(tp,phi,amp,offs,acq,len_acq,T1,T2);

% Rodrigues' rotation formula, approximate relaxation during pulses -> even faster
% [macq,del_w]=sim_spin_dynamics_arb4(tp,phi,amp,offs,acq,len_acq,T1,T2);
% ----------------------------------------------------------------------