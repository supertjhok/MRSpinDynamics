% General CPMG sequence with variations between refocusing pulses
% ------------------------------------------------------------------------
% texc,pexc,aexc = excitation pulse times, phases, amplitudes
% tref,pref,aref = refocusing pulse times, phases, amplitudes
% NE = number of echoes
% TE = echo spacing
% ------------------------------------------------------------------------

function [macq,del_w0]=...
    cpmg_van_spin_dynamics_arb_pulsevar(texc,pexc,aexc,tref,pref,aref,NE,len_acq,del_pref,del_aref)

% Create pulse sequence
% ----------------------------------------------------------------------
nexc=length(texc)+1;
nref=length(tref);

tp=zeros(1,nexc+NE*nref); phi=tp; amp=tp; acq=tp;

tp(1:nexc-1)=texc; phi(1:nexc-1)=pexc; amp(1:nexc-1)=aexc;
for i=1:NE
    tmp=nexc+(i-1)*nref+1:nexc+i*nref;
    tp(tmp)=tref; phi(tmp)=pref; amp(tmp)=aref;
    % Perturb pulse
    phi(tmp(2:end-1))=phi(tmp(2:end-1))+del_pref(i); % Pulse phase variation
    amp(tmp(2:end-1))=amp(tmp(2:end-1))+del_aref(i); % Pulse amplitude variation
    
    acq(nexc+i*nref)=1; % Acquire every echo
end
%acq(nexc+NE*nref)=1; % Acquire last echo only

% Current code - specify arbitrary (w0,w1) map
% ----------------------------------------------------------------------
numpts=2e4; % Number of spin vectors to simulate

maxoffs=10;  % Maximum static offset
del_w0=linspace(-maxoffs,maxoffs,numpts); % Uniform static field gradient

max_w1=0; % Maximum RF inhomogeneity
w_1=linspace(1-max_w1,1+max_w1,numpts); % Uniform RF field gradient

[macq]=sim_spin_dynamics_arb6(tp,phi,amp,acq,len_acq,del_w0,w_1);