% Offset of pulse amplitude and/or timing (uniform for all echoes)

function [eint0,eint,eint_err]=cpmg_envelope_misset(sel,ne,T_E,sigma,numruns)

% Uniform distribution of del_w0 (uniform gradient)
maxoffs=20;
numpts=1e4+1; % number of spin vectors to simulate

% Resonance offset distribution
del_w=linspace(-maxoffs,maxoffs,numpts); % Deterministic distribution

% Load optimal envelopes
tmp=load('opt_cpmg_envelope_results.mat');
switch sel
    case 0 % No envelope compensation
        avals=1;
    case 1 % Ideal envelope compensation
        avals=tmp.avals;
    case 2 % 2-level envelope compensation
        avals=tmp.avals2;
    case 3 % 3-level envelope compensation
        avals=tmp.avals3;
end
k=length(avals);

% Excitation pulse
texc=pi/2;
pexc=pi;
aexc=1;

% Refocusing pulse
tref0=pi;
pref0=pi/2;
aref0=1;

T_FP=T_E-tref0; % Free precession time
t_acq=T_FP; % Acquisition time

tref=(T_FP/2)*ones(1,3*ne);
pref=zeros(1,3*ne);
aref=pref;

tref(1,2:3:3*ne-1)=tref0*ones(1,ne);
pref(1,2:3:3*ne-1)=pref0*ones(1,ne);

% Set pulse amplitudes and durations
for j=1:k-1
    aref(1,2+3*(j-1))=avals(j);
    tref(1,2+3*(j-1))=tref(1,2+3*(j-1))/aref(1,2+3*(j-1));
end

aref(1,2+3*(k-1):3:3*ne-1)=avals(k)*ones(1,ne-(k-1));
tref(1,2+3*(k-1):3:3*ne-1)=tref(1,2+3*(k-1):3:3*ne-1)./aref(1,2+3*(k-1):3:3*ne-1);

% Calculate echoes with timing correction and phase cycling
[mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref,pref,aref,ne,del_w,t_acq);
[mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref,pref,aref,ne,del_w,t_acq);
mecho=(mecho1-mecho2)/2;

% Asymptotic echo
masy=mecho(ne,:);
normasy=sqrt(trapz(del_w,abs(masy.*masy)));

eint0=zeros(1,ne);
for n=1:ne
    eint0(n)=trapz(del_w,abs(mecho(n,:).*masy))/normasy;
end

% Vary pulse amplitudes
misset=linspace(1-sigma,1+sigma,numruns);

% Calculate echo integrals
aref1=zeros(1,3*ne);
eint=zeros(numruns,ne);
for i=1:numruns
    aref1(1,2:3:3*ne-1)=aref(1,2:3:3*ne-1)*misset(i);

    % Calculate echoes with timing correction and phase cycling
    [mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref,pref,aref1,ne,del_w,t_acq);
    [mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref,pref,aref1,ne,del_w,t_acq);
    mecho=(mecho1-mecho2)/2;
    
    for n=1:ne
        eint(i,n)=trapz(del_w,abs(mecho(n,:).*masy))/normasy;
    end
    disp(['Finished run #' num2str(i)])
end

close all;
eint_err=zeros(numruns,ne);

figure(1);
plot(eint0,'b-'); hold on;
for i=1:numruns
    plot(eint(i,:),'ro');
end

figure(2);
for i=1:ne
    eint_err(:,i)=eint(:,i)/eint0(i)-1;
    plot(misset-1,eint_err(:,i),'r--'); hold on;
end