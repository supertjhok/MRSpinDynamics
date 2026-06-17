% Random distribution of amplitude and/or timing errors

function [eint0,eint,eint_mean,eint_std]=cpmg_envelope_statistics(sel,ne,T_E,sigma,numruns,tnoise)

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

% Vary pulse amplitudes and durations randomly
noise_a=aref0*sigma*randn(2*numruns,ne);
noise_t=tref0*sigma*randn(2*numruns,ne);

tref1=zeros(1,3*ne);
aref1=tref1;
tref2=tref1;
aref2=aref1;

% Calculate echo integrals
eint=zeros(numruns,ne);
for i=1:numruns
    aref1(1,2:3:3*ne-1)=aref(1,2:3:3*ne-1)+noise_a(1+2*(i-1),:);
    aref2(1,2:3:3*ne-1)=aref(1,2:3:3*ne-1)+noise_a(2*i,:);
    if tnoise % Timing noise enabled
        tref1(1,2:3:3*ne-1)=tref(1,2:3:3*ne-1)+noise_t(1+2*(i-1),:);
        tref2(1,2:3:3*ne-1)=tref(1,2:3:3*ne-1)+noise_t(2*i,:);
    else % No timing noise
        tref1(1,2:3:3*ne-1)=tref(1,2:3:3*ne-1);
        tref2(1,2:3:3*ne-1)=tref(1,2:3:3*ne-1);
    end
    % Calculate echoes with timing correction and phase cycling
    [mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref1,pref,aref1,ne,del_w,t_acq);
    [mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref2,pref,aref2,ne,del_w,t_acq);
    mecho=(mecho1-mecho2)/2;
    
    for n=1:ne
        eint(i,n)=trapz(del_w,abs(mecho(n,:).*masy))/normasy;
    end
    disp(['Finished run #' num2str(i)])
end

close all;
eint_mean=zeros(1,ne); eint_std=zeros(1,ne);

figure(1);
plot(eint0,'b-'); hold on;
for i=1:numruns
    plot(eint(i,:),'ro');
end

for i=1:ne
    eint_mean(i)=mean(eint(:,i));
    eint_std(i)=std(eint(:,i));
end
plot(eint_mean,'r--');

figure(2);
plot(eint_std,'r--');