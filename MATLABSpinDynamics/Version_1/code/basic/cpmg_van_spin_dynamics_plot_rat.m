% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% rat = (refocusing pulse length)/T_90
% NE = number of echoes
% TE = echo spacing
% Delays follow pulses

function [echo_pk,echo_rms]=cpmg_van_spin_dynamics_plot_rat(exc_name,ref_name,T_90,rat,NE,T_FP,T1,T2)

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
        tref=T_90*results{i,2};
        pref=results{i,3};
    end
end

nexc=length(texc);
nref=length(tref);

tp=zeros(1,nexc+NE*nref);
phi=tp;
tf=tp;

tp(1:nexc)=texc;
phi(1:nexc)=pexc;
tf(1:nexc-1)=zeros(1,nexc-1);

if nexc==1 % Assume rectangular pulse
    tf(nexc)=0.5*T_FP-2*T_90/pi; % Martin's timing correction
else % Not a rectangular pulse
    tf(nexc)=0.5*T_FP;
end

echo_pk=zeros(1,length(rat)); echo_rms=echo_pk;
figure(1);
for j=1:length(rat)
    for i=1:NE
        tp(nexc+(i-1)*nref+1:nexc+i*nref)=rat(j)*tref;
        phi(nexc+(i-1)*nref+1:nexc+i*nref)=pref;
        tf(nexc+(i-1)*nref+1:nexc+i*nref-1)=zeros(1,nref-1);
        tf(nexc+i*nref)=T_FP;
    end
    tf(nexc+NE*nref)=T_FP/2;
    disp(rat(j))
    
    [echo,tvect]=sim_spin_dynamics_allpw(T_90,tp,phi,tf,T1,T2);
    echo_pk(j)=max(abs(echo));
    echo_rms(j)=trapz(tvect,abs(echo).^2)*1e8;
    
    plot(tvect/(T_90*1e-6),abs(echo)); hold on;
end

figure(2); %clf;
plot(rat,echo_pk,'b-');
xlabel('T_{180} / T_{90}')
ylabel('Maximum value of echo');

figure(3); %clf;
plot(rat,echo_rms,'b-');
xlabel('T_{180} / T_{90}')
ylabel('Squared integral of echo');