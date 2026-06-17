% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% TE = echo spacing
% Delays follow pulses

function [echo_pk,echo_rms]=cpmg_van_spin_dynamics_plot(name,T_90,NE,T_FP,T1,T2)

tmp=load('exc_timings.mat');
results=tmp.results;

tmp=size(results);
numres=tmp(1);
for i=1:numres
    if strcmpi(name,results{i,1})
        texc=T_90*results{i,2};
        pexc=results{i,3};
    end
end

if strcmpi(name,'rectangular')
    % Rectangular refocusing pulse
    tref=2*T_90;
    pref=pi/2;
else
    % RP2-1.0a refocusing pulse
    tref=2*T_90*[0.14,0.72,0.14];
    pref=pi*[3,1,3]/2;
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

for i=1:NE
    tp(nexc+(i-1)*nref+1:nexc+i*nref)=tref;
    phi(nexc+(i-1)*nref+1:nexc+i*nref)=pref;
    tf(nexc+(i-1)*nref+1:nexc+i*nref-1)=zeros(1,nref-1);
    tf(nexc+i*nref)=T_FP;
end
tf(nexc+NE*nref)=T_FP/2;

[echo,tvect]=sim_spin_dynamics_allpw(T_90,tp,phi,tf,T1,T2);
echo_pk=max(abs(echo));
echo_rms=trapz(tvect,abs(echo).^2)*1e8;

figure(1); %clf;
%plot(tvect*1e6,real(echo),'b-'); hold on;
%plot(tvect*1e6,imag(echo),'r-');
plot(tvect/(T_90*1e-6),abs(echo),'k-'); hold on;
set(gca,'FontSize',14);
xlabel('Normalized time, t / T_{90}');
ylabel('Waveform of echo, s(t)')

zf=8; % zero-filling to get smoother spectra
fs=1/(tvect(2)-tvect(1));
f1=1/(4*T_90*1e-6);
fvect=linspace(-fs/2,fs/2,length(tvect)*zf)/f1;

echo_zf=zeros(zf*length(echo),1);
echo_zf((zf-1)*length(echo)/2:(zf+1)*length(echo)/2-1)=echo;
spect=zf*abs(fftshift(fft(real(echo_zf))+fft(imag(echo_zf))))/length(echo_zf);

figure(2); %clf;
plot(fvect,spect,'k-'); hold on;
set(gca,'FontSize',14);
xlabel('Normalized frequency, \omega / \omega_{1}');
ylabel('Echo spectrum (amplitude)')

ind=find(fvect>0);
spect_sum=cumtrapz(fvect(ind),spect(ind).^2);

figure(3); %clf;
plot(fvect(ind),spect_sum/max(spect_sum),'k-'); hold on;
set(gca,'FontSize',14);
xlabel('Normalized frequency, \omega / \omega_{1}');
ylabel('Cumulative echo power')
