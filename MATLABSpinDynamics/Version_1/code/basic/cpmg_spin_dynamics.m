% NE - number of echoes
% TE - echo spacing
% rat - T90/T180 pulse duration ratio
% Delays follow pulses

function [echo_pk,spect_rms]=cpmg_spin_dynamics(T_90,rat,NE,TE,T1,T2)

T_180=T_90*rat;

tp=T_180*ones(1,NE+1);
tp(1)=T_90;

phi1=(pi/2)*ones(1,NE+1);
phi1(1)=0;

phi2=phi1;
phi2(1)=pi;

tf=(TE-T_180)*ones(1,NE+1);
%tf(1)=0.5*(TE-T_180);
% Martin's correction for gradient fields - gives more peak signal by making SE
% and DE echoes occur closer together in time
tf(1)=0.5*(TE-T_180-T_180/pi);
tf(NE+1)=0.5*(TE-T_180);

% PAP
%[echo1,tvect]=sim_spin_dynamics_allpw(T_90,tp,phi1,tf,T1,T2);
%[echo2,tvect]=sim_spin_dynamics_allpw(T_90,tp,phi2,tf,T1,T2);
%echo=(echo1-echo2);

% No PAP
[echo,tvect]=sim_spin_dynamics_allpw(T_90,tp,phi1,tf,T1,T2);

echo_pk=max(abs(echo));

%close all;

figure(1);
plot(tvect*1e6,abs(echo),'k-'); hold on;
%plot(tvect*1e6,real(echo),'b-'); 
%plot(tvect*1e6,imag(echo),'r-');
set(gca,'FontSize',14);
xlabel('Time (\mus)');
ylabel('Waveform of echo, s(t)')

% Calculate spectrum - note that trans can also be plotted directly to show
% the spectrum
fs=1/(tvect(2)-tvect(1));
necho=length(echo);
del_f=linspace(-fs/2,fs/2,necho);

% Spectrum must be normalized
spect=abs(fftshift(fft(real(echo))+fft(imag(echo))))/necho;
spect_rms=sqrt(trapz(del_f/fs,spect.^2));

figure(2);
plot(del_f/1e3,spect,'k-'); hold on;
set(gca,'FontSize',14);
xlabel('Frequency (kHz)');
ylabel('Spectrum of echo, |S(f)|')