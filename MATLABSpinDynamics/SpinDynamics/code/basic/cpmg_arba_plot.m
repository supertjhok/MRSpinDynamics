function [echo_pk,echo_rms]=cpmg_arba_plot(texc,pexc,aexc,tref,pref,aref,T_90,NE,T_FP)

len_acq=T_FP-T_90;
delt=6*T_90;

[refmat,del_w,w1_max]=calc_refocusing_mat_arba(tref,pref,aref,T_90,NE,T_FP);
[echo,tvect,echo_pk,echo_rms]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,w1_max,delt,len_acq);

figure(1);
plot(tvect*1e6/T_90,abs(echo),'k-'); hold on;
xlabel('Normalized time T / T_{90}');
%plot(tvect*1e6+2*T_FP*NE,abs(echo),'k-'); hold on;
%xlabel('Time (\mus)');
ylabel('Asymptotic CPMG echo');

figure(2);
zf=8; % zero-filling to get smoother spectra
fs=1/(tvect(2)-tvect(1));
f1=1/(4*T_90*1e-6);
fvect=linspace(-fs/2,fs/2,length(tvect)*zf)/f1;

echo_zf=zeros(zf*length(echo),1);
echo_zf((zf-1)*length(echo)/2:(zf+1)*length(echo)/2-1)=echo;
%spect=zf*abs(fftshift(fft(echo_zf)));
spect=zf*abs(fftshift(fft(real(echo_zf))+fft(imag(echo_zf))));

plot(fvect,spect,'k-'); hold on;
xlabel('Normalized frequency, \omega / \omega_{1}');
ylabel('Echo spectrum (amplitude)')
