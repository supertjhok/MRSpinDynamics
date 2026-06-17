function [echo_pk,echo_rms]=cpmg_burst_arba_plot(texc,pexc,aexc,tref,pref,aref,T_90,NE1,T_FP1,NE2,T_FP2,delt)

% Cannot use random del_w for this function, else the two refocusing cycles
% will have different resonant offset vectors
[refmat1,del_w,w1_max]=calc_refocusing_mat_arba(tref,pref,aref,T_90,NE1,T_FP1);
[refmat2,del_w,w1_max]=calc_refocusing_mat_arba(tref,pref,aref,T_90,NE2,T_FP2);

numpts=length(del_w);
refmat=zeros(3,3,numpts);
for k=1:numpts
    refmat(:,:,k)=refmat2(:,:,k)*refmat1(:,:,k);
end

[echo,tvect,echo_pk,echo_rms]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,w1_max,delt);

figure(1);
%plot(tvect*1e6,abs(echo),'k-'); hold on;
plot(tvect*1e6+2*(T_FP1*NE1+T_FP2*NE2),abs(echo),'k-'); hold on;
xlabel('Time (\mus)');
ylabel('Asymptotic CPMG echo');

figure(2);
zf=8; % zero-filling to get smoother spectra
fs=1/(tvect(2)-tvect(1));
f1=1/(4*T_90*1e-6);
fvect=linspace(-fs/2,fs/2,length(tvect)*zf)/f1;

echo_zf=zeros(zf*length(echo),1);
echo_zf((zf-1)*length(echo)/2:(zf+1)*length(echo)/2-1)=echo;
spect=zf*abs(fftshift(fft(echo_zf)))/length(echo_zf);

figure(2); %clf;
plot(fvect,spect,'k-'); hold on;
xlabel('Normalized frequency, \omega / \omega_{1}');
ylabel('Echo spectrum (amplitude)')
