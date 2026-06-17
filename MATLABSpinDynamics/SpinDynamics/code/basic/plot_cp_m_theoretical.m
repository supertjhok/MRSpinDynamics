function [snr_pow]=plot_cp_m_theoretical

T_90=pi/2; % normalized
delt=0.01*T_90;
len_acq=6.2*pi;
tvect=-len_acq/2:delt:len_acq/2;

% standard 90 - standard 180

tref=pi*[6 1 6]; pref=pi*[0 0 0]; aref=[0 1 0];
texc=[pi/2]; pexc=[pi/2]; aexc=[1];
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy1=(masya-masyb)/2; snr_pow(1)=trapz(abs(masy1).^2);
echo1=calc_echo(masy1,del_w,tvect);

% strong 90 - standard 180

tref=pi*[6 1 6]; pref=pi*[0 0 0]; aref=[0 1 0];
a=10; % normalized 90 amplitude
texc=[pi/2]/a; pexc=[pi/2]; aexc=[a];
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy2=(masya-masyb)/2; snr_pow(2)=trapz(abs(masy2).^2);
echo2=calc_echo(masy2,del_w,tvect);

% strong 90 - RPP-1.0

tref=pi*[6 0.14 0.72 0.14 6]; pref=pi*[0 1 0 1 0]; aref=[0 1 1 1 0];
a=10; % normalized 90 amplitude
texc=[pi/2]/a; pexc=[pi/2]; aexc=[a];
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy3=(masya-masyb)/2; snr_pow(3)=trapz(abs(masy3).^2);
echo3=calc_echo(masy3,del_w,tvect);

% BPP_1 - RPP-1.0

tref=pi*[6 0.14 0.72 0.14 6]; pref=pi*[0 1 0 1 0]; aref=[0 1 1 1 0];
dat=cp_m_pulse_data;
texc=(pi/2)*dat(:,2); nexc=length(texc);
pexc=[pi/2]*ones(1,nexc); pexc(1:2:nexc-1)=-pexc(1:2:nexc-1); aexc=ones(1,nexc);
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy4=(masya-masyb)/2; snr_pow(4)=trapz(abs(masy4).^2);
echo4=calc_echo(masy4,del_w,tvect);

figure(1);
plot(del_w,real(masy1),'b-'); hold on;
plot(del_w,real(masy2),'r-');
plot(del_w,real(masy3),'k-');
xlabel('Normalized frequency, \omega / \omega_{1}');
ylabel('Asymptotic magnetization (normalized)')

figure(2);
plot(del_w,real(masy1),'b-'); hold on;
plot(del_w,real(masy4),'m-');
xlabel('Normalized frequency, \omega / \omega_{1}');
ylabel('Asymptotic magnetization (normalized)')

figure(3);
plot(tvect/(2*T_90),real(echo1)/max(abs(echo1)),'b-'); hold on;
plot(tvect/(2*T_90),real(echo2)/max(abs(echo1)),'r-');
plot(tvect/(2*T_90),real(echo3)/max(abs(echo1)),'k-');
xlabel('Normalized time, t / T_{180}');
ylabel('Asymptotic echo (normalized)')

figure(4);
plot(tvect/(2*T_90),real(echo1)/max(abs(echo1)),'b-'); hold on;
plot(tvect/(2*T_90),imag(echo1)/max(abs(echo1)),'b--');
plot(tvect/(2*T_90),real(echo4)/max(abs(echo1)),'m-');
plot(tvect/(2*T_90),imag(echo4)/max(abs(echo1)),'m--');
xlabel('Normalized time, t / T_{180}');
ylabel('Asymptotic echo (normalized)')

snr_pow=snr_pow/snr_pow(1); % Normalize SNR to standard 90 - 180 case

function [echo]=calc_echo(masy,del_w,tvect)

echo=zeros(1,length(tvect));
for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end