function [snr_pow]=plot_oct_rect_theoretical

T_90=pi/2; % normalized
delt=0.01*T_90;
len_acq=4*pi;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));

% standard 90 - standard 180

tref=pi*[3 1 3]; pref=pi*[0 0 0]; aref=[0 1 0];
texc=[pi/2]; pexc=[pi/2]; aexc=[1];
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy1=(masya-masyb)/2; snr_pow(1)=trapz(abs(masy1).^2);

% OCT - standard 180

dat=load('dat_files\results_mag13.mat'); results=dat.results;
texc=results{28,1}; pexc=results{28,2}; aexc=ones(1,length(texc));
tref=results{28,3}; pref=results{28,4}; aref=[0 1 0];
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy2=(masya-masyb)/2; snr_pow(2)=trapz(abs(masy2).^2);

% OCT - standard 135

dat=load('dat_files\results_mag14.mat'); results=dat.results;
texc=results{25,1}; pexc=results{25,2}; aexc=ones(1,length(texc));
tref=results{25,3}; pref=results{25,4}; aref=[0 1 0];
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy3=(masya-masyb)/2; snr_pow(3)=trapz(abs(masy3).^2);

% OCT - standard 124

dat=load('dat_files\results_mag14.mat'); results=dat.results;
texc=results{31,1}; pexc=results{31,2}; aexc=ones(1,length(texc));
tref=results{31,3}; pref=results{31,4}; aref=[0 1 0];
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy4=(masya-masyb)/2; snr_pow(4)=trapz(abs(masy4).^2);

figure(1);
plot(del_w,real(masy1),'b-'); hold on;
plot(del_w,real(masy2),'r-');
plot(del_w,real(masy3),'k-');
plot(del_w,real(masy4),'m-');
xlabel('Normalized frequency, \omega / \omega_{1}');
ylabel('Asymptotic magnetization (normalized)')

snr_pow=snr_pow/snr_pow(1); % Normalize SNR to standard 90 - 180 case