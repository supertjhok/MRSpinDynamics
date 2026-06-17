function [snr_pow]=plot_oct_rpp_theoretical

T_90=pi/2; % normalized
delt=0.01*T_90;
len_acq=4*pi;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));

figure(1);
snr_pow=zeros(1,7);

% standard 90 - standard 180

tref=pi*[3 1 3]; pref=pi*[0 0 0]; aref=[0 1 0];
texc=[pi/2]; pexc=[pi/2]; aexc=[1];
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
[masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
[masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
masy=(masya-masyb)/2; snr_pow(1)=trapz(abs(masy).^2);
plot(del_w,real(masy),'b-'); hold on;

% (OCT_EXCA to OCT_EXCF)- RP2-1.0
tref=pi*[3 0.14 0.72 0.14 3]; pref=pi*[0 1 0 1 0]; aref=[0 1 1 1 0];
dat=load('dat_files\results_mag_all.mat'); results=dat.results_sort;

ind=[1,2,4,9,33,47];
for i=1:length(ind)    
    texc=results{ind(i),1}; pexc=results{ind(i),2}; aexc=ones(1,length(texc));
    [neff,del_w]=calc_rot_axis_arba(tref,pref,aref); % Calculate refocusing axis
    [masya]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
    [masyb]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
    masy=(masya-masyb)/2; snr_pow(i+1)=trapz(abs(masy).^2);
    plot(del_w,real(masy),'r-');
end

xlabel('Normalized frequency, \omega / \omega_{1}');
ylabel('Asymptotic magnetization (normalized)')

snr_pow=snr_pow/snr_pow(1); % Normalize SNR to standard 90 - 180 case