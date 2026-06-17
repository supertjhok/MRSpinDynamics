% Read OCT pulses from data files and plot best SNR
% Allow pulse amplitude to be arbitrary

function [snr_norm,snr_matched_norm]=plot_oct_snr_varyamp(filname,pulse_num,amp)

% Simulation parameters
numpts=2001;  maxoffs=20;
del_w=linspace(-maxoffs,maxoffs,numpts);
tacq=4*pi; window = sinc(del_w*tacq/(2*pi)); % window function for acquisition
window=window./sum(window);

% Reference - rectangular 90/180
texc0=[pi/2 -1]; pexc0=[pi/2 0]; aexc0=[1 0];
tref0=pi*[3 1 3]; pref0=[0 0 0]; aref0=[0 1 0];
[neff0]=calc_rot_axis_arba2(tref0,pref0,aref0,del_w,0);
[masy0_1]=cpmg_van_spin_dynamics_asymp_mag3(texc0,pexc0,aexc0,neff0,del_w,tacq);
[masy0_2]=cpmg_van_spin_dynamics_asymp_mag3(texc0,pexc0+pi,aexc0,neff0,del_w,tacq);
masy0=(masy0_1-masy0_2)/2;
snr0=trapz(del_w,abs(masy0).^2);
[echo0,tvect]=calc_time_domain_echo(masy0,del_w);

% Load OCT pulses
tmp=load(filname); results=tmp.results;

% Calculate SNR
texc=results{pulse_num,1}; pexc=results{pulse_num,2}; aexc=amp*ones(1,length(texc));
tref=results{pulse_num,3}; pref=results{pulse_num,4}; aref=amp*[0 ones(1,length(tref)-2) 0];
%texc=[pi/2 -1]; pexc=[pi/2 0]; aexc=amp*[1 0];
%tref=pi*[3 1 3]; pref=[0 0 0]; aref=amp*[0 1 0];
[neff]=calc_rot_axis_arba2(tref,pref,aref,del_w,0);
[masy_1]=cpmg_van_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,del_w,tacq); % PI cycle
[masy_2]=cpmg_van_spin_dynamics_asymp_mag3(texc,-pexc,aexc,neff,del_w,tacq);
masy=(masy_1-masy_2)/2;
% No PI cycle
%[masy]=cpmg_van_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,del_w,tacq);
snr_norm=trapz(del_w,abs(masy).^2)/snr0;

% Maximum possible signal - perfect matching
masy_matched = dot(neff,neff).*neff(1,:);
masy_matched = conv(abs(masy_matched),window,'same');
snr_matched_norm=trapz(del_w,abs(masy_matched).^2)/snr0;

figure(1); plot(del_w,real(masy),'b-'); % Spectrum

[echo,~]=calc_time_domain_echo(masy,del_w); % Time domain
figure(2); clf;
plot(tvect/(pi/2),real(echo),'b-'); hold on; plot(tvect/(pi/2),imag(echo),'r-');

% Plot reference - rectangular 90/180 and refocusing axis

figure(1); hold on;
plot(del_w,real(masy0),'m-'); % Spectrum
plot(del_w,abs(masy_matched),'k--');
xlabel('\Delta\omega_{0} / \omega_{1}');
ylabel('M_{asy,y} , |n_{y}|');
title(['N_{exc} = ' num2str(length(texc)) ', SNR = ' num2str(snr_norm)])

figure(2);
plot(tvect/(pi/2),real(echo0),'m-'); %plot(tvect/(pi/2),imag(echo0),'b--');
xlim([-tacq/2 tacq/2]/(pi/2))
xlabel('T / T_{90}');
ylabel('M_{asy,y}');
title(['N_{exc} = ' num2str(length(texc)) ', SNR = ' num2str(snr_norm)])