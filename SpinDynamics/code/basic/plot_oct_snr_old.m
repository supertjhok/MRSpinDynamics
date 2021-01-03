% Read OCT pulses from data files and plot best SNR
% Assume RP2-1.0 refocusing pulses

function [snr_norm,snr_matched_norm]=plot_oct_snr_old(filname)

% Simulation parameters
numpts=2001;  maxoffs=20;
del_w=linspace(-maxoffs,maxoffs,numpts);
tacq=4*pi; window = sinc(del_w*tacq/(2*pi)); % window function for acquisition

% Reference - rectangular 90/180
texc0=[pi/2 -1]; pexc0=[pi/2 0]; aexc0=[1 0];
tref0=pi*[3 1 3]; pref0=[0 0 0]; aref0=[0 1 0];
[neff0]=calc_rot_axis_arba2(tref0,pref0,aref0,del_w,0);
[masy0_1]=cpmg_van_spin_dynamics_asymp_mag3(texc0,pexc0,aexc0,neff0,del_w,tacq);
[masy0_2]=cpmg_van_spin_dynamics_asymp_mag3(texc0,pexc0+pi,aexc0,neff0,del_w,tacq);
masy0=(masy0_1-masy0_2)/2;
snr0=trapz(del_w,abs(masy0).^2);
[echo0,tvect]=calc_time_domain_echo(masy0,del_w);

% Load OCT excitation pulses
tmp=load(filname); results=tmp.results;
siz=size(results);

% RP2-1.0 refocusing pulse
tref=pi*[3 0.14 0.72 0.14 3]; pref=[0 pi 0 pi 0]; aref=[0 ones(1,length(tref)-2) 0];
[neff]=calc_rot_axis_arba2(tref,pref,aref,del_w,0);

% Maximum possible signal - perfect matching
masy_matched = dot(neff,neff).*neff(1,:); fy = conv(masy_matched,window);
masy_matched = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(window);
snr_matched_norm=trapz(del_w,abs(masy_matched).^2)/snr0;
    
% Calculate SNR
snr_norm=zeros(1,siz(1)); snr_max=0;
for i=1:siz(1)
    texc=results{i,1}; pexc=results{i,2}; aexc=ones(1,length(texc));
    [masy_1]=cpmg_van_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,del_w,tacq);
    [masy_2]=cpmg_van_spin_dynamics_asymp_mag3(texc,-pexc,aexc,neff,del_w,tacq); % PI cycle
    masy=(masy_1-masy_2)/2;
    snr_norm(i)=trapz(del_w,abs(masy).^2)/snr0;
    
    if snr_norm(i)>snr_max
        snr_max=snr_norm(i);
        figure(1); plot(del_w,real(masy),'b-'); % Spectrum
             
        [echo,~]=calc_time_domain_echo(masy,del_w); % Time domain
        figure(2); clf;
        plot(tvect/(pi/2),real(echo),'b-'); hold on; plot(tvect/(pi/2),imag(echo),'r-');
    end
end

% Plot reference - rectangular 90/180 and refocusing axis

figure(1); hold on;
plot(del_w,real(masy0),'m-'); % Spectrum
plot(del_w,abs(masy_matched),'k--');
xlabel('\Delta\omega_{0} / \omega_{1}');
ylabel('M_{asy,y} , |n|^{2}|n_{y}|');
title(['N_{exc} = ' num2str(length(texc)) ', SNR = ' num2str(snr_max)])

figure(2);
plot(tvect/(pi/2),real(echo0),'m-'); %plot(tvect/(pi/2),imag(echo0),'b--');
xlim([-tacq/2 tacq/2]/(pi/2))
xlabel('T / T_{90}');
ylabel('M_{asy,y}');
title(['N_{exc} = ' num2str(length(texc)) ', SNR = ' num2str(snr_max)])