function [snr_norm]=plot_oct_ref_pulse_vantroy(pulse_num,tacq)

numpts=2001;  maxoffs=20;
del_w=linspace(-maxoffs,maxoffs,numpts);
wndw = sinc(del_w*tacq/(2*pi)); % wndw function for acquisition
wndw = wndw./sum(wndw);

% Load OCT refocusing pulse
filname=['LongerRefocusingPulses\Matlab_codes\Pulses\Axis_plus_antisymmetry\PASym_'...
    num2str(pulse_num) '_1.mat'];
tmp=load(filname); APulse=tmp.APulse; pulse=APulse.pulse;
tmp=size(pulse); nref=tmp(1);
tref=0.1*pi*ones(1,nref); % Segment length = 0.1 x T_180
pref=atan2(pulse(:,3),pulse(:,2))';
aref=sqrt(pulse(:,2).^2+pulse(:,3).^2)'./(2*pi*5e3); % Normalize to w1=1
tref=[3*pi tref 3*pi]; pref=[0 pref 0]; aref=[0 aref 0]; % Add free precession intervals

% Calculate refocusing axis
[neff]=calc_rot_axis_arba2(tref,pref,aref,del_w,0);
nx=dot(neff,neff).*neff(1,:); nx = conv(abs(nx),wndw,'same');
%nx = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(wndw);
%snr = trapz(del_w,(hamming(numpts)'.*abs(nx)).^2); % opt_wndw
snr = trapz(del_w,abs(nx).^2); % no opt_wndw

% Rectangular 180 pulse - reference
tref0=pi*[3 1 3]; pref0=pi*[0 1 0]; aref0=[0 1 0];
[neff0]=calc_rot_axis_arba2(tref0,pref0,aref0,del_w,0);
nx0=dot(neff0,neff0).*neff0(1,:); nx0 = conv(abs(nx0),wndw,'same');
%nx0 = fy(((numpts+1)/2:3*(numpts-1)/2+1));
%snr0 = trapz(del_w,(hamming(numpts)'.*abs(nx0)).^2); % opt_wndw
snr0 = trapz(del_w,abs(nx0).^2); % no opt_wndw

snr_norm=snr/snr0;
disp(snr_norm);

% Plot phase of OCT refocusing pulse
figure(1); clf;
[~,~]=make_continuous_phase(tref(2:length(tref)-1),pref(2:length(tref)-1));

% Plot refocusing axis
figure(2); clf;
%plot(del_w,wndw,'b--'); hold on;
plot(del_w,abs(nx),'k-'); hold on;
plot(del_w,abs(nx0),'r-');
xlabel('\Delta\omega_{0} / \omega_{1}');
ylabel('|n|^{2}|n_{y}|');