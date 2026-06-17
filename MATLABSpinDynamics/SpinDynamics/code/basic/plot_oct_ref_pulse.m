function [snr_norm]=plot_oct_ref_pulse(filname,pulse_num,tacq)

numpts=2001;  maxoffs=20;
del_w=linspace(-maxoffs,maxoffs,numpts);
wndw = sinc(del_w*tacq/(2*pi)); % wndw function for acquisition
wndw = wndw./sum(wndw);

% OCT refocusing pulse
tmp=load(filname); results=tmp.results;
tref=results{pulse_num,1}; pref=results{pulse_num,2}; aref=[0 ones(1,length(tref)-2) 0];
%tref=pi*[3 0.14 0.72 0.14 3]; pref = pi*[0 1 0 1 0]; aref = [0 1 1 1 0]; % RP2-1.0 pulse

[neff]=calc_rot_axis_arba3(tref,pref,aref,del_w,0);
nx=neff(1,:); nx = conv(nx,wndw,'same');
nz=neff(3,:);
%snr = trapz(del_w,(hamming(numpts)'.*abs(nx)).^2); % opt_wndw
snr = trapz(del_w,abs(nx).^2); % no opt_wndw

% Rectangular 180 pulse - reference
tref0=pi*[3 1 3]; pref0=pi*[0 0 0]; aref0=[0 1 0];
[neff0]=calc_rot_axis_arba3(tref0,pref0,aref0,del_w,0);
nx0=neff0(1,:); nx0 = conv(nx0,wndw,'same');
nz0=neff0(3,:);
%snr0 = trapz(del_w,(hamming(numpts)'.*abs(nx0)).^2); % opt_wndw
snr0 = trapz(del_w,abs(nx0).^2); % no opt_wndw

snr_norm=snr/snr0;
disp(snr_norm)

% Plot phase of OCT refocusing pulse
figure(1); clf;
[~,~]=make_continuous_phase(tref(2:length(tref)-1),pref(2:length(tref)-1));

% Plot refocusing axis
figure(2); clf;
plot(del_w,abs(nx),'k-'); hold on;
plot(del_w,abs(nx0),'r-');
xlabel('\Delta\omega_{0} / \omega_{1}');
ylabel('|n_{x}|');

% Plot refocusing axis
figure(3); clf;
plot(del_w,nz,'k-'); hold on;
plot(del_w,nz0,'r-');
xlabel('\Delta\omega_{0} / \omega_{1}');
ylabel('n_{z}');