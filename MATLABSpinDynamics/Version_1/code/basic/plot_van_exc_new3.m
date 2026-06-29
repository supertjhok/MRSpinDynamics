function [tvect,echo_int,asymp_rms]=plot_van_exc_new3(path,nums,nignore,norm)

close all;
% Input parameters
T_180=500; % us
zf=8; % zero-filling factor to get smoother spectra
p1=0.03*linspace(-1,1,1e2); % range of frequency-dependent linear phase correction parameters
wmax=20; % maximum offset frequency (normalized) of interest

if length(nums)==1
    [data,parameter]=readbrukerfile([path '\cpmg_oneshot_sp'],nums(1));
else % Phase cycling
    [data1,parameter]=readbrukerfile([path '\cpmg_oneshot_sp'],nums(1));
    [data2,~]=readbrukerfile([path '\cpmg_oneshot_sp'],nums(2));
    data=data1-data2; % Phase cycling
end

% Read other parameters from data file
ne=parameter.l(4); % Number of echoes
dw=parameter.dw;

T_ref=parameter.pulses(3); % Length of refocusing pulse (us)
T_FP=parameter.delays(21)*1e6; % Free precession time (us)
te=(2*T_FP+T_ref)/1e6; % Echo spacing (sec)
te_vect=te*linspace(1,ne,ne);

sizdat=size(data);
len=sizdat(1); % Total length
le=len/ne; % Samples per echo
tacq=dw*linspace(-(le-1)/2,(le-1)/2,le);

tvect=zeros(len,1);
for i=1:ne
    tvect((i-1)*le+1:i*le)=tacq+(i-1)*te;
end

% Plot CPMG decays
figure(1);
plot(tvect*1e3,abs(data(:,sizdat(2))),'k-'); hold on;
xlabel('Time (ms)');
ylabel('|Echoes|');
title('Echo decay');

% Plot asymptotic echo shape
echo_asymp=zeros(le,1);
for i=nignore+1:ne
    echo_asymp=echo_asymp+data((i-1)*le+1:i*le,sizdat(2));
end

phi=atan2(sum(imag(echo_asymp)),sum(real(echo_asymp))); % Estimate echo phase
echo_asymp=echo_asymp*exp(-1i*phi); % Rotate asymptotic echo onto real axis
asymp_rms=sqrt(trapz(tacq,abs(echo_asymp).^2));

figure(2);
plot(tacq*1e6/T_180,real(echo_asymp),'b-'); hold on;
plot(tacq*1e6/T_180,imag(echo_asymp),'r-');
xlabel('Normalized time, t / T_{180}');
ylabel('Asymptotic echo');

% Calculate echo integrals with matched filtering
echo_int=zeros(1,ne);
for i=1:ne
    echo_curr=data((i-1)*le+1:i*le,sizdat(2))*exp(-1i*phi);
    echo_int(i)=trapz(tacq,echo_curr.*conj(echo_asymp))/asymp_rms;
end

% Fit echo integrals to exponential, ignore first nignore echoes
optfun=@(vars)expfun(vars,te_vect(nignore+1:ne),echo_int(nignore+1:ne)/max(abs(echo_int)));
vars=fmincon(optfun,[1,1],[],[],[],[],[0,0],[1e2,1e2]);
A=vars(1)*max(abs(echo_int)); T2=vars(2);
disp(A)

% Use imaginary channel of last 75% of echoes to estimate snr (avoids
% transient artifacts) - in voltage units
% snr=A/std(imag(echo_int(ne/4:ne)));

figure(3);
plot(te_vect*1e3,real(echo_int)/A,'b-'); hold on;
plot(te_vect*1e3,imag(echo_int)/A,'r-');
plot(te_vect*1e3,exp(-te_vect/T2),'k--');
legend('Data (real)','Data (imag)',['Fit, T_{2} = ' num2str(round(T2)*1e3) ' ms']);
xlabel('Time (ms)');
ylabel('Echo amplitude (normalized)');

ws=2*pi/dw; % Hz
wvect=linspace(-ws/2,ws/2,le*zf); wind=find(abs(wvect)*T_180*1e-6/pi<wmax);
echo_zf=zeros(zf*le,1);
echo_zf((zf-1)*le/2:(zf+1)*le/2-1)=echo_asymp;

im_rms_min=0;
for i=1:length(p1) % Calculate frequency-dependent phase correction
    spect=zf*(fftshift(fft(ifftshift(echo_zf)))).*exp(-1i*p1(i)*wvect*T_180*1e-6)';
    im_rms=sqrt(trapz(wvect(wind),imag(spect(wind)).^2));
    if i==1 || im_rms<im_rms_min
        im_rms_min=im_rms; ind=i;
    end
end
disp(p1(ind))
spect=zf*(fftshift(fft(ifftshift(echo_zf)))).*exp(-1i*p1(ind)*wvect*T_180*1e-6)';
asymp_rms=sqrt(trapz(wvect(wind),real(spect(wind)).^2));

figure(4);
plot(wvect*T_180*1e-6/pi,real(spect)/norm,'b-'); hold on;
plot(wvect*T_180*1e-6/pi,imag(spect)/norm,'r-');
xlabel('\Delta\omega_{0}/\omega_{1}');
ylabel('Asymptotic magnetization (normalized)')
xlim([-20 20])

function val=expfun(vars,te_vect,echo_int)

A=vars(1); T2=vars(2);

ideal=A*exp(-te_vect/T2);
val=sum((ideal-real(echo_int)).^2);