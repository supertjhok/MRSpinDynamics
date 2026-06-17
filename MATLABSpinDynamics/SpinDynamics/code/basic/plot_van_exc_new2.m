function [echo_int]=plot_van_exc_new2(nums,clr)

% Parameters
T_180 = 500; % us
P_180 = 17.6; % dB
ne=64; % Number of echoes

T_FP = T_180*6; % Free precession time (us)
te=(T_FP+T_180)/1e6; % Echo spacing (sec)

if length(nums)==1
    [data,parameter]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(1));
else % Phase cycling
    [data1,parameter]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(1));
    [data2,~]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(2));
    data=data1-data2; % Phase cycling
end

dw=parameter.dw;

sizdat=size(data);
len=sizdat(1); % Total length
le=len/ne; % Samples per echo
te_vect=dw*linspace(0,le-1,le);

tvect=zeros(len,1);
for i=1:ne
    tvect((i-1)*le+1:i*le)=te_vect+(i-1)*te;
end

% Plot CPMG decays
figure(1);
plot(tvect*1e3,abs(data(:,sizdat(2))),clr); hold on;
xlabel('Time (ms)');
ylabel('|Echo|');
title('Echo decay');

% Plot asymptotic echo shape
echo=zeros(le,1);
for i=11:ne
    echo=echo+data((i-1)*le+1:i*le,sizdat(2));
end

phi=atan(sum(imag(echo))/sum(real(echo))); % Estimate echo phase
echo=echo*exp(-1i*phi); % Rotate echo onto real axis

figure(2);
te_vect=(te_vect-max(te_vect)/2)/(T_180/2);
plot(te_vect*1e6,-real(echo),clr); hold on;
xlabel('Normalized time, t / T_{90}');
ylabel('Real part of echo');

% Calculate SNR
echo_int=sqrt(trapz(te_vect,abs(echo).^2));