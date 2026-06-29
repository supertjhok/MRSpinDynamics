function [echo_int]=plot_van_exc_new(nums)

% Parameters
T_180 = 500; % us
P_180 = 17.6; % dB
ne=64; % Number of echoes

T_FP = T_180*6; % Free precession time (us)
te=(T_FP+T_180)/1e6; % Echo spacing (sec)

[data1,parameter1]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(1)); % Rectangular
sizdat=size(data1);

[data2,parameter2]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(2)); % VAN_EXC / RP2
[data3,parameter3]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(3));
data2=data2-data3; % Phase cycling

[data3,parameter3]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(4)); % CP-M15 / RP2
[data4,parameter4]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(5)); % Rect / RP2

dw=parameter1.dw;

sizdat=size(data1);
len=sizdat(1); % Total length
le=len/ne; % Samples per echo
te_vect=dw*linspace(0,le-1,le);

tvect=zeros(len,1);
for i=1:ne
    tvect((i-1)*le+1:i*le)=te_vect+(i-1)*te;
end

% Plot CPMG decays
close all;
figure(1);
plot(tvect*1e3,abs(data1(:,sizdat(2))));
xlabel('Time (ms)');
title('Rectangular');

figure(2);
plot(tvect*1e3,abs(data2(:,sizdat(2))));
xlabel('Time (ms)');
title('VAN-EXC / RP2-1.0');

figure(3);
plot(tvect*1e3,abs(data3(:,sizdat(2))));
xlabel('Time (ms)');
title('CP-M15 / RP2-1.0');

figure(4);
plot(tvect*1e3,abs(data4(:,sizdat(2))));
xlabel('Time (ms)');
title('Rect / RP2-1.0');

% Plot asymptotic echo shape
echo1=zeros(le,1); echo2=echo1; echo3=echo1; echo4=echo1;
for i=11:ne
    echo1=echo1+data1((i-1)*le+1:i*le,sizdat(2));
    echo2=echo2+data2((i-1)*le+1:i*le,sizdat(2));
    echo3=echo3+data3((i-1)*le+1:i*le,sizdat(2));
    echo4=echo4+data4((i-1)*le+1:i*le,sizdat(2));
end

figure(4);
te_vect=(te_vect-max(te_vect)/2)/(T_180/2);
plot(te_vect*1e6,abs(echo1),'b-'); hold on;
plot(te_vect*1e6,abs(echo2),'r-');
plot(te_vect*1e6,abs(echo3),'k-');
plot(te_vect*1e6,abs(echo4),'m-');
xlabel('Normalized time, t / T_{90}');
legend('Rectangular','VAN-EXC','CP-M15','Hard');

% Calculate SNR
echo_int=zeros(3,1);
echo_int(1)=sqrt(trapz(te_vect,abs(echo1).^2));
echo_int(2)=sqrt(trapz(te_vect,abs(echo2).^2));
echo_int(3)=sqrt(trapz(te_vect,abs(echo3).^2));
echo_int(4)=sqrt(trapz(te_vect,abs(echo4).^2));
echo_int=echo_int/echo_int(1); % Normalize to rectangular case