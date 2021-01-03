function [echo_int]=plot_van_exc(choice)

% Parameters
T_180 = 380; % us
P_180 = 15.2; % dB
ne=32; % Number of echoes

% Experiment numbers (choose set)

switch choice
    case 1
        % Case 1
        T_FP = T_180*6;
        nums=linspace(8,12,5);
        
    case 2
        % Case 2
        T_FP = T_180*9;
        nums=linspace(13,17,5);
end

te=(T_FP+T_180)/1e6; % Echo spacing (sec)

[data1,parameter1]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(1)); % Rectangular

[data2,parameter2]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(2)); % Amplitude
[data3,parameter3]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(3));
data2=data2-data3; % Phase cycling

[data4,parameter2]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(4)); % Power
[data5,parameter3]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(5));
data3=data4-data5; % Phase cycling

% Subtract baseline offset
off_re=mean(real(data2(end-10:end-10,2)));
off_im=mean(imag(data2(end-10:end-10,2)));
data2(:,2)=data2(:,2)-off_re-sqrt(-1)*off_im;

off_re=mean(real(data3(end-10:end-10,2)));
off_im=mean(imag(data3(end-10:end-10,2)));
data3(:,2)=data3(:,2)-off_re-sqrt(-1)*off_im;

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
plot(tvect*1e3,abs(data1(:,2)));
xlabel('Time (ms)');
title('Rectangular');

figure(2);
plot(tvect*1e3,abs(data2(:,2)));
xlabel('Time (ms)');
title('VAN-EXC-AMPL / RP2-1.0');

figure(3);
plot(tvect*1e3,abs(data3(:,2)));
xlabel('Time (ms)');
title('VAN-EXC / RP2-1.0');

% Plot asymptotic echo shape
echo1=zeros(le,1); echo2=echo1; echo3=echo1;
for i=11:ne
    echo1=echo1+data1((i-1)*le+1:i*le,2);
    echo2=echo2+data2((i-1)*le+1:i*le,2);
    echo3=echo3+data3((i-1)*le+1:i*le,2);
end

figure(4);
te_vect=te_vect-max(te_vect)/2;
plot(te_vect*1e6,abs(echo1),'b-'); hold on;
plot(te_vect*1e6,abs(echo2),'r-');
plot(te_vect*1e6,abs(echo3),'k-');
xlabel('Time (us)');
legend('Rectangular','VAN-EXC-AMPL','VAN-EXC');

% Calculate SNR
echo_int=zeros(3,1);
echo_int(1)=trapz(te_vect,abs(echo1).^2);
echo_int(2)=trapz(te_vect,abs(echo2).^2);
echo_int(3)=trapz(te_vect,abs(echo3).^2);
echo_int=echo_int/echo_int(1); % Normalize to rectangular case