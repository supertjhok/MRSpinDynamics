function [echo_int]=plot_arbphase(choice)

% Parameters
ne=32; % Number of echoes

% Experiment numbers
switch choice
    case 1
        T_180 = 100; % us
        P_180 = 3.6; % dB
        T_FP = T_180*34.2;
        nums=linspace(18,22,5);
        
    case 2
        T_180 = 380; % us
        P_180 = 15.2; % dB
        T_FP = T_180*9;
        nums=linspace(23,27,5);
end

te=(T_FP+T_180)/1e6; % Echo spacing (sec)

[data1,parameter1]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(1)); % Rectangular
[data2,parameter2]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(2)); % RESULTS_0
[data3,parameter3]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(3)); % RESULTS_1
[data4,parameter4]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(4)); % RESULTS_2
[data5,parameter5]=readbrukerfile('van_exc\cpmg_oneshot_sp',nums(5)); % RESULTS_3

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
title('\Delta\phi = 0');

figure(3);
plot(tvect*1e3,abs(data3(:,2)));
xlabel('Time (ms)');
title('\Delta\phi = \pi/2');

figure(4);
plot(tvect*1e3,abs(data4(:,2)));
xlabel('Time (ms)');
title('\Delta\phi = \pi');

figure(5);
plot(tvect*1e3,abs(data5(:,2)));
xlabel('Time (ms)');
title('\Delta\phi = 3\pi/2');

% Plot asymptotic echo shape
echo1=zeros(le,1); echo2=echo1; echo3=echo1; echo4=echo1; echo5=echo1;
for i=11:ne
    echo1=echo1+data1((i-1)*le+1:i*le,2);
    echo2=echo2+data2((i-1)*le+1:i*le,2);
    echo3=echo3+data3((i-1)*le+1:i*le,2);
    echo4=echo4+data4((i-1)*le+1:i*le,2);
    echo5=echo5+data5((i-1)*le+1:i*le,2);
end

figure(6);
te_vect=te_vect-max(te_vect)/2;
plot(te_vect*1e6,real(echo1),'b-'); hold on;
plot(te_vect*1e6,imag(echo1),'b--');
xlabel('Time (us)');
legend('Rectangular (real)','Rectangular (imag)');

figure(7);
plot(te_vect*1e6,real(echo2),'b-'); hold on;
plot(te_vect*1e6,imag(echo2),'b--');
plot(te_vect*1e6,real(echo4),'k-');
plot(te_vect*1e6,imag(echo4),'k--');
xlabel('Time (us)');
legend('\Delta\phi = 0 (real)','\Delta\phi = 0 (imag)',...
    '\Delta\phi = \pi (real)','\Delta\phi = \pi (imag)');

figure(8);
plot(te_vect*1e6,real(echo3),'r-'); hold on;
plot(te_vect*1e6,imag(echo3),'r--');
plot(te_vect*1e6,real(echo5),'m-');
plot(te_vect*1e6,imag(echo5),'m--');
xlabel('Time (us)');
legend('\Delta\phi = \pi/2 (real)','\Delta\phi = \pi/2 (imag)',...
    '\Delta\phi = 3\pi/2 (real)','\Delta\phi = 3\pi/2 (imag)');

% Calculate SNR
echo_int=zeros(5,1);
echo_int(1)=trapz(te_vect,abs(echo1).^2);
echo_int(2)=trapz(te_vect,abs(echo2).^2);
echo_int(3)=trapz(te_vect,abs(echo3).^2);
echo_int(4)=trapz(te_vect,abs(echo4).^2);
echo_int(5)=trapz(te_vect,abs(echo5).^2);
echo_int=echo_int/echo_int(1); % Normalize to rectangular case