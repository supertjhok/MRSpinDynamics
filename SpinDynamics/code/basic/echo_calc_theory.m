function [echo_pk,echo_rms]=echo_calc_theory 

%close all;
i=sqrt(-1);

T_90=20;

% Experiments 34 - 41
T_FP_4=8*T_90; % Free precession time, pref_4
T_FP_1=14*T_90; % Free precession time, pref_1, rect
T_FP_2=15.5*T_90; % Free precession time, rect/4

% Experiments 42 - 49
%T_FP_4=4.5*T_90; % Free precession time, pref_4
%T_FP_1=10.5*T_90; % Free precession time, pref_1, rect
%T_FP_2=12*T_90; % Free precession time, rect/4

N_E=10; % Number of echoes

tmp=load('exc_timings.mat');
results=tmp.results; % CP-M15
texc=T_90*results{16,2};
pexc=results{16,3};

tmp=load('wave\longerpulses.mat'); % PULSE_4
pulse_4=tmp.pulse_4;
comp_4=pulse_4(:,1)+i*pulse_4(:,2);
tref_4=T_90*0.2*ones(1,40);
pref_4=phase(comp_4)+pi/2;
aref_4=abs(comp_4)./max(abs(comp_4));

tref_1=T_90*2*[0.14 0.72 0.14]; % RP2-1.0a
pref_1=(pi/2)*[1,3,1];
aref_1=ones(1,3);

% These 8 cases correspond to experiments 34 - 41 (or 42 - 49) in
% longer_refocusing_pulses\cpmg_oneshot_sp
echo_pk=zeros(8,1); echo_rms=echo_pk;

[echo_pk(1),echo_rms(1)]=cpmg_arba_plot(T_90,0,1,2*T_90,pi/2,1,T_90,N_E,T_FP_1);
[echo_pk(2),echo_rms(2)]=cpmg_arba_plot(texc,pexc,ones(1,20),tref_4,pref_4,aref_4,T_90,N_E,T_FP_4);
[echo_pk(3),echo_rms(3)]=cpmg_arba_plot(T_90/10,0,10,tref_4,pref_4,aref_4,T_90,N_E,T_FP_4);

figure(1);
xlim([-3 3]);
legend('rect / rect','CP-M15 / PULSE-4','ideal / PULSE-4');
input('Press any key')

close all;
[echo_pk(1),echo_rms(1)]=cpmg_arba_plot(T_90,0,1,2*T_90,pi/2,1,T_90,N_E,T_FP_1);
[echo_pk(7),echo_rms(7)]=cpmg_arba_plot(texc,pexc,ones(1,20),tref_1,pref_1,aref_1,T_90,N_E,T_FP_1);
[echo_pk(8),echo_rms(8)]=cpmg_arba_plot(T_90/10,0,10,tref_1,pref_1,aref_1,T_90,N_E,T_FP_1);

figure(1);
xlim([-3 3]);
legend('rect / rect','CP-M15 / RP2-1.0','ideal / RP2-1.0');
input('Press any key')

close all;
[echo_pk(2),echo_rms(2)]=cpmg_arba_plot(texc,pexc,ones(1,20),tref_4,pref_4,aref_4,T_90,N_E,T_FP_4);
[echo_pk(4),echo_rms(4)]=cpmg_arba_plot(texc/2,pexc,2*ones(1,20),tref_4,pref_4,aref_4,T_90,N_E,T_FP_4);
[echo_pk(5),echo_rms(5)]=cpmg_arba_plot(texc/4,pexc,4*ones(1,20),tref_4,pref_4,aref_4,T_90,N_E,T_FP_4);

figure(1);
xlim([-3 3]);
legend('0 dB / 0 dB','6 dB / 0 dB','12 dB / 0 dB');
title('CP-M15 / PULSE-4');
input('Press any key')

close all;
[echo_pk(1),echo_rms(1)]=cpmg_arba_plot(T_90,0,1,2*T_90,pi/2,1,T_90,N_E,T_FP_1);
[echo_pk(6),echo_rms(6)]=cpmg_arba_plot(T_90/4,0,4,2*T_90/4,pi/2,4,T_90,N_E,T_FP_2);

figure(1);
xlim([-3 3]);
legend('0 dB / 0 dB','12 dB / 12 dB');
title('rect / rect');

echo_rms/echo_rms(1)