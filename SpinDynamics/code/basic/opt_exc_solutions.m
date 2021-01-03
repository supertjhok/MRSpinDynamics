% Using active-set algorithm or interior-point algorithm
% Using alternating 0/180 phase
% Without symmetry constraint

function opt_exc_solutions

close all;
T_90=100; % Rectangular 90 pulse length, us
T_FP=1000; % Free precession time, us
NE=10; % Number of echoes to simulate
pk=[]; rms=[]; name={}; count=1;
results={};

T1=100; T2=100; % ms

% Rectangular - rectangular (reference case)
T_180=2*T_90;
[echo,tvect]=cpmg_van_spin_dynamics_echo(T_90,T_180,0,pi/2,T_90,NE,T_FP,T1,T2);
echo_rect=abs(echo);

results{count,1}='Rectangular';
results{count,2}=1;
results{count,3}=0;
results{count,4}=max(echo_rect);
results{count,5}=trapz(tvect,echo_rect.^2);
count=count+1;

figure(2);
plot(tvect*1e6,echo_rect/max(echo_rect),'b--','LineWidth',2); hold on;

% Optimal pulse sequences

RP2_tp=2*T_90*[0.14 0.72 0.14]; % RP2-1.0a refocusing pulse
RP2_ph=pi*[3,1,3]/2;

% Excitation pulses

% Simultaneous peak/RMS optimization

%2 peaks
texc=[24.0882 12.8072 10.9593 22.4468 11.7351 2.6679 2.6997 34.4875 9.5512...
    5.7400 17.5549 2.0054 10.9230 10.9493 5.6538 4.6649 22.6085 15.0626 19.0203 3.1246]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416...
    0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M10',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% 1 peak
texc=[19.2227 6.0393 2.0866 10.0998 16.4463 29.8969 16.0887 6.0887 19.3662 2.1919...
    21.3240 2.7375 6.5507 16.1753 16.3436 2.0088 13.0869 6.4862 10.6676 2.7684]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416...
    0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M9',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% 1 peak
texc=[3.9639 16.9383 10.6496 14.9261 50.9267 16.7695 2.1138 32.6648 13.2419 18.7537]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M12',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

texc=[11.5069 6.7740 17.9808 9.2441 10.5111 50.8009 9.8460 7.7639 21.6700 10.7587]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M13',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% 1 peak
texc=[5.8892 11.8304 11.6280 5.4449 5.9596 12.3659 13.6616 9.9422 9.4375 11.7715...
    18.5484 14.4181 23.0698 8.3836 34.4977 2.6896 12.2816 2.0261 17.6360 15.3105]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416...
    0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M8',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% Peak optimization
% CP-M7
texc=[24.9805 18.6140 16.6194 39.8424 3.1616 5.1289 19.9711 13.8811 10.8652 2.8484]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M7',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% CP-M6
texc=[6.1197 16.5651 2.0414 2.0490 11.5938 2.2779 9.6843 22.1612 7.5433 4.9491...
    15.6606 15.3331 12.7730 20.1323 23.3122 8.3837 5.2463 19.5875 3.9401 3.4790]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416...
    0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M6',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% CP-M5
texc=[6.5344 11.4933 13.4345 10.9366 8.6173 8.6514 13.2264 13.9620 17.5239...
    14.7454 12.8518 6.7818 6.4270 13.3218 18.2601]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416];
[results,count]=addtoresults(results,count,'CP-M5',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% CP-M4
texc=[2.1344 10.6309 2.8523 2.3213 14.6117 19.4037 21.3717 18.1949 3.6197 4.2407]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M4',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% CP-M3
texc=[10.3373 17.0938 21.5759 18.2558 12.3376 3.4453 8.6927 17.3486 2.6076 4.2928]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M3',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% CP-M2
texc=[5.1680 10.1744 11.1102 12.3010 10.3418 8.3121 7.3921 7.9705 13.0997 16.3253]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M2',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

%CP-M1
texc=[4.8830 3.4198 4.1772 13.4830 18.1684 20.1003 21.1427 17.5803 3.2594 4.7709]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M1',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% Variable phase
texc=[5.5590 39.7647 27.0829 8.8147 9.0547 2.0241 14.9240 7.6207 26.4273 5.2660]/20;
pexc=[4.7030 1.3188 5.9643 6.2507 1.9416 6.2113 1.6989 2.5577 0.0796 3.1798];
[results,count]=addtoresults(results,count,'CP-M11',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

% Optimization in the presence of B1 inhomogeneity

% del_w1=0.7
texc=[2.1355 9.6901 12.0290 10.0615 13.8844 22.3770 25.1689 14.0356 5.3030...
    15.0943 20.2627 13.2032 14.3871 12.5364 2.0090 11.4801 24.2996 7.1089 3.490 25.9166]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416...
    0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M14',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

texc=[4.5181 12.4048 6.0959 2.2792 7.0175 13.5255 19.8885 9.9264 12.2873 4.5133...
    2.9271 25.5894 6.9146 10.1824 33.3112 11.0890 8.7904 24.5275 13.4434 18.5024]/20;
pexc=[3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416 0 3.1416...
    0 3.1416 0 3.1416 0];
[results,count]=addtoresults(results,count,'CP-M15',texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect);

save exc_timings.mat results

% With symmetry constraint - nothing as good for the same total number of
% segments

% With random 0/180 phase - nothing as good for the same total number of
% segments

function [echo_pk,echo_rms]=plot_fun(texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect,clr)

figure(1);
plot(texc/2,clr,'LineWidth',2); hold on;

[echo,tvect]=cpmg_van_spin_dynamics_echo(T_90*texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2);
echo_pk=max(abs(echo));
echo_rms=trapz(tvect,abs(echo).^2);

% Normalize to rectangular case
echo=abs(echo)/max(echo_rect);

set(gca,'FontSize',14);
xlabel('Segment number');
ylabel('Segment length (units of T_{180})');
axis([1 15 0 0.8]);

figure(2);
plot(tvect*1e6,echo,clr,'LineWidth',2); hold on;

set(gca,'FontSize',14);
xlabel('Time (\mus)');
ylabel('Normalized echo shape');

function [results,count]=addtoresults(results,count,name,texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect)

[pk,rms]=plot_fun(texc,RP2_tp,pexc,RP2_ph,T_90,NE,T_FP,T1,T2,echo_rect,'b-');
results{count,1}=name;
results{count,2}=texc;
results{count,3}=pexc;
results{count,4}=pk;
results{count,5}=rms;
count=count+1;