function plot_rat(rat)

[echo_pk1,echo_rms1]=cpmg_van_spin_dynamics_plot_rat...
    ('rectangular','rectangular',20,rat,10,1000,100,100);
[echo_pk2,echo_rms2]=cpmg_van_spin_dynamics_plot_rat...
    ('CP-M8','RP2-1.0a',20,rat,10,1000,100,100);
[echo_pk3,echo_rms3]=cpmg_van_spin_dynamics_plot_rat...
    ('CP-M15','RP2-1.0a',20,rat,10,1000,100,100);

close all;
ind=find(rat==2); % Normalize to rectangular, T_180/T_90=2

plot(rat,echo_rms1/echo_rms1(ind),'b-');
hold on;
plot(rat,echo_rms2/echo_rms1(ind),'r-');
plot(rat,echo_rms3/echo_rms1(ind),'k-');
xlabel('T_{180} / T_{90}');
ylabel('Squared integral of echo');