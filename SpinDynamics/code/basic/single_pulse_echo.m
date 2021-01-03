function single_pulse_echo(T_90,tp,T1,T2)

phi=0;
tf=320;

[echo,tvect]=sim_spin_dynamics_allpw(T_90,tp,phi,tf,T1,T2);
tvect=tvect*1e6+tf;

close all;
figure(1);
plot(tvect,real(echo),'b-'); hold on;
plot(tvect,imag(echo),'r-');
plot(tvect,abs(echo),'k-');