function plot_varyte_theory(T_90,NE,T1,T2)

numpts=20;
T_FP=2*T_90*linspace(2,40,numpts);
echo_pks=zeros(3,numpts); echo_rms=echo_pks;

for i=1:numpts
    [echo_pks(1,i),echo_rms(1,i)]=cpmg_van_spin_dynamics_plot('rectangular',T_90,NE,T_FP(i),T1,T2);
    [echo_pks(2,i),echo_rms(2,i)]=cpmg_van_spin_dynamics_plot('CP-M8',T_90,NE,T_FP(i),T1,T2);
    [echo_pks(3,i),echo_rms(3,i)]=cpmg_van_spin_dynamics_plot('CP-M15',T_90,NE,T_FP(i),T1,T2);
    disp(i)
end

% Normalize by rectangular case
for i=3:-1:1
    echo_rms(i,:)=echo_rms(i,:)./echo_rms(1,:);
end

close all;
open('cpmg_oneshot_sp_tevar.fig'); % Experimental results

rat=T_FP/(2*T_90);

plot(rat,echo_rms(2,:),'b--'); hold on;
plot(rat,echo_rms(3,:),'k--');
%xlabel('T_{FP} / T_{180}')
%ylabel('Asymptotic echo amplitude (normalized)');
%legend('CP\_M8','CP\_M15');