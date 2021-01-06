% Calculate asymptotic magnetization and echo for a CPMG sequence assuming
% no probe effects
% -----------------------------------------------------------------------
close all;
clear all;

[sp, pp] = set_params_ideal; % Define system parameters
[masy] = calc_masy_ideal(sp,pp); % Simulate ideal system

figure;
%plot(sp.del_w,real(masy),'b--'); hold on; plot(sp.del_w,imag(masy),'r--');
plot(sp.del_w,real(masy),'LineWidth',1);
hold on;
plot(sp.del_w,imag(masy),'LineWidth',1);
title('Asymptotic magnetization')
xlabel('\Delta\omega_{0}/\omega_{1,max}')
ylabel('M_{asy}')
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
legend('Real','Imaginary')
% export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\echoMagAsympIdeal.pdf

% Calculate and plot time-domain echo
[echo_asy,tvect]=calc_time_domain_echo(masy,sp.del_w,1,1);