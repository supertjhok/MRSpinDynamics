close all
clear all

[sp, pp] = set_params_matched; % Define system parameters
[masy]=calc_masy_ideal(sp,pp); % Simulate ideal system


figure;
%plot(sp.del_w,real(masy),'b--'); hold on; plot(sp.del_w,imag(masy),'r--');
plot(sp.del_w,real(masy),'LineWidth',2);
hold on;
plot(sp.del_w,imag(masy),'LineWidth',2);
title('Asymptotic magnetization')
xlabel('\Delta\omega_{0}/\omega_{1,max}')
ylabel('M_{asy}, M_{rx}')
whiteBg
setSize
font
legend('Real','Imaginary')
export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\echoMagAsympIdeal.pdf
% Calculate time-domain echo
[echo_asy,tvect]=calc_time_domain_echo(masy,sp.del_w,1,1);