close all
clear all
[sp, pp] = set_params_tuned_JMR; % Define system parameters
count = 1;
qValues = [0.1 1 100 500]
for i = 1:size(qValues,2)
    sp.Q =qValues(i);
    sp.R = 2*pi*sp.f0*sp.L/sp.Q;
    [mrx,masy]=calc_masy_tuned_probe_lp(sp,pp); % Simulate narrowband system
    [echo_rx(:,count),tvect]=calc_time_domain_echo(mrx,sp.del_w,1,1);
    title(['Time-domain echo (Q=' num2str(qValues(i)) ')']);
%     export_fig(['D:\Dropbox\TuneMatchJMR\Figures\Updated\TunedQ\' num2str(qValues(i)) '.pdf'])
    count = count+1;
end
%   mrx = (mrx - min(mrx)) / ( max(mrx) - min(mrx) )

figure;
% plot(sp.del_w,real(masy)); 
% hold on; 
% plot(sp.del_w,imag(masy));
plot(sp.del_w,real(mrx),'LineWidth',2);
hold on;
plot(sp.del_w,imag(mrx),'LineWidth',2);
title('Asymptotic magnetization')
xlabel('\Delta\omega_{0}/\omega_{1,max}')
ylabel('M_{asy}, M_{rx}')
legend('Real','Imaginary')
whiteBg
setSize
font
% export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\echoMagAsympTuned.pdf
[echo_rx,tvect]=calc_time_domain_echo(mrx,sp.del_w,1,1);