close all
% 
[sp, pp] = set_params_tuned_JMR; % Define system parameters
[mrx,masy,SNR]=calc_masy_tuned_probe_lp(sp,pp); % Simulate narrowband system
% [params, sp, pp] = set_params_tuned_Orig; % Define system parameters
% [mrx,masy]=calc_masy_tuned_probe_lp_orig(params,sp,pp); % Simulate narrowband system

%   mrx = (mrx - min(mrx)) / ( max(mrx) - min(mrx) ) 0.952-0.9191
SNR
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
figure 
plot(sp.del_w,abs(mrx))
title('abs(mrx)');
[echo_rx,tvect]=calc_time_domain_echo(mrx,sp.del_w,1,1);