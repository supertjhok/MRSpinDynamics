close all
[sp, pp] = set_params_matched; % Define system parameters

params = 1;
% [mrx,tvect,SNR]=calc_masy_matched_probe(sp,pp); % Simulate narrowband system
% [mrx,tvect,SNR]=calc_masy_matched_probe_WURST(sp,pp); % Simulate narrowband system

[ output_args ] = sim_inv_matched_probe_WURST( params )
% figure;
% %plot(sp.del_w,real(masy),'b--'); hold on; plot(sp.del_w,imag(masy),'r--');
% plot(sp.del_w,real(mrx),'LineWidth',2);
% hold on;
% plot(sp.del_w,imag(mrx),'LineWidth',2);
% title('Asymptotic magnetization')
% xlabel('\Delta\omega_{0}/\omega_{1,max}')
% ylabel('M_{asy}, M_{rx}')
% whiteBg
% setSize
% font
% legend('Real','Imaginary')
% % export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\echoMagAsympMatched.pdf
% 
% % Calculate time-domain echo
% [echo_rx,tvect]=calc_time_domain_echo(mrx,sp.del_w,1,1);