% MATCHEDWURSTINVERSION
% Run a matched-probe WURST inversion example.
%
% Purpose
%   Sets matched-probe system parameters and calls the WURST inversion
%   simulation entry point. The commented sections show earlier plotting and
%   asymptotic-magnetization checks.
%
% Inputs
%   This script takes no function arguments. It currently passes params = 1 to
%   sim_inv_matched_probe_WURST.
%
% Outputs
%   Leaves sp, pp, params, and output_args in the workspace.
%
% Key functions
%   set_params_matched, sim_inv_matched_probe_WURST.
%
% Notes
%   This appears to be an exploratory script rather than a polished example.
%   Review the params placeholder before using it as a canonical workflow.
% -------------------------------------------------------------------------
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
