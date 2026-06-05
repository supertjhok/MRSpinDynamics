% TUNEDPROBEEFFECTS_CPMG_ASYMP
% Calculate asymptotic magnetization and echo for a CPMG sequence with a
% tuned probe.
%
% Purpose
%   Demonstrates the tuned-probe CPMG asymptotic workflow. The script builds
%   tuned-probe parameters, computes asymptotic and received magnetization,
%   plots both spectra, and computes the received time-domain echo.
%
% Inputs
%   This script takes no function arguments. It uses set_params_tuned_Orig
%   to construct the circuit, simulation, and pulse-sequence parameters.
%
% Outputs
%   Prints SNR, creates a figure of M_asy and M_rx versus normalized offset,
%   and leaves params, sp, pp, mrx, masy, SNR, echo_rx, and tvect in the
%   workspace.
%
% Key functions
%   set_params_tuned_Orig, calc_masy_tuned_probe_lp_Orig,
%   calc_time_domain_echo.
%
% Notes
%   The received magnetization is divided by Q for plotting clarity.
% -------------------------------------------------------------------------
%close all
 
%[sp, pp] = set_params_tuned_JMR; % Define system parameters
%[mrx,masy,SNR]=calc_masy_tuned_probe_lp(sp,pp); % Simulate narrowband system

[params,sp,pp] = set_params_tuned_Orig; % Define system parameters
[mrx,masy,SNR] = calc_masy_tuned_probe_lp_Orig(params,sp,pp);

SNR

figure;
plot(sp.del_w,real(masy),'LineWidth',1);
hold on;
plot(sp.del_w,imag(masy),'LineWidth',1);

% Divide received magnetization by peak TF gain (=Q) for clarity
Grx_max=sp.Q;
plot(sp.del_w,real(mrx)/Grx_max,'LineWidth',1);
plot(sp.del_w,imag(mrx)/Grx_max,'LineWidth',1);

title('Asymptotic magnetization')
xlabel('\Delta\omega_{0}/\omega_{1,max}')
ylabel('M_{asy}, M_{rx}')
legend({'Real(M_{asy})','Imag(M_{asy})','Real(M_{rx})','Imag(M_{rx})'})
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
% export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\echoMagAsympTuned.pdf

%figure; 
%plot(sp.del_w,abs(mrx))
%title('abs(M_{rx})');

% Plot time-domain echo
[echo_rx,tvect]=calc_time_domain_echo(mrx/Grx_max,sp.del_w,1,1);
