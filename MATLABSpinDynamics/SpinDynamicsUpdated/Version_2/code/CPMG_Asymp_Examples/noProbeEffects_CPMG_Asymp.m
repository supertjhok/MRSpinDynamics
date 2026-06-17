% NOPROBEEFFECTS_CPMG_ASYMP
% Calculate asymptotic magnetization and echo for a CPMG sequence with no
% probe-circuit effects.
%
% Purpose
%   Demonstrates the ideal-probe CPMG asymptotic workflow. The script builds
%   the ideal system parameters, computes asymptotic magnetization, plots the
%   real and imaginary components over offset frequency, and computes the
%   corresponding time-domain echo.
%
% Inputs
%   This script takes no function arguments. It uses set_params_ideal to
%   construct the simulation and pulse-sequence parameter structures.
%
% Outputs
%   Creates a figure of M_asy versus normalized offset and leaves masy,
%   echo_asy, tvect, sp, and pp in the workspace.
%
% Key functions
%   set_params_ideal, calc_masy_ideal, calc_time_domain_echo.
%
% Notes
%   Run from a MATLAB path that includes the Version_2 code folders.
% -------------------------------------------------------------------------
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
