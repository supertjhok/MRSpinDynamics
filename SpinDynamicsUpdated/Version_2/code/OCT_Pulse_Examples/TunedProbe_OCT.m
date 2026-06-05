% TUNEDPROBE_OCT
% Generate and evaluate broadband OCT pulses for tuned-probe CPMG-like
% sequences.
%
% Purpose
%   Optimizes refocusing and excitation pulses for a tuned probe, plots
%   refocusing-pulse performance, asks the user to choose a refocusing pulse,
%   and then optimizes excitation pulses against that choice.
%
% Inputs
%   This script takes no function arguments. It prompts interactively for the
%   refocusing pulse number. File names and pulse lengths are set near the top
%   of the script.
%
% Outputs
%   Writes optimization result MAT-files using rfilname, efilname, and
%   efilname_inv. Also creates plots through the optimization plotting helpers.
%
% Key functions
%   opt_ref_pulse_tuned_repeat, plot_opt_ref_results_tuned,
%   opt_exc_pulse_tuned_repeat.
%
% Notes
%   Change the output file names before rerunning if existing optimization
%   results should not be overwritten.
%
% Written by: Soumyajit Mandal, 01/07/21
% -------------------------------------------------------------------------

% Set optimization parameters
% ----------------------------------------------------------------------
numiter = 24; % Number of optimization runs
% WARNING: This variable is not automatically set. Make sure it is
% consistent across all the optimization functions.

ref_len = 1.5; % Refocusing pulse length (units of T_180)
T_E = 7*pi; % Echo spacing (normalized)
exc_len = 8*pi; % Excitation pulse length (units of T_180)

% File names for storing refocusing and excitation pulse results
% WARNING: Change these to avoid your old results from getting overwritten!
% ----------------------------------------------------------------------
rfilname = ['ref_tuned_500k_' num2str(ref_len)];
efilname = ['exc_tuned_500k_' num2str(ref_len)]; % Original AMEX
efilname_inv = [efilname '_inv']; % Inverse AMEX

% Optimize refocusing pulse
% ----------------------------------------------------------------------
opt_ref_pulse_tuned_repeat(ref_len,T_E,rfilname);

% Plot SNR for all refocusing pulses, and detailed results for a random pulse
% ----------------------------------------------------------------------
pulse_num = round(numiter*rand(1));
[neff,SNR_ref] = plot_opt_ref_results_tuned(rfilname,pulse_num);

% Ask the user to select the refocusing pulse used to generate an AMEX
% excitation pulse
% ----------------------------------------------------------------------
rpulse_num = input(['Select the refocusing pulse number [1-' num2str(numiter) ']: ']);

% Optimize excitation pulse
% WARNING: This step will take a while
% ----------------------------------------------------------------------
opt_exc_pulse_tuned_repeat(rfilname,rpulse_num,exc_len,efilname)

% Plot SNR for all excitation pulses, and detailed results for a random pulse
% ----------------------------------------------------------------------
pulse_num = round(numiter*rand(1));
[mrx,masy,echo_rx,tvect,SNR_exc]=plot_opt_exc_results_tuned(efilname,pulse_num);

% Generate inverse pulses for all the optimized excitation pulses
% WARNING: This step will take a while. 
% To reduce run time, replace 'linspace(1,numiter,numiter)' below with a shorter
% vector containing the pulse numbers you really want to optimize
% ----------------------------------------------------------------------
opt_exc_pulse_tuned_inv_repeat(efilname,linspace(1,numiter,numiter),efilname_inv);
