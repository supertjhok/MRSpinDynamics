% SENSITIVITYPARAMETER
% Sweep a tuned-probe hard-refocusing sensitivity parameter.
%
% Purpose
%   Evaluates plot_hardRef_tuned_probe_lp over a range of sensitivity values
%   and plots the resulting SNR curve.
%
% Inputs
%   This script takes no function arguments. sens and rectangular-pulse
%   settings are defined directly in the script.
%
% Outputs
%   Creates an SNR-versus-sensitivity figure and leaves sens and SNRhard in the
%   workspace.
%
% Key functions
%   plot_hardRef_tuned_probe_lp.
%
% Notes
%   This is an exploratory parameter-sensitivity utility.
% -------------------------------------------------------------------------

sens = 1e-5:1e-5:1e-3;
SNRhard = zeros(length(sens));
for j = 1:length(sens)
    vars.sens = sens(j);
    vars.rat = 74/37; %t180/t90
    vars.len = 1;
    vars.techo=15*2*pi/2;
     [masy,SNRhard(j)] = plot_hardRef_tuned_probe_lp(vars);
end

figure
plot(sens,SNRhard)
