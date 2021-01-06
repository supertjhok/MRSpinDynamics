% Example of generating broadband OCT excitation and refocusing pulses and
% evaluating their performance

ref_len = 1.5; % Refocusing pulse length (units of T_180)
T_E = 7*pi; % Echo spacing (normalized)
filname = ['ref_tuned_500k_' num2str(ref_len) '.mat']; % File name for storing results

% Optimize refocusing pulse
opt_ref_pulse_tuned_repeat(ref_len,T_E,filname);