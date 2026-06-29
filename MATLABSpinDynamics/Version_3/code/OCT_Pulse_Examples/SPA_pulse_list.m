% Broadband SPA refocusing pulses from Mandal et al., JMR (2014) 
% Phases are listed below, amplitudes are constant
% Each segment has a length of 0.1*T_180

function [SPA_pulses]=SPA_pulse_list

SPA_pulses = {};

SPA_pulses{1} = pi*[1 1 0 1 0 1 0 1 1]; % 0.9 x T_180t

SPA_pulses{2} = pi*[1 1 0 0 0 0 0 0 1 1]; % 1.0 x T_180

SPA_pulses{3} = pi*[1 1 0 0 1 0 1 0 1 0 0 1 1]; % 1.3 x T_180

SPA_pulses{4} = pi*[1 0 1 0 0 1 1 0 0 0 0 0 0 1 1 0 0 1 0 1]; % 2.0 x T_180

SPA_pulses{5} = pi*[1 0 1 0 0 1 1 0 0 0 0 0 0 0 1 1 0 0 1 0 1]; % 2.1 x T_180

SPA_pulses{6} = pi*[0 1 0 1 0 0 1 1 0 1 1 1 1 1 0 0 0 1 1 1 1 1 0 1 1 0 0 1 0 1 0]; % 3.1 x T_180

SPA_pulses{7} = pi*[1 0 1 1 1 1 1 0 1 1 0 1 0 0 0 1 1 0 1 1 0 0 0 1 0 1 1 0 1 1 1 1 1 0 1]; % 3.5 x T_180

SPA_pulses{8} = pi*[0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 1 1 0 1 1 0 0 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0]; % 3.9 x T_180

SPA_pulses{9} = pi*[0 1 1 0 1 1 1 1 1 1 0 1 0 0 1 1 1 0 0 1 0 1 0 1 0 1 0 1 0 0 1 1 1 0 0 1 0 1 1 1 1 1 1 0 1 1 0]; % 4.7 x T_180

SPA_pulses{10} = pi*[1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1]; % 5.5 x T_180