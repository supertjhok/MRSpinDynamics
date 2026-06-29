function [ f0,amp,phase,time,acq ] = create_WURST( freq,numSteps )
% CREATE_WURST Build a normalized WURST inversion pulse table.
%
% Signature
%   [f0,amp,phase,time,acq] = create_WURST(freq,numSteps)
%
% Inputs
%   freq - Sweep half-bandwidth in normalized frequency units.
%   numSteps - Number of piecewise-constant pulse segments.
%
% Outputs
%   f0 - Frequency sweep values from -freq to +freq.
%   amp - WURST amplitude envelope.
%   phase - RF phase for each segment.
%   time - Normalized duration of each segment.
%   acq - Acquisition flag vector; WURST pulse segments do not acquire.
% -------------------------------------------------------------------------
Nw = 2;

f0 = linspace(-freq,freq,numSteps);
tvecW = linspace(0,numSteps,numSteps);

amp = (1-abs(cos((pi.*tvecW)./(1))).^Nw);

phase = zeros(1,size(f0,2))*pi/2;


time = ones(1,size(f0,2))./numSteps;
acq = zeros(1,size(f0,2));



end

