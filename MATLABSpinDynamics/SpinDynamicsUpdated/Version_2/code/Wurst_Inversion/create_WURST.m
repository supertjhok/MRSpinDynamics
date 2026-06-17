function [ f0,amp,phase,time,acq ] = create_WURST( freq,numSteps )
%CREATE_WURST Summary of this function goes here
Nw = 2;

%   Detailed explanation goes here
f0 = linspace(-freq,freq,numSteps);
tvecW = linspace(0,numSteps,numSteps)

amp = (1-abs(cos((pi.*tvecW)./(1))).^Nw);

phase = zeros(1,size(f0,2))*pi/2;


time = ones(1,size(f0,2))./numSteps;
acq = zeros(1,size(f0,2));



end

