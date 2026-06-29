% CHIRP
% Generate and plot a simple square-wave chirp pulse train.
%
% Purpose
%   Demonstrates a linearly swept square-wave pulse train for quick signal
%   visualization.
%
% Inputs
%   This script takes no function arguments. Sampling rate, time axis, and
%   start/end frequencies are defined directly in the script.
%
% Outputs
%   Creates a figure of the chirp waveform and leaves fs, t, f, and x in the
%   workspace.
%
% Key functions
%   square, plot.
%
% Notes
%   Requires the Signal Processing Toolbox for square in some MATLAB
%   installations. Treat this as a utility/demo script, not a canonical
%   simulation workflow.
% -------------------------------------------------------------------------

fs = 1e3;                   % sample freq 1kHz
 D = 0 : 1/fs : 2;           % pulse delay times
 t = 0 : 1/fs : 20;          % signal evaluation time
 f1=1;
 f2=10;
 a=(f2-f1)/(t(end)-t(1));
 f=f1+a*t;
 x=square(2*pi*f.*t);
 figure; plot(t,x);
 axis([0 20 -2 2]);
 title('Train chirp pulse');
