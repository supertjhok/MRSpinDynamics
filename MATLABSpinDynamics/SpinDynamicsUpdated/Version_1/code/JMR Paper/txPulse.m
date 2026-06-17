%Tx Pulse Example
% Define pulse system parameters
clear all
% Global parameters
% --------------------------------------------

% System parameters
% --------------------------------------------
sp.f0 = 1e6; % Target matching frequency (= Larmor frequency), Hz
sp.fin = 1e6; % Input frequency, Hz

% Coil parameters
% --------------------------------------------
sp.L = 10e-6; % H
sp.Q = 50;

% Transmitter parameters
% --------------------------------------------
sp.Rs = 50; % Series resistance, Ohms

% Simulation parameters
% --------------------------------------------
sp.numpts=2000;  sp.maxoffs=10;
sp.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);

% Plotting parameters
% --------------------------------------------
sp.plt_tx = 1; 

% Tx Pulse Parameter
% --------------------------------------------
pp.N=32; % quantization step = N x input RF frequency
pp.T_90=25e-6;
pp.psi=0; % Absolute RF phase at t=0
pp.tdw=0.5e-6; % Receiver dwell time
pp.amp_zero=1e-4; % Minimum amplitude for calculations
pp.preDelay =30e-6; %Delay before pulse
pp.postDelay=50e-6; 
% ----------------------------------------------
% Run simulation
% ----------------------------------------------
[x]=plotTxPulseMatched(sp,pp);
