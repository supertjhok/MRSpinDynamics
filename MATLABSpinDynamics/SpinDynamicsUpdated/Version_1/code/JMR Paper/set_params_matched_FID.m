


% Set parameters for a tuned-and-matched probe
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19

function [sp, pp] = set_params_matched_FID

sp.k=1.381e-23; % J/K
sp.T=300; % Sample temperature
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

% Receiver parameters
% --------------------------------------------
sp.Rin = 50; % Input impedance, Ohms
sp.NF = 1; % Noise figure, dB

% Simulation parameters
% --------------------------------------------
sp.m0=1; % Initial magnetization
sp.mth=1; % Asymptotic / thermal magnetization
sp.numpts=2000;  
sp.maxoffs=50;
sp.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);
sp.w_1 = ones(size(sp.del_w));
sp.w_1r = sp.w_1;
sp.T1 = 2000;
sp.T2 = 2000;
% Matched filter type
sp.mf_type=1; % 1 -> matched (white noise), 2 -> matched (colored noise)

% Plotting parameters
% --------------------------------------------
sp.plt_tx = 0; sp.plt_rx = 0; sp.plt_sequence = 1; % Plot on/off
sp.plt_axis = 0;  sp.plt_mn = 0; sp.plt_echo = 0;

% Pulse sequence parameters
% --------------------------------------------
pp.N=32; % quantization step = N x input RF frequency
pp.T_90=25e-6;

pp.psi=0; % Absolute RF phase at t=0

% Excitation pulse
pp.texc=[1]*pp.T_90; 
pp.pexc=[pi/2];
pp.aexc=[1];
pp.tcorr=-(2/pi)*pp.T_90; % Timing correction for excitation pulse

pp.tacq=[100]*pp.T_90; % Acquisition time for observing echo
pp.tdw=1000e-6; % Receiver dwell time

pp.amp_zero=1e-4; % Minimum amplitude for calculations

end
