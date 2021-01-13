% Set simulation parameters assuming no probe dynamics 
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% Last updated: 12/30/20

function [sp, pp] = set_params_ideal

% Global parameters
% --------------------------------------------
sp.k=1.381e-23; % J/K
sp.T=300; % Sample temperature
sp.gamma = 2*pi*42.577e6;
sp.grad = 1;

%Sample parameters
sp.D = 2e-12;

% System parameters
% --------------------------------------------
sp.f0 = 0.5e6; % Target matching frequency (= Larmor frequency), Hz
sp.fin = 0.5e6; % Input frequency, Hz

% Simulation parameters
% --------------------------------------------
sp.m0=1; % Initial magnetization
sp.mth=1; % Asymptotic / thermal magnetization
%sp.numpts=2000;
sp.numpts=1e4;
sp.maxoffs=10;
sp.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts); % Static Gradient

% Matched filter type
sp.mf_type=2; % 1 -> matched (white noise), 2 -> matched (colored noise)

% Plotting parameters
% --------------------------------------------
sp.plt_tx = 0; 
sp.plt_rx = 0; 
sp.plt_sequence = 0; % Plot on/off
sp.plt_axis = 0; 
sp.plt_mn = 0; 
sp.plt_echo = 0;

% Pulse sequence parameters
% --------------------------------------------
pp.N=32; % quantization step = N x input RF frequency
pp.T_90=25e-6;
pp.T_180=2*pp.T_90; % Rectangular T_90 and T_180
pp.psi=0; % Absolute RF phase at t=0
%pp.preDelay = 20e-6;
%pp.postDelay = 70e-6;
pp.preDelay = 75e-6;
pp.postDelay = 75e-6;

% Excitation pulse
pp.texc=[1]*pp.T_90; 
pp.pexc=[pi/2]; 
pp.aexc=[1];
pp.tcorr=-(2/pi)*pp.T_90; % Timing correction for excitation pulse

% Refocusing cycle
pp.tref=[pp.preDelay pp.T_180 pp.postDelay]; 
pp.pref=[0 0 0]; 
pp.aref=[0 1 0];

pp.pcycle = 1;
pp.tacq=[3]*pp.T_180; % Acquisition time for observing echo
pp.tdw=0.5e-6; % Receiver dwell time

pp.amp_zero=1e-4; % Minimum amplitude for calculations
end