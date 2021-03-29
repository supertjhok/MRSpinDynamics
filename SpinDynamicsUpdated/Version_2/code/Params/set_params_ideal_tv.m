% Set simulation parameters assuming no probe dynamics 
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% Last updated: 03/16/21

function [sp, pp] = set_params_ideal_tv

% Global parameters
% --------------------------------------------
sp.k=1.381e-23; % J/K
sp.T=300; % Sample temperature
sp.gamma = 2*pi*42.577e6; % (rad/s)/T for 1H
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

% Define static field distribution
sp.nx=1; sp.ny=1e3; sp.nz=1; 
sp.maxoffs_x=0; sp.maxoffs_y=0.1; sp.maxoffs_z=0;

% Matched filter type
sp.mf_type=2; % 1 -> matched (white noise), 2 -> matched (colored noise)
sp.numpar=8; % Number of available parallel workers in MATLAB

% Plotting parameters
% --------------------------------------------
sp.plt_tx = 0; 
sp.plt_rx = 0; 
sp.plt_sequence = 0; % Plot on/off
sp.plt_axis = 0; 
sp.plt_mn = 0; 
sp.plt_echo = 0;
sp.plt_fields = 0;
sp.plt_output = 0;

% Pulse sequence parameters
% --------------------------------------------
pp.N=32; % quantization step = N x input RF frequency
pp.T_90=25e-6;
pp.T_180=2*pp.T_90; % Rectangular T_90 and T_180
pp.psi=0; % Absolute RF phase at t=0
pp.preDelay = 125e-6;
pp.postDelay = 125e-6;

% Excitation pulse
pp.texc=[0.5]*pp.T_90; 
pp.pexc=[pi/2]; 
pp.aexc=[2];
pp.tcorr=-(2/pi)*pp.T_90/pp.aexc; % Timing correction for excitation pulse

% RP2 refocusing cycle
%pp.tref=[pp.preDelay pp.T_180*[0.14 0.72 0.14] pp.postDelay]; 
%pp.pref=[0 pi*[1 0 1] 0]; 
%pp.aref=[0 [1 1 1] 0];

% Rect refocusing cycle
pp.tref=[pp.preDelay pp.T_180 pp.postDelay]; 
pp.pref=[0 0 0]; 
pp.aref=[0 1 0];

pp.NE=800; % Number of echoes
pp.NEmin=10; % Minimum number of echoes
pp.pcycle = 1;
pp.tacq=[3]*pp.T_180; % Acquisition time for observing echo
pp.tdw=2e-6; % Receiver dwell time

pp.amp_zero=1e-4; % Minimum amplitude for calculations
end