
%SET_PARAMS_UNTUNED Summary of this function goes here
%   Detailed explanation goes here
function [sp,pp] = set_params_untuned_JMR
% Global parameters
% --------------------------------------------
% sp.k=1.381e-23; % J/K
% sp.T=300; % Sample temperature

% System parameters
% --------------------------------------------
sp.f0 = 5e6; % Target matching frequency (= Larmor frequency), Hz
sp.fin = 1e6; % Input frequency, Hz
sp.w0 = 2*pi*sp.fin;
pp.w = sp.w0;

% Coil parameters
% --------------------------------------------
sp.L = 10e-6; % H
sp.Q = 50;
sp.R = 2*pi*sp.f0*sp.L/sp.Q;

%TUning Params
%-------------------------------------

sp.C = 1/((2*pi*sp.f0)^2*sp.L);

% Transmitter parameters
% --------------------------------------------
sp.Rs = 2; % Series resistance, Ohms
pp.Rsref = [2 2 2];

% Receiver parameters
% --------------------------------------------
sp.Rin = 2; % Input impedance, Ohms
sp.NF = 1; % Noise figure, dB


% Simulation parameters
% --------------------------------------------
sp.m0=1; % Initial magnetization
sp.mth=1; % Asymptotic / thermal magnetization
sp.numpts=2000;  
sp.maxoffs=100;
sp.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);%Static Gradient

% Matched filter type
sp.mf_type=2; % 1 -> matched (white noise), 2 -> matched (colored noise)

% Plotting parameters
% --------------------------------------------
sp.plt_tx = 1; 
sp.plt_rx = 1; 
sp.plt_sequence = 1; % Plot on/off
sp.plt_axis = 1; 
sp.plt_mn = 1; 
sp.plt_echo = 1;

% Pulse sequence parameters
% --------------------------------------------
pp.N=32; % quantization step = N x input RF frequency
pp.T_90=25e-6;
pp.T_180=pp.T_90; % Rectangular T_90 and T_180
pp.psi=0; % Absolute RF phase at t=0
pp.preDelay = 30e-6;
pp.postDelay = 50e-6;
% Excitation pulse
pp.texc=[1]*pp.T_90; 
pp.pexc=[pi/2]; 
pp.aexc=[1];
pp.tcorr=-(2/pi)*pp.T_90; % Timing correction for excitation pulse

% Refocusing cycle
pp.tref=[pp.preDelay pp.T_180 pp.postDelay]; 
pp.pref=[0 0 0]; 
pp.aref=[0 1 0];

pp.tacq=[5]*pp.T_180; % Acquisition time for observing echo
pp.tdw=0.5e-6; % Receiver dwell time

pp.amp_zero=1e-4; % Minimum amplitude for calculations
end