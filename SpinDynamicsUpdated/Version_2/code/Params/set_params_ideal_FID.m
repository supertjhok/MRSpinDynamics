function [sp,pp] = set_params_ideal_FID
%SET_PARAMS_IDEAL_FID Summary of this function goes here
%   Detailed explanation goes here

T1 = 2000e-3;
T2 = 2000e-3;
% Global parameters
% --------------------------------------------
sp.k=1.381e-23; % J/K
sp.T=300; % Sample temperature

% System parameters
% --------------------------------------------
sp.f0 = 10e6; % Target matching frequency (= Larmor frequency), Hz
sp.fin = 10e6; % Input frequency, Hz


% Simulation parameters
% --------------------------------------------
sp.m0=1; % Initial magnetization
sp.mth=1; % Asymptotic / thermal magnetization

sp.numpts=20000; 
sp.maxoffs=10;
sp.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);
sp.w_1=ones(1,sp.numpts); % Uniform transmit w_1
sp.w_1r=ones(1,sp.numpts); % Uniform receiver sensitivity
sp.T1=T1*ones(1,sp.numpts);
sp.T2=T2*ones(1,sp.numpts); % Relaxation constants



% Matched filter type
sp.mf_type=1; % 1 -> matched (white noise), 2 -> matched (colored noise)

% Plotting parameters
% --------------------------------------------
sp.plt_tx = 0; 
sp.plt_rx = 0; 
sp.plt_sequence = 0; % Plot on/off
sp.plt_axis = 0; 
sp.plt_mn = 1;
sp.plt_echo = 1;

% Pulse sequence parameters
% --------------------------------------------
pp.N=32; % quantization step = N x input RF frequency
pp.T_90=25e-6;

pp.acqDelay = pp.T_90/10;
pp.acqTpTime = pp.T_90;

pp.psi=0; % Absolute RF phase at t=0a


pp.tacq=pp.T_90; % Acquisition time for observing echo
pp.tdw=0.5e-6; % Receiver dwell time

pp.amp_zero=1e-4; % Minimum amplitude for calculations
end

