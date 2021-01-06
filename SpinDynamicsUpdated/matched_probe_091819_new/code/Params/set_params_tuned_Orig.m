function [params,sp,pp] = set_params_tuned_Orig
% Global parameters
% --------------------------------------------
sp.k = 1.381e-23; % J/K
sp.T = 300; % Sample temperature

sp.gamma = 2*pi*42.577e6;

% System parameters
% --------------------------------------------
sp.f0 = 1e6; % Target matching frequency (= Larmor frequency), Hz
sp.fin = 1e6; % Input frequency, Hz
sp.w0 = 2*pi*sp.fin;
pp.w = sp.w0;

% Coil parameters
% --------------------------------------------
sp.L = 10e-6; % H
sp.Q = 50;
sp.R = 2*pi*sp.f0*sp.L/sp.Q;

% Tuning Params
%-------------------------------------
sp.C = 1/((2*pi*sp.f0)^2*sp.L);

% Transmitter parameters
% --------------------------------------------
sp.Rs = 1; % Series resistance, Ohms

% Receiver parameters
% --------------------------------------------
sp.Rin = 1e6; % Input impedance, Ohms
% sp.Cin = 5e-12;
sp.Cin = 5e-12;
sp.Rd = 1e6;

sp.NF = 1; % Noise figure, dB
sp.vn = 0.5e-9; 
sp.in = 2e-15; % Receiver input noise voltage [V/sqrt(Hz)] and current [A/sqrt(Hz)]

% Simulation parameters
% --------------------------------------------
sp.m0=1; % Initial magnetization
sp.mth=1; % Asymptotic / thermal magnetization
sp.numpts=1e4;  
sp.maxoffs=10;
sp.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);%Static Gradient
sp.sens=0.0296; %Coil sensitivity (T/A)

% Matched filter type
sp.mf_type=2; % 1 -> matched (white noise), 2 -> matched (colored noise)

% Plotting parameters
% --------------------------------------------
sp.plt_tx = 1; 
sp.plt_rx = 0; 
sp.plt_sequence = 0; % Plot on/off
sp.plt_axis = 0; 
sp.plt_mn = 0; 
sp.plt_echo = 1;

% Pulse sequence parameters
% --------------------------------------------
pp.N=32; % quantization step = N x input RF frequency
pp.T_90=25e-6;
pp.T_180=2*pp.T_90; % Rectangular T_90 and T_180
pp.psi=0; % Absolute RF phase at t=0
%pp.preDelay = 20e-6;
%pp.postDelay = 50e-6;
pp.preDelay = 75e-6;
pp.postDelay = 75e-6;

% Excitation pulse
pp.texc=[1]*pp.T_90; 
pp.pexc=[pi/2]; 
pp.aexc=[1];
pp.tcorr=-(2/pi)*pp.T_90; % Timing correction for excitation pulse
pp.tqs = 5e-6; % Q-switch delay
pp.trd = 5e-6; % Ring-down delay
%pp.trd = 2*pp.T_90;

% Refocusing cycle
% pp.tref=[3 1 3]*pp.T_180; 
pp.tref=[pp.preDelay pp.T_180 pp.postDelay]; 
pp.pref=[0 0 0]; 
pp.aref=[0 1 0];
pp.Rsref=[2 2 20]; %(Qsw_off, Tx_on, Qsw_on)

pp.pcycle = 1;
pp.tacq=[3]*pp.T_180; % Acquisition time for observing echo
pp.tdw=0.5e-6; % Receiver dwell time

pp.amp_zero=1e-4; % Minimum amplitude for calculations

%Params
params.texc = pp.texc;
params.pexc = pp.pexc;
params.aexc = pp.aexc;
params.trd  = pp.trd; 

params.tref = pp.tref(2);
params.pref = pp.pref(2);
params.aref = pp.aref(2);
params.tfp  = pp.preDelay; % Free precession time
params.tqs  = pp.tqs;

params.tacq = pp.tacq; % Acquisition time
params.Rs   = [pp.Rsref(1) pp.Rsref(2) pp.Rsref(3)]; %(Qsw_off, Tx_on, Qsw_on)
params.pcycle  = 1; % PAP phase cycle
end

