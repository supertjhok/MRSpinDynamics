function [params,sp,pp] = set_params_tuned_Orig
% Global parameters
% --------------------------------------------
sp.k=1.381e-23; % J/K
sp.T=300; % Sample temperature

sp.gamma = 42.577e6*2*pi;

% System parameters
% --------------------------------------------
sp.f0 = 1e6; % Target matching frequency (= Larmor frequency), Hz
sp.fin = 1e6; % Input frequency, Hz
sp.w0 = 2*pi*sp.fin;
pp.w = sp.w0;

% Coil parameters
% --------------------------------------------
sp.L = 10e-6; % H
sp.Q =500;
sp.R = 2*pi*sp.f0*sp.L/sp.Q;

%TUning Params
%-------------------------------------

sp.C = 1/((2*pi*sp.f0)^2*sp.L);

% Transmitter parameters
% --------------------------------------------
sp.Rs = 1; % Series resistance, Ohms
pp.Rsref = [1 1 1];

% Receiver parameters
% --------------------------------------------
sp.Rin = 1000000000; % Input impedance, Ohms
% sp.Cin = 5e-12;
sp.Cin = 5e-12;
sp.Rd = 1000000;
sp.NF = 1; % Noise figure, dB
sp.vn=0.1e-9; 
sp.in=0.1e-12; % Non-resonant receiver input noise voltage [V/sqrt(Hz)] and current [A/sqrt(Hz)]


% Simulation parameters
% --------------------------------------------
sp.m0=1; % Initial magnetization
sp.mth=1; % Asymptotic / thermal magnetization
sp.numpts=2000;  
sp.maxoffs=10;
sp.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);%Static Gradient
sp.sens =1;
% Matched filter type
sp.mf_type=1; % 1 -> matched (white noise), 2 -> matched (colored noise)

sp.tsetup=10e-6; % simulation time before tx pulses (sec)
sp.tdecay=50e-6; % simulation time after tx pulses (sec)

% Plotting parameters
% --------------------------------------------
sp.plt_tx = 0; 
sp.plt_rx = 0; 
sp.plt_sequence = 0; % Plot on/off
sp.plt_axis = 0; 
sp.plt_mn = 0; 
sp.plt_echo = 1;



% Pulse sequence parameters
% --------------------------------------------
pp.N=32; % quantization step = N x input RF frequency
pp.T_90=50e-6;
pp.T_180=2*pp.T_90; % Rectangular T_90 and T_180
pp.psi=0; % Absolute RF phase at t=0
pp.preDelay = 0;
pp.postDelay = 0;

% Excitation pulse
pp.texc=[1]*pp.T_90; 

pp.pexc=[pi/2]; 
pp.aexc=[1];
pp.tcorr=-(2/pi)*pp.T_90; % Timing correction for excitation pulse
pp.tqs = 1e-6;
pp.trd = 2e-6;

% Refocusing cycle
% pp.tref=[3 1 3]*pp.T_180; 
% pp.pref=[0 0 0]; 
% pp.aref=[0 1 0];
pp.tref=[1]*pp.T_180; 
pp.pref=[0]; 
pp.aref=[1];


pp.pcycle = 1;
pp.tacq=[3]*pp.T_180; % Acquisition time for observing echo
pp.tdw=0.5e-6; % Receiver dwell time

pp.amp_zero=1e-4; % Minimum amplitude for calculations

%Paramsss
params.texc = pp.texc;
params.pexc = pp.pexc;
params.aexc = pp.aexc;
params.tref = pp.tref;
params.pref = pp.pref;
params.aref = pp.aref;
params.tfp  = 3*pp.T_180;
params.tqs  = 10e-6;  %Qswitch
params.tacq = pp.tacq;
params.Rs   = [10 100000000 100000000] %(Qsw_on,Qsw_off,Tx_on)
params.pcycle  = 1;
end

