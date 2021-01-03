function [sp,pp]=set_params_lp_tapped

% Fundamental constants
% --------------------------------------------
sp.k=1.381e-23; % J/K

% Spin parameters
% --------------------------------------------
sp.gamma=2*pi*42.57e6; % proton, (rad/s)/T

% Frequency scaling parameter
% --------------------------------------------
sp.w0n=2*pi*0.257e6; % Nominal frequency = 257 kHz
sp.fscale=1; % Frequency scaling factor, normalized to nominal frequency

% Transmitter parameters referred to the "tap"
% --------------------------------------------
sp.Ntap = 8; % Autotransformer tap ratio
sp.L = 372.3e-9;
sp.C =  1/(sp.L*(2*pi*0.257*1e6)^2);
sp.R = 7.6e-3;

sp.Ron = 5; %Rs for Tx on
sp.Roff = 5; %Rs for Tx off, Q switch on

% Receiver parameters
% --------------------------------------------
sp.Rd=7e2/(sp.Ntap^2); % Damping resistance [not damped, damped] [Ohms]
sp.Rin=1e6; sp.Cin=10e-12; % Input resistance and capacitance [Ohms, F]
sp.Nrx=4; % Turns ratio of input noise-matching transformer
sp.vn=0.5e-9/sp.Nrx; sp.in=0.1e-12*sp.Nrx; % Resonant receiver input noise voltage [V/sqrt(Hz)] and current [A/sqrt(Hz)]
%sp.vn=0.1e-9; sp.in=0.1e-12; % Non-resonant receiver input noise voltage [V/sqrt(Hz)] and current [A/sqrt(Hz)]

% Sample parameters
% --------------------------------------------
sp.w0=sp.w0n*sp.fscale; % Larmor frequency
sp.T=300; % Sample temperature

% Simulation parameters
% --------------------------------------------
sp.numpts=2001;  sp.maxoffs=10;
sp.del_w=linspace(-sp.maxoffs,sp.maxoffs,sp.numpts);
% sp.sens=3.3e-3*sp.fscale/sp.Ntap; % Coil sensitivity x input voltage, (T/A) x V
sp.sens=2.100e-04;
sp.mf_type=2; % 0 -> flat in time, 1 -> matched (white noise), 2 -> matched (colored noise)

% Plotting parameters
% --------------------------------------------
sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1;

% Pulse sequence parameters
% --------------------------------------------
pp.w=2*pi*0.257e6*sp.fscale; 
pp.N=16/sp.fscale; % RF frequency, number of quantization steps per cycle
% pp.T_90=40e-6; pp.T_180=70e-6; % Rectangular T_90 and T_180
pp.T_90=37e-6; pp.T_180=74e-6; % Rectangular T_90 and T_180
pp.NumPhases = 32; %number of phases when discretizing ref pulse

%pp.tref=pp.T_180*[3 0.14 0.72 0.14 3]; pp.pref=pi*[0 1 0 1 0]; pp.aref=[0 1 1 1 0]; pp.Rsref=[10 4 4 4 10];
%pp.tacq=4*pp.T_180;

pp.amp_zero=1e-4; % Minimum amplitude for calculations