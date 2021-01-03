% Create rectangular RF pulse, Ncyc cycles
% Assignment of Tx states:
% (1) negative pulse, (2) zero, (3) positive pulse, (4) Q-switch

function resonant_tx_thirdorder_rect_pulse(Ncyc)

%close all;

% Simulation parameters
params.del_tf=2*pi*0.01;
params.wn = params.del_tf/pi; % Normalized RF frequency (sampling frequency = 2)
[b,a] = butter(4,0.7*params.wn); % Butterworth LPF, cutoff at 0.7*(RF frequency)
params.btr_b = b; params.btr_a =a;
[gd,~] = grpdelay(b,a); % Group delay of Butterworth LPF
grd = gd(1)*params.del_tf; % Group delay at low frequencies
params.grd = grd;

% Probe parameters
params.R1=0.01; params.L1=0.5e-6; params.Q0=50;

% Transmitter parameters
params.Rs=[2 2 2 2];  params.Ls=0.5e-6*[1 1 1 1]; params.VBB=[-100 0 100 0];

% Create a rectangular pulse, Ncyc cycles long
ttran=2*pi*linspace(0,Ncyc*5,Ncyc*5+1)/5; % 5 transition times per cycle
ttran(Ncyc*5+2)=2*ttran(Ncyc*5+1); % Q-switch period

% Phase modulation
%ttran(end/2:end)=ttran(end/2:end)+pi/2;

txstat=2*ones(1,Ncyc*5); % Zero states
txstat(2:5:Ncyc*5-3)=3; txstat(4:5:Ncyc*5-1)=1; % On and off states
txstat(5*Ncyc+1)=4; % Q-switch state

params.ttran=ttran; params.txstat=txstat;

% Plotting parameters
params.plt_tx=1; params.clr='r-';

[out]=resonant_tx_thirdorder2(params); % Find coil current
[outr] = resonant_tx_thirdorder_rotframe2(params,out); % Convert to rotating frame