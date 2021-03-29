% Optimize CPMG refocusing pulse for time-varying B0 fields
% No transmitter and receiver bandwidth effects
% Written by: Soumyajit Mandal, 03/18/21
% --------------------------------------------------------------
% params = [tref, tfp , pref, tacq] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [out]=opt_ref_pulse_ideal_tv(params,sp,pp)

nref=length(params.tref);
start=params.pref;
params.aref=ones(1,nref); % Segments have arbitrary phase and constant amplitude

% Refocusing pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=zeros(1,nref); % Lower bound
ub=2*pi*ones(1,nref); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point, sqp, or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','final','TolFun',1e-4,'MaxFunEvals',1e4);
%options=optimset('Algorithm','active-set','Display','final','TolFun',1e-4,'MaxFunEvals',2.5e4);
%options=optimset('Algorithm','sqp','Display','final','TolFun',1e-4,'MaxFunEvals',2e4);

params.pref=fmincon(@(opt_params)fit_function(opt_params,params,sp,pp),start,[],[],[],[],lb,ub,[],options);
SNR=eval_function(params,sp,pp);

out.tref=params.tref;
out.pref=params.pref;
out.aref=params.aref;
out.axis_rms=SNR;
out.params=params;
out.sp=sp;
out.pp=pp;

function [val]=fit_function(opt_params,params,sp,pp)

% Read parameters
tvect=params.tvect; % Acquisition time vector
w_1n=(pi/2)/pp.T_90; % Nominal nutation frequency

% Create refocusing cycle
pp.tref=[params.tfp params.tref params.tfp]/w_1n;
pp.pref=[0 opt_params 0]; pp.aref=[0 params.aref 0];

% Main scan
[~,echo_rx,~]=sim_cpmg_ideal_tv_final(sp,pp);

% Reference scan
pp.NE=pp.NEmin; % Mininum number of echoes
sp.B_0t=zeros(1,pp.NE); % No field fluctuation
[~,echo_rx_ref,~]=sim_cpmg_ideal_tv_final(sp,pp);
echo_rx_ref=conj(echo_rx_ref)/sqrt(trapz(tvect,abs(echo_rx_ref).^2));

% Find echo amplitude after matched filtering
echo_rms=trapz(tvect,echo_rx.*echo_rx_ref); % Estimate echo rms

% Optimize SNR
val=-real(echo_rms)/1e4;

function [SNR]=eval_function(params,sp,pp)

% Read parameters
tvect=params.tvect; % Acquisition time vector
w_1n=(pi/2)/pp.T_90; % Nominal nutation frequency

% Create refocusing cycle
pp.tref=[params.tfp params.tref params.tfp]/w_1n;
pp.pref=[0 params.pref 0]; pp.aref=[0 params.aref 0];

% Main scan
[~,echo_rx,~]=sim_cpmg_ideal_tv_final(sp,pp);

% Reference scan
pp.NE=pp.NEmin; % Mininum number of echoes
sp.B_0t=zeros(1,pp.NE); % No field fluctuation
[~,echo_rx_ref,~]=sim_cpmg_ideal_tv_final(sp,pp);
echo_rx_ref=conj(echo_rx_ref)/sqrt(trapz(tvect,abs(echo_rx_ref).^2));

% Find echo amplitude after matched filtering
echo_rms=trapz(tvect,echo_rx.*echo_rx_ref); % Estimate echo rms
SNR=real(echo_rms)/1e4;
