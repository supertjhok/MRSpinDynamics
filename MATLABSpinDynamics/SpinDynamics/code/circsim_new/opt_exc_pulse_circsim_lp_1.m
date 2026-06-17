% Optimize CPMG excitation pulse
% Include transmitter and receiver bandwidth effects, tuned probe
% Soumyajit Mandal, 03/19/13
% --------------------------------------------------------------
% params = [texc, pexc, tacq, Rs(Qsw_on, Qsw_off, Tx_on), neff] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [out]=opt_exc_pulse_circsim_lp_1(params,sp,pp)

sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off plots

nexc=length(params.texc);
start=params.pexc;
params.aexc=ones(1,nexc); % Segments have arbitrary phase and constant amplitude

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=-2*pi*zeros(1,nexc); % Lower bound
ub=2*pi*ones(1,nexc); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point, sqp, or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',1.5e4);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',2.5e4);
%options=optimset('Algorithm','sqp','Display','iter','TolFun',1e-4,'MaxFunEvals',2e4);

params.pexc=fmincon(@(opt_params)fit_function(opt_params,params,sp,pp),start,[],[],[],[],lb,ub,[],options);
SNR=eval_function(params,sp,pp);

out.texc=params.texc;
out.pexc=params.pexc;
out.aexc=params.aexc;
out.axis_rms=SNR;
out.params=params;
out.sp=sp;
out.pp=pp;

function [val]=fit_function(opt_params,params,sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma); 

% Create excitation pulse, including delays to allow ringdown
pp.tref=[params.texc params.tqs params.trd]*T_90/(pi/2);
pp.pref=[opt_params 0 0]; pp.aref=[params.aexc 0 0]; 
pp.Rsref=[params.Rs(3)*ones(1,length(params.texc)) params.Rs(2) params.Rs(1)];

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

[tvect,Icr] = tuned_probe_lp(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude

% Add timing correction to remove ringdown delays
texc=[texc -(params.tqs+params.trd)]; pexc=[pexc 0]; aexc=[aexc 0];

% Calculate spin dynamics
[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,params.tacq);
[~,SNR]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver

% Optimize SNR
val=-SNR/1e8;

function [SNR]=eval_function(params,sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma); 

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Turn on plots

% Create excitation pulse, including delays to allow ringdown
pp.tref=[params.texc params.tqs params.trd]*T_90/(pi/2);
pp.pref=[params.pexc 0 0]; pp.aref=[params.aexc 0 0]; 
pp.Rsref=[params.Rs(3)*ones(1,length(params.texc)) params.Rs(2) params.Rs(1)];

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

[tvect,Icr] = tuned_probe_lp(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude

% Add timing correction to remove ringdown delays
texc=[texc -(params.tqs+params.trd)]; pexc=[pexc 0]; aexc=[aexc 0];

% Calculate spin dynamics
[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,params.tacq);
[~,SNR]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver
SNR=SNR/1e8;
