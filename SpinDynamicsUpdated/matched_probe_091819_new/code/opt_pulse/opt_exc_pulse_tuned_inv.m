% Optimize CPMG excitation pulse to create a phase-cycle pair with
% magnetization produced by an existing pulse
% Include transmitter and receiver bandwidth effects (tuned probe)
% Written by: Soumyajit Mandal, 03/19/13
% Last modified: 01/06/21
% --------------------------------------------------------------
% params = [texc, pexc, tacq, Rs(Qsw_on, Qsw_off, Tx_on), neff, mrx, SNR] (all times normalized to w1 = 1)
% mrx -> target magnetization, SNR -> SNR of target magnetization
% --------------------------------------------------------------

function [out]=opt_exc_pulse_tuned_inv(params,sp,pp)

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
options=optimset('Algorithm','interior-point','Display','final','TolFun',1e-4,'MaxFunEvals',2e4);
%options=optimset('Algorithm','active-set','Display','final','TolFun',1e-4,'MaxFunEvals',2e4);
%options=optimset('Algorithm','sqp','Display','final','TolFun',1e-4,'MaxFunEvals',2e4);

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
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Convert acquisition time to normalized time
tacq=(pi/2)*params.tacq/T_90;

% Create excitation pulse, including delays to allow ringdown
pp.tref=[params.texc params.tqs params.trd];
pp.pref=[opt_params 0 0]; pp.aref=[params.aexc 0 0]; 
pp.Rsref=[params.Rs(2)*ones(1,length(params.texc)) params.Rs(3) params.Rs(1)];

[tvect,Icr,~,~] = tuned_probe_lp_Orig(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude
ind=find(aexc==0); pexc(ind)=0;

% Add timing correction to remove ringdown delays
texc=[texc -(params.tqs+params.trd)*(pi/2)/T_90]; 
pexc=[pexc 0]; aexc=[aexc 0];

% Calculate spin dynamics
[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,tacq);
[mrx,SNR]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver

% Optimize inversion of input magnetization
val=trapz(sp.del_w,abs(params.mrx+mrx))+0.8*abs(SNR/1e8-params.SNR);

function [SNR]=eval_function(params,sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma);
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Convert acquisition time to normalized time
tacq=(pi/2)*params.tacq/T_90;

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Turn on plots

% Create excitation pulse, including delays to allow ringdown
pp.tref=[params.texc params.tqs params.trd];
pp.pref=[params.pexc 0 0]; pp.aref=[params.aexc 0 0]; 
pp.Rsref=[params.Rs(2)*ones(1,length(params.texc)) params.Rs(3) params.Rs(1)];

[tvect,Icr,~,~] = tuned_probe_lp_Orig(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude
ind=find(aexc==0); pexc(ind)=0;

% Add timing correction to remove ringdown delays
texc=[texc -(params.tqs+params.trd)*(pi/2)/T_90]; 
pexc=[pexc 0]; aexc=[aexc 0];

% Calculate spin dynamics
[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,tacq);
[~,SNR]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver
SNR=SNR/1e8; % SNR in voltage units
