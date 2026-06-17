% Optimize CPMG excitation pulse to create a phase-cycle pair with
% magnetization produced by an existing pulse
% Include transmitter and receiver bandwidth effects
% Soumyajit Mandal, 03/19/13
% --------------------------------------------------------------
% params = [texc, pexc, tacq, Rs(off,on), neff, mrx, SNR] (all times normalized to w1 = 1)
% mrx -> target magnetization, SNR -> SNR of target magnetization
% --------------------------------------------------------------

function [out]=opt_exc_pulse_circsim_inv(params,sp,pp)

sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off plots

nexc=length(params.texc);
start=params.pexc;

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=-2*pi*zeros(1,nexc); % Lower bound
ub=2*pi*ones(1,nexc); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point, sqp, or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',2e4);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',2.5e4);
%options=optimset('Algorithm','sqp','Display','iter','TolFun',1e-4,'MaxFunEvals',2e4);

params.pexc=fmincon(@(opt_params)fit_function(opt_params,params,sp,pp),start,[],[],[],[],lb,ub,[],options);
SNR=eval_function(params,sp,pp);

out.texc=params.texc;
out.pexc=params.pexc;
out.aexc=params.aexc;
out.axis_rms=SNR;
out.sp=sp;
out.pp=pp;

function [val]=fit_function(opt_params,params,sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma); 

% Create excitation pulse
pp.tref=params.texc*T_90/(pi/2);
pp.pref=opt_params; pp.aref=params.aexc; 
pp.Rsref=params.Rs(2)*ones(1,length(opt_params));

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

[tvect,Icr] = tuned_probe(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude

% Calculate spin dynamics
[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,params.tacq);
[mrx,SNR]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver

% Optimize inversion of input magnetization
val=trapz(sp.del_w,abs(params.mrx+mrx))+0.8*abs(SNR/1e8-params.SNR);

function [SNR]=eval_function(params,sp,pp)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma); 

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Turn on plots

% Create excitation pulse
pp.tref=params.texc*T_90/(pi/2);
pp.pref=params.pexc; pp.aref=params.aexc; 
pp.Rsref=params.Rs(2)*ones(1,length(params.pexc));

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

[tvect,Icr] = tuned_probe(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude

% Calculate spin dynamics
[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,params.tacq);
[~,SNR]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver
SNR=SNR/1e8;
