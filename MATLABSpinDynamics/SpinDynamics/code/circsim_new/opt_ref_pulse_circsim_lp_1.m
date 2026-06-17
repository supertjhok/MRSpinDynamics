% Optimize CPMG refocusing pulse
% Include transmitter and receiver bandwidth effects, tuned probe
% Soumyajit Mandal, 03/19/13
% --------------------------------------------------------------
% params = [tref, tfp , pref, tacq, Rs(Qsw on, Qsw off, Tx on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [out]=opt_ref_pulse_circsim_lp_1(params,sp,pp)

sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off plots

del_w=sp.del_w;
window = sinc(del_w*params.tacq/(2*pi)); % window function for acquisition
window=window./sum(window);

nref=length(params.tref);
start=params.pref;
params.aref=ones(1,nref); % Segments have arbitrary phase and constant amplitude

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=-2*pi*zeros(1,nref); % Lower bound
ub=2*pi*ones(1,nref); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point, sqp, or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',1e4);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',2.5e4);
%options=optimset('Algorithm','sqp','Display','iter','TolFun',1e-4,'MaxFunEvals',2e4);

params.pref=fmincon(@(opt_params)fit_function(opt_params,params,sp,pp,window),start,[],[],[],[],lb,ub,[],options);
SNR=eval_function(params,sp,pp,window);

out.tref=params.tref;
out.pref=params.pref;
out.aref=params.aref;
out.axis_rms=SNR;
out.params=params;
out.sp=sp;
out.pp=pp;

function [val]=fit_function(opt_params,params,sp,pp,window)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma); 

% Create refocusing cycle
pp.tref=[params.tfp params.tref params.tqs (params.tfp-params.tqs)]*T_90/(pi/2);
pp.pref=[0 opt_params 0 0]; pp.aref=[0 params.aref 0 0]; 
pp.Rsref=[params.Rs(1) params.Rs(3)*ones(1,length(params.tref)) params.Rs(2) params.Rs(1)];

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

[tvect,Icr] = tuned_probe_lp(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
trefc=delt*ones(1,length(tvect));
prefc=atan2(imag(Icr),real(Icr));
arefc=abs(Icr)*sp.sens/B1max;
arefc(arefc<amp_zero)=0; % Threshold amplitude

[neff]=calc_rot_axis_arba3(trefc,prefc,arefc,sp.del_w,sp.plt_axis);
nx=neff(1,:)+1i*neff(2,:); nx = conv(abs(nx),window,'same');
[~,SNR]=tuned_probe_rx(sp,pp,nx); % Filtering by tuned receiver

% Optimize SNR
val=-SNR/1e8;

function [SNR]=eval_function(params,sp,pp,window)

T_90=pp.T_90; % Rectangular T_90 time
B1max=(pi/2)/(T_90*sp.gamma); 

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Turn on plots

% Create refocusing cycle
pp.tref=[params.tfp params.tref params.tqs (params.tfp-params.tqs)]*T_90/(pi/2);
pp.pref=[0 params.pref 0 0]; pp.aref=[0 params.aref 0 0]; 
pp.Rsref=[params.Rs(1) params.Rs(3)*ones(1,length(params.tref)) params.Rs(2) params.Rs(1)];

amp_zero=pp.amp_zero; % Minimum amplitude for calculations

[tvect,Icr] = tuned_probe_lp(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
trefc=delt*ones(1,length(tvect));
prefc=atan2(imag(Icr),real(Icr));
arefc=abs(Icr)*sp.sens/B1max;
arefc(arefc<amp_zero)=0; % Threshold amplitude

[neff]=calc_rot_axis_arba3(trefc,prefc,arefc,sp.del_w,sp.plt_axis);
nx=neff(1,:)+1i*neff(2,:); nx = conv(abs(nx),window,'same');
[~,SNR]=tuned_probe_rx(sp,pp,nx); % Filtering by tuned receiver
SNR=SNR/1e8;