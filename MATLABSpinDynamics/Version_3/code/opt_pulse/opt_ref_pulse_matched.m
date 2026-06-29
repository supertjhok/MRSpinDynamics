% Optimize CPMG refocusing pulse
% Include transmitter and receiver bandwidth effects, matched probe
% Written by: Soumyajit Mandal, 01/04/21
% Last modified: 01/04/21
% --------------------------------------------------------------
% pp = [tref, tfp , pref, aref, tacq] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [out]=opt_ref_pulse_matched(sp,pp)

sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; % Turn off plots

% Convert acquisition time to normalized time
tacq=(pi/2)*pp.tacq/pp.T_90;

del_w=sp.del_w;
window = sinc(del_w*tacq/(2*pi)); % window function for acquisition
window=window./sum(window);

nref=length(pp.tref);
start=pp.pref;
pp.aref=ones(1,nref); % Segments have arbitrary phase and constant amplitude

% Refocusing pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=-2*pi*zeros(1,nref); % Lower bound
ub=2*pi*ones(1,nref); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point, sqp, or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','final','TolFun',1e-4,'MaxFunEvals',1e4);
%options=optimset('Algorithm','active-set','Display','final','TolFun',1e-4,'MaxFunEvals',2.5e4);
%options=optimset('Algorithm','sqp','Display','final','TolFun',1e-4,'MaxFunEvals',2e4);

pp.pref=fmincon(@(opt_params)fit_function(opt_params,sp,pp,window),start,[],[],[],[],lb,ub,[],options);
SNR=eval_function(sp,pp,window);

out.tref=pp.tref;
out.pref=pp.pref;
out.aref=pp.aref;
out.axis_rms=SNR;
out.sp=sp;
out.pp=pp;

function [val]=fit_function(opt_params,sp,pp,window)

% Create refocusing cycle
pp.tref=[pp.tfp pp.tref pp.tfp];
pp.pref=[0 opt_params 0]; pp.aref=[0 pp.aref 0]; 

% Calculate refocusing axis
sp.plt_tx=0; 
[neff]=calc_rot_axis_matched_probe(sp,pp);

nx=neff(1,:)+1i*neff(2,:); nx = conv(abs(nx),window,'same');
[~,~,~,SNR]=matched_probe_rx(sp,pp,nx,sp.tf1,sp.tf2); % Filtering by matched receiver

% Optimize SNR
val=-SNR;

function [SNR]=eval_function(sp,pp,window)

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Turn on plots

% Create refocusing cycle
pp.tref=[pp.tfp pp.tref pp.tfp];
pp.pref=[0 pp.pref 0]; pp.aref=[0 pp.aref 0]; 

% Calculate refocusing axis
%sp.plt_tx=0; 
[neff]=calc_rot_axis_matched_probe(sp,pp);

nx=neff(1,:)+1i*neff(2,:); nx = conv(abs(nx),window,'same');
[~,~,~,SNR]=matched_probe_rx(sp,pp,nx,sp.tf1,sp.tf2); % Filtering by matched receiver