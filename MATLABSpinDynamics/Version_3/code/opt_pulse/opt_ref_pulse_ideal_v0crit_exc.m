% Optimize CPMG refocusing pulse to maximize critical velocity
% No transmitter and receiver bandwidth effects
% Assume specified (pre-calculated) excitation pulse vector
% Written by: Soumyajit Mandal, 03/26/21
% --------------------------------------------------------------
% params = [tref, tfp , pref, tacq] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [out]=opt_ref_pulse_ideal_v0crit_exc(params,sp,pp)

w_1n=(pi/2)/pp.T_90; % Nominal nutation frequency
tacq=w_1n*pp.tacq; % Normalized acquisition window length
window = sinc(sp.del_w*tacq/(2*pi)); % window function for acquisition
pp.window = window./sum(window);

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
[SNR,v0crit_av]=eval_function(params,sp,pp);

out.tref=params.tref;
out.pref=params.pref;
out.aref=params.aref;
out.axis_rms=SNR;
out.v0crit_av=v0crit_av;
out.params=params;
out.sp=sp;
out.pp=pp;

function [val]=fit_function(opt_params,params,sp,pp)

% Read parameters
del_w=sp.del_w; % Offset frequency vector

% Create refocusing cycle
pp.tref=[params.tfp params.tref params.tfp];
pp.pref=[0 opt_params 0]; pp.aref=[0 params.aref 0];

% Refocusing cycle
[neff,alpha]=calc_rot_axis_arba4(pp.tref,pp.pref,pp.aref,del_w,0); % Refocusing axis
[v0crit]=calc_v0crit(del_w,neff,alpha,0); % Critical velocity

% Optimize (SNR after matched filtering) + (minimum v0,crit)
mexc=pp.mexc; masy=dot(mexc,neff).*(neff(1,:)+1i*neff(2,:));
masy=conv(masy,pp.window,'same'); % Filtering by acquisition window
val=-trapz(del_w,abs(masy).^2)-1e2/(trapz(del_w,1./v0crit));
%val=-trapz(del_w,abs(masy).^2)+1e-2*(trapz(del_w,1./v0crit));

function [SNR,v0crit_av]=eval_function(params,sp,pp)

% Read parameters
del_w=sp.del_w; % Offset frequency vector

% Create refocusing cycle
pp.tref=[params.tfp params.tref params.tfp];
pp.pref=[0 params.pref 0]; pp.aref=[0 params.aref 0];

% Refocusing cycle
[neff,alpha]=calc_rot_axis_arba4(pp.tref,pp.pref,pp.aref,del_w,0); % Refocusing axis
[v0crit]=calc_v0crit(del_w,neff,alpha,0); % Critical velocity

% Optimized value of metric
mexc=pp.mexc; masy=dot(mexc,neff).*(neff(1,:)+1i*neff(2,:));
masy=conv(masy,pp.window,'same'); % Filtering by acquisition window
SNR=trapz(del_w,abs(masy).^2); % SNR after matched filtering
v0crit_av=1e2/(trapz(del_w,1./v0crit)); % Average v0,crit