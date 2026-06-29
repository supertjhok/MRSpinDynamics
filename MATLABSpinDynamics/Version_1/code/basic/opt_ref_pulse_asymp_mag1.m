% Optimize CPMG refocusing pulse, even number of segments
% Soumyajit Mandal, 02/27/13
% --------------------------------------------------------------
% Assume pulses with interal anti-symmetry
% --------------------------------------------------------------

function [out]=opt_ref_pulse_asymp_mag1(params)

tfp=params.tfp;
tref=params.tref;
pref=params.pref;

del_w=params.del_w; numpts=length(del_w);
opt_window=params.opt_window; % window function for optimization
tacq=params.tacq;
window = sinc(del_w*tacq/(2*pi)); % window function for acquisition

nref=length(tref);
start=pref;
aref=ones(1,nref); % Segments have arbitrary phase and constant amplitude

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=-2*pi*zeros(1,nref); % Lower bound
ub=2*pi*ones(1,nref); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point, sqp, or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',2.5e4);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',2.5e4);
%options=optimset('Algorithm','sqp','Display','iter','TolFun',1e-4,'MaxFunEvals',2e4);

pref=fmincon(@(params)fit_function(params,tref,aref,tfp,del_w,opt_window,window),start,[],[],[],[],lb,ub,[],options);
trefc=[tfp tref tref tfp]; % Create refocusing cycle
prefc=[0 pref -fliplr(pref) 0];
arefc=[0 aref aref 0];
[neff]=calc_rot_axis_arba2(trefc,prefc,arefc,del_w,0);
nx=dot(neff,neff).*neff(1,:); fy = conv(nx,window);
nx = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(window);

out.tref=tref;
out.pref=pref;
out.aref=aref;
out.axis_rms=sqrt(trapz(del_w,abs(nx).^2));

function val=fit_function(params,tref,aref,tfp,del_w,opt_window,window)

numpts=length(del_w);
trefc=[tfp tref tref tfp]; % Create refocusing cycle
prefc=[0 params -fliplr(params) 0];
arefc=[0 aref aref 0];
[neff]=calc_rot_axis_arba2(trefc,prefc,arefc,del_w,0);
nx=dot(neff,neff).*neff(1,:); fy = conv(abs(nx),window);
%nx = fy(((numpts+1)/2:3*(numpts-1)/2+1)).*opt_window./sum(window);
nx = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(window);

% Optimize refocusing axis
%val=-sqrt(trapz(del_w,abs(nx).^2));

% Add penalty for asymmetric spectrum
val=-sqrt(trapz(del_w,abs(nx).^2))+0.5*sqrt(trapz(del_w,(abs(nx-fliplr(nx))).^2));