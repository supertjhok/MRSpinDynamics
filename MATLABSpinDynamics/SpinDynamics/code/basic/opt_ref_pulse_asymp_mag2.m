% Optimize CPMG refocusing pulse, odd number of segments
% Soumyajit Mandal, 02/27/13
% --------------------------------------------------------------
% Assume pulses with interal anti-symmetry
% --------------------------------------------------------------

function [out]=opt_ref_pulse_asymp_mag2(params)

tfp=params.tfp;
tref=params.tref;
pref=params.pref;

del_w=params.del_w;
opt_window=params.opt_window; % window function for optimization
tacq=params.tacq;
wndw = sinc(del_w*tacq/(2*pi)); % window function for acquisition
wndw = wndw./sum(wndw);

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

pref=fmincon(@(params)fit_function(params,tref,aref,tfp,del_w,opt_window,wndw),start,[],[],[],[],lb,ub,[],options);
trefc=[tfp/2 tref tref(1:nref-1) tfp/2]; % Create refocusing cycle
prefc=[0 pref -fliplr(pref(1:nref-1)) 0];
arefc=[0 aref aref(1:nref-1) 0];
[neff]=calc_rot_axis_arba3(trefc,prefc,arefc,del_w,0);
nx = neff(1,:); nx = conv(nx,wndw,'same');

out.tref=tref;
out.pref=pref;
out.aref=aref;
out.axis_rms=sqrt(trapz(del_w,abs(nx).^2));

function val=fit_function(params,tref,aref,tfp,del_w,opt_window,wndw)

nref=length(tref);
trefc=[tfp/2 tref tref(1:nref-1) tfp/2]; % Create refocusing cycle
prefc=[0 params -fliplr(params(1:nref-1)) 0];
arefc=[0 aref aref(1:nref-1) 0];
[neff]=calc_rot_axis_arba3(trefc,prefc,arefc,del_w,0);
nx = neff(1,:); %nz = neff(3,:);
nxw = conv(nx,wndw,'same'); % Acquisition window

% Optimize refocusing axis
%val=-sqrt(trapz(del_w,abs(nxw).^2));

% Add penalty for asymmetric spectrum
val=-sqrt(trapz(del_w,abs(nxw).^2))+0.5*sqrt(trapz(del_w,(abs(nx-fliplr(nx))).^2));%+...
   % +0.5*sqrt(trapz(del_w,(abs(nz+fliplr(nz))).^2));