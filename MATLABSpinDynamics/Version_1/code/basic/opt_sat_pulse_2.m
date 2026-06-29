% Optimize saturation pulse - segment lengths only
% Soumyajit Mandal, 02/19/13
% --------------------------------------------------------------

function [out]=opt_sat_pulse_2(params)

tp=params.tp;
phi=params.phi;
del_w=params.del_w;
wtfun=params.wtfun; % frequency-domain weighting function for calculating saturation

start=tp;
np=length(phi);
amp=ones(1,np); % Segments have arbitrary phase and constant amplitude

% Saturation pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=zeros(1,np); % Lower bound
ub=pi*ones(1,np); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',10000);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',5000);

tp=fmincon(@(params)fit_function(params,phi,amp,wtfun,del_w),start,[],[],[],[],lb,ub,[],options);
[~,~,Mz]=sim_spin_dynamics_single_pulse(tp,phi,amp,del_w);

figure(1);
plot(del_w,abs(Mz),'k-'); hold on;

out.tp=tp;
out.phi=phi;
out.amp=amp;
out.sat=trapz(del_w,(1-abs(Mz)).*wtfun);

function val=fit_function(tp,phi,amp,wtfun,del_w)

[~,~,Mz]=sim_spin_dynamics_single_pulse(tp,phi,amp,del_w);
val=-trapz(del_w,(1-abs(Mz)).*wtfun);