% Optimize CPMG excitation pulse, precalculate refocusing matrix
% Soumyajit Mandal, 09/21/10
% Add single-pulse echo, variable excitation-refocusing delay 02/15/11

function [texc,pexc,T_ER,spe_rms,echo_rms]=opt_exc_pulse_refmat_spe(nseg,T_90,T_FP,refmat,tpar,start)

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
if isempty(start)
    start=T_90*rand(1,nseg+1); % Random initial condition
end
start(nseg+1)=T_FP/2; % Final variable is excitation-refocusing delay (T_ER)

lb=zeros(1,nseg+1); % Lower bound

ub=10*T_90*ones(1,nseg+1); % Upper bound
ub(nseg+1)=T_FP;

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',5000);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',5000);

% Segments have 0/180 alternating phase
pexc=zeros(1,nseg);
for i=1:2:nseg
    pexc(i)=pi;
end

params=fmincon(@(params)fit_function(params,pexc,T_90,refmat,tpar),start,[],[],[],[],lb,ub,[],options);
texc=params(1:nseg);
T_ER=params(nseg+1);

outs=zeros(1,4);
[outs(1) outs(2) outs(3) outs(4)]=cpmg_van_spin_dynamics_refmat_spe(texc,pexc,T_90,T_ER,refmat,tpar);
spe_pk=outs(1);
spe_rms=outs(2);
echo_pk=outs(3);
echo_rms=outs(4);

function val=fit_function(params,pexc,T_90,refmat,tpar)

texc=params(1:length(params)-1);
T_ER=params(end);

outs=zeros(1,4);
[outs(1) outs(2) outs(3) outs(4)]=cpmg_van_spin_dynamics_refmat_spe(texc,pexc,T_90,T_ER,refmat,tpar);

%val=-(outs(2)+outs(4))*1e1;  % Optimize RMS sum
val=-outs(2)*outs(4)*1e2;  % Optimize RMS product