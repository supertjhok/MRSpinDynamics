% Optimize CPMG excitation pulse, precalculate refocusing matrix, include RF
% inhomogeneity
% Soumyajit Mandal, 09/21/10

function [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_refmat_delw1(nseg,T_90,del_w1,T_FP,refmat,delt,start)

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
if isempty(start)
    start=T_90*rand(1,nseg); % Random initial condition
end
lb=0.1*T_90*ones(1,nseg); % Lower bound
ub=10*T_90*ones(1,nseg); % Upper bound

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

% Segments have random 0/180 phase
%pexc=pi*(rand(1,nseg)>0.5);

texc=fmincon(@(params)fit_function(params,pexc,T_90,del_w1,T_FP,refmat,delt),start,[],[],[],[],lb,ub,[],options);

outs=zeros(1,2);
[outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat_delw1(texc,pexc,T_90,del_w1,T_FP,refmat,delt);
echo_pk=outs(1);
echo_rms=outs(2);

function val=fit_function(texc,pexc,T_90,del_w1,T_FP,refmat,delt)

outs=zeros(1,2);
[outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat_delw1(texc,pexc,T_90,del_w1,T_FP,refmat,delt);

%val=-outs(1); %Optimize peak
%val=-outs(2)*0.3e8;  % Optimize RMS
val=-0.5*(outs(1)+outs(2)*0.3e8); % Optimize both peak and RMS