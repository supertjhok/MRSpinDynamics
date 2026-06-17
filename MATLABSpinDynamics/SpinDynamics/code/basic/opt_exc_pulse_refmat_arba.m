% Optimize CPMG excitation pulse, precalculate refocusing matrix
% Soumyajit Mandal, 09/21/10
% Allow arbitrary pulse amplitudes, 02/25/11

function [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_refmat_arba(nseg,refmat,del_w,w1_max,delt,start)

T_90=1e6*pi/(2*w1_max); % in us

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
if isempty(start)
    start=T_90*rand(1,nseg); % Random initial condition
end
lb=zeros(1,nseg); % Lower bound
ub=10*T_90*ones(1,nseg); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',5000);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',5000);

% Segments have 0/180 alternating phase and constant amplitude
pexc=zeros(1,nseg);
for i=1:2:nseg
    pexc(i)=pi;
end
aexc=ones(1,nseg);

texc=fmincon(@(params)fit_function(params,pexc,aexc,refmat,del_w,w1_max,delt),start,[],[],[],[],lb,ub,[],options);

outs=zeros(1,2);
[echo tvect outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,w1_max,delt);
echo_pk=outs(1);
echo_rms=outs(2);

function val=fit_function(texc,pexc,aexc,refmat,del_w,w1_max,delt)

outs=zeros(1,2);
[echo tvect outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat_arba(texc,pexc,aexc,refmat,del_w,w1_max,delt);

%val=-outs(1); %Optimize peak
%val=-outs(2)*0.3e8;  % Optimize RMS
val=-0.5*(outs(1)+outs(2)*0.3e8); % Optimize both peak and RMS