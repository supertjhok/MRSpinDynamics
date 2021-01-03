% Optimize CPMG excitation pulse, precalculate refocusing matrix
% Allow excitation pulse to have arbitrary phase relative to the refocusing
% pulse
% Soumyajit Mandal, 09/21/10

function [texc,pexc,delph,echo_pk,echo_rms]=opt_exc_pulse_refmat_arbphase(nseg,T_90,T_FP,refmat,delt,start)

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
if isempty(start)
    start=[10*T_90*rand(1,nseg) 2*pi*rand(1,1)]; % Random initial condition
end
lb=[0.1*T_90*ones(1,nseg) 0]; % Lower bound
ub=[10*T_90*ones(1,nseg) 2*pi]; % Upper bound

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

soln=fmincon(@(params)fit_function(params,pexc,T_90,T_FP,refmat,delt),start,[],[],[],[],lb,ub,[],options);
texc=soln(1:nseg);
delph=soln(nseg+1);

outs=zeros(1,2);
[outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat(texc,pexc+delph,T_90,T_FP,refmat,delt);
echo_pk=outs(1);
echo_rms=outs(2);

function val=fit_function(params,pexc,T_90,T_FP,refmat,delt)

nseg=length(pexc);
texc=params(1:nseg);
delph=params(nseg+1);

outs=zeros(1,2);
[outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat(texc,pexc+delph,T_90,T_FP,refmat,delt);

%val=-outs(1); %Optimize peak
%val=-outs(2)*0.3e8;  % Optimize RMS
val=-0.5*(outs(1)+outs(2)*0.3e8); % Optimize both peak and RMS