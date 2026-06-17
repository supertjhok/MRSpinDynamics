% Optimize CPMG excitation pulse
% Soumyajit Mandal, 09/12/10

function [texc,pexc,echo_pk]=opt_exc_pulse(nseg,T_90,NE,T_FP,T1,T2)

% Refocusing pulse definition
tref=2*T_90*[0.14 0.72 0.14]; % RP2-1.0a
pref=(pi/2)*[3,1,3];

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
start=T_90*rand(1,nseg); % Random initial condition
lb=0.1*T_90*ones(1,nseg); % Lower bound
ub=10*T_90*ones(1,nseg); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4);

% Segments have 0/180 alternating phase
pexc=zeros(1,nseg);
for i=1:2:nseg
    pexc(i)=pi;
end

% Segments have random 0/180 phase
%pexc=pi*(rand(1,nseg)>0.5);

texc=fmincon(@(params)fit_function(params,tref,pexc,pref,T_90,NE,T_FP,T1,T2),start,[],[],[],[],lb,ub,[],options);
echo_pk=cpmg_van_spin_dynamics(texc,tref,pexc,pref,T_90,NE,T_FP,T1,T2);

function val=fit_function(texc,tref,pexc,pref,T_90,NE,T_FP,T1,T2)

% Enforce symmetry
%texc_2=zeros(1,2*length(texc)-1);
%texc_2(1:length(texc))=texc;
%texc_2(length(texc)+1:2*length(texc)-1)=fliplr(texc(1:length(texc)-1));

%pexc_2=zeros(1,2*length(texc)-1);
%pexc_2(1:length(texc))=pexc;
%pexc_2(length(texc)+1:2*length(texc)-1)=fliplr(pexc(1:length(texc)-1));

%echo_pk=cpmg_van_spin_dynamics(texc_2,tref,pexc_2,pref,T_90,NE,T_FP,T1,T2);

% Don't enforce symmetry
echo_pk=cpmg_van_spin_dynamics(texc,tref,pexc,pref,T_90,NE,T_FP,T1,T2);

val=-echo_pk;