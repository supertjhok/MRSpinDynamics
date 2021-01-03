% Optimize CPMG excitation and refocusing pulses pulse
% Soumyajit Mandal, 09/12/10

function [texc,pexc,tref,pref,echo_pk,echo_rms]=opt_exc_ref_pulses(nexc,nref,T_90,NE,T_FP,delt)

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
start=T_90*rand(1,nexc+nref); % Random initial condition
lb=0.1*T_90*ones(1,nexc+nref); % Lower bound
ub=[10*T_90*ones(1,nexc) 2*T_90*ones(1,nref)]; % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4);

% Segments have 0/180 alternating phase
pexc=zeros(1,nexc); pref=zeros(1,nref);
for i=1:2:nexc
    pexc(i)=pi;
end

for i=1:2:nref
    pref(i)=pi/2;
    if i<nref
        pref(i+1)=3*pi/2;
    end
end

% Total refocusing pulse length < 2*T_90
A=zeros(nexc+nref);
A(:,nexc+1:nexc+nref)=ones(nexc+nref,nref);
b=2*T_90*ones(nexc+nref,1);

soln=fmincon(@(params)fit_function(params,nexc,pexc,nref,pref,T_90,NE,T_FP,delt),start,A,b,[],[],lb,ub,[],options);
texc=soln(1:nexc); tref=soln(nexc+1:nexc+nref);

outs=zeros(1,2);
[outs(1) outs(2)]=cpmg_van_spin_dynamics_norelax(texc,tref,pexc,pref,T_90,NE,T_FP,delt);
echo_pk=outs(1);
echo_rms=outs(2);

function val=fit_function(params,nexc,pexc,nref,pref,T_90,NE,T_FP,delt)

outs=zeros(1,2);
texc=params(1:nexc);
tref=params(nexc+1:nexc+nref);
[outs(1) outs(2)]=cpmg_van_spin_dynamics_norelax(texc,tref,pexc,pref,T_90,NE,T_FP,delt);

%val=-outs(1); %Optimize peak
%val=-outs(2)*0.3e8;  % Optimize RMS
val=-0.5*(outs(1)+outs(2)*0.3e8); % Optimize both peak and RMS