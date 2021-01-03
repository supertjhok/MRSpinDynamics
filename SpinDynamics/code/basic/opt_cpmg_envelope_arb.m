% Generalize opt_cpmg_envelope to arbitrary functions

function [pvals,final_val]=opt_cpmg_envelope_arb(ne,T_E,optfun,a_init)

% Excitation pulse
texc=pi/2;
pexc=pi;
aexc=1;

% Refocusing pulse
tref0=pi;
pref0=pi/2;
aref0=1;

% Asymptotic echo
[masy,del_w]=asy_new(texc,pexc,aexc,tref0,pref0,aref0,T_E);
normasy=sqrt(trapz(del_w,abs(masy.*masy)));

T_FP=T_E-tref0; % Free precession time
t_acq=T_FP; % Acquisition time

tref=(T_FP/2)*ones(1,3*ne);
pref=zeros(1,3*ne);
aref=zeros(1,3*ne);

tref(1,2:3:3*ne-1)=tref0*ones(1,ne);
pref(1,2:3:3*ne-1)=pref0*ones(1,ne);
aref(1,2:3:3*ne-1)=aref0*ones(1,ne);

% Optimization variables
if isempty(a_init)
    a_init=2*rand(1,2*ne); % Initial values
end
a_lb=zeros(1,2*ne); % Lower bounds
a_ub=2*ones(1,2*ne); % Upper bounds

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-3,'TolX',5e-4,'MaxFunEvals',3000);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-3,'TolX',5e-4,'MaxFunEvals',3000);

pvals=fmincon(@(params)fit_function(params,texc,pexc,aexc,tref,pref,aref,...
    ne,del_w,t_acq,masy,normasy,optfun),a_init,[],[],[],[],a_lb,a_ub,[],options);

final_val=plot_function(pvals,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,masy,normasy,optfun);


function val=fit_function(params,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,masy,normasy,optfun)

% Set pulse amplitudes and durations
aref(1,2:3:3*ne-1)=aref(1,2:3:3*ne-1).*params(1:ne);
tref(1,2:3:3*ne-1)=tref(1,2:3:3*ne-1).*params(ne+1:2*ne);

% Calculate echoes with timing correction and phase cycling
[mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref,pref,aref,ne,del_w,t_acq);
[mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref,pref,aref,ne,del_w,t_acq);
mecho=(mecho1-mecho2)/2;

% Calculate error variable
val=0;
for n=1:ne
    eint=trapz(del_w,abs(mecho(n,:).*masy))/normasy;
    val=val+sqrt((eint-normasy*optfun(n))^2);
end

function val=plot_function(params,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,masy,normasy,optfun)

% Set pulse amplitudes and durations
aref(1,2:3:3*ne-1)=aref(1,2:3:3*ne-1).*params(1:ne);
tref(1,2:3:3*ne-1)=tref(1,2:3:3*ne-1).*params(ne+1:2*ne);

% Calculate echoes with timing correction and phase cycling
[mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref,pref,aref,ne,del_w,t_acq);
[mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref,pref,aref,ne,del_w,t_acq);
mecho=(mecho1-mecho2)/2;

figure(2);
% Calculate echo integrals
eint=zeros(1,ne);
val=0;
for n=1:ne
    eint(n)=trapz(del_w,abs(mecho(n,:).*masy))/normasy;
    val=val+sqrt((eint(n)-normasy*optfun(n))^2);
    
    plot(del_w+2*(n-1)*max(del_w),abs(masy),'r--'); hold on;
    plot(del_w+2*(n-1)*max(del_w),abs(mecho(n,:)));
end

figure(3);
plot(eint,'bo-');