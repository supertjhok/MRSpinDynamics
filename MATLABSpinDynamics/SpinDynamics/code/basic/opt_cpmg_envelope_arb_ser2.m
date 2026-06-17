% Generalize opt_cpmg_envelope to arbitrary functions
% Optimize each echo individually
% Do not use default echo as a matched filter. Instead, define amplitude of
% each echo by calculating its mean squared value. However, optfun is still
% normalized to the default echo for convenience.

function [pvals,final_val]=opt_cpmg_envelope_arb_ser2(ne,T_E,optfun,a_init)

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

optfun=optfun*normasy; % Scale optfun to asymptotic echo

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
a_ub=5*ones(1,2*ne); % Upper bounds

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',5e-4,'TolX',5e-4,'MaxFunEvals',3000);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',5e-4,'TolX',5e-4,'MaxFunEvals',3000);

numpts=length(del_w);

m0=1;
mvect1=zeros(3,1,numpts); mvect2=mvect1; % Magnetization vectors
mvect1(1,1,:)=m0*ones(1,1,numpts); % Initial mag vectors are along z-axis
mvect2(1,1,:)=m0*ones(1,1,numpts);

% Optimize each echo separately
pvals=zeros(1,2*ne);
for echonum=1:ne
    a_init0=[a_init(echonum) a_init(echonum+ne)];
    a_lb0=[a_lb(echonum) a_lb(echonum+ne)];
    a_ub0=[a_ub(echonum) a_ub(echonum+ne)];
    tref0=tref(3*(echonum-1)+1:3*echonum);
    pref0=pref(3*(echonum-1)+1:3*echonum);
    aref0=aref(3*(echonum-1)+1:3*echonum);
    
    tmp=fmincon(@(params)fit_function(params,texc,pexc,aexc,tref0,pref0,aref0,...
        del_w,t_acq,optfun,mvect1,mvect2,echonum),a_init0,[],[],[],[],a_lb0,a_ub0,[],options);
    pvals(echonum)=tmp(1);
    pvals(echonum+ne)=tmp(2);
    
    % Update magnetization vectors
    aref0(2)=aref0(2)*pvals(echonum);
    tref0(2)=tref0(2)*pvals(echonum+ne);
    if echonum==1
        [~,mvect1]=sim_spin_dynamics_arba_echoes2([texc -1/aexc(1)],[pexc 0],[aexc 0],tref0,pref0,aref0,1,del_w,t_acq);
        [~,mvect2]=sim_spin_dynamics_arba_echoes2([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref0,pref0,aref0,1,del_w,t_acq);
    else
        [~,mvect1]=sim_spin_dynamics_arba_oneecho(tref0,pref0,aref0,del_w,t_acq,mvect1);
        [~,mvect2]=sim_spin_dynamics_arba_oneecho(tref0,pref0,aref0,del_w,t_acq,mvect2);
    end
    disp(['Finished optimizing echo #' num2str(echonum)])
end

final_val=plot_function(pvals,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,optfun);


function val=fit_function(params,texc,pexc,aexc,tref,pref,aref,del_w,...
    t_acq,optfun,mvect1,mvect2,echonum)

% Set pulse amplitudes and durations
aref(2)=aref(2)*params(1);
tref(2)=tref(2)*params(2);

% Calculate echoes with timing correction and phase cycling
if echonum==1
    [mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref,pref,aref,1,del_w,t_acq);
    [mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref,pref,aref,1,del_w,t_acq);
else
    [mecho1,~]=sim_spin_dynamics_arba_oneecho(tref,pref,aref,del_w,t_acq,mvect1);
    [mecho2,~]=sim_spin_dynamics_arba_oneecho(tref,pref,aref,del_w,t_acq,mvect2);
end
mecho=(mecho1-mecho2)/2;

% Calculate error variable
eint=sqrt(trapz(del_w,abs(mecho.*mecho)));
val=sqrt((eint-optfun(echonum))^2);

function val=plot_function(params,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,optfun)

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
    eint(n)=sqrt(trapz(del_w,abs(mecho(n,:).*mecho(n,:))));
    val=val+sqrt((eint(n)-optfun(n))^2);
    
    plot(del_w+2*(n-1)*max(del_w),abs(mecho(n,:))); hold on;
end

figure(5);
plot(eint,'bo-');