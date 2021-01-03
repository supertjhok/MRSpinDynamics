% Only use k amplitude levels: first (k-1) refocusing pulses, and then all other
% refocusing pulses
function [avals]=opt_cpmg_envelope_simple(ne,T_E,k)

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
aref=pref;

tref(1,2:3:3*ne-1)=tref0*ones(1,ne);
pref(1,2:3:3*ne-1)=pref0*ones(1,ne);

% Optimization variables
a_init=2*rand(1,k); % Initial values
a_lb=zeros(1,k); % Lower bounds
a_ub=2*ones(1,k); % Upper bounds

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-3,'MaxFunEvals',5000);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-3,'MaxFunEvals',5000);

avals=fmincon(@(params)fit_function(params,texc,pexc,aexc,tref,pref,aref,...
    ne,del_w,t_acq,masy,normasy),a_init,[],[],[],[],a_lb,a_ub,[],options);

plot_function(avals,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,masy,normasy);


function val=fit_function(params,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,masy,normasy)

% Set pulse amplitudes and durations
k=length(params);
for j=1:k-1
aref(1,2+3*(j-1))=params(j);
tref(1,2+3*(j-1))=tref(1,2+3*(j-1))/aref(1,2+3*(j-1));
end

aref(1,2+3*(k-1):3:3*ne-1)=params(k)*ones(1,ne-(k-1));
tref(1,2+3*(k-1):3:3*ne-1)=tref(1,2+3*(k-1):3:3*ne-1)./aref(1,2+3*(k-1):3:3*ne-1);

% Calculate echoes with timing correction and phase cycling
[mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref,pref,aref,ne,del_w,t_acq);
[mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref,pref,aref,ne,del_w,t_acq);
mecho=(mecho1-mecho2)/2;

% Calculate error variable
val=0;
for n=1:ne
    eint=trapz(del_w,abs(mecho(n,:).*masy))/normasy;
    val=val+sqrt((eint-normasy)^2);
end

function plot_function(params,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,masy,normasy)

% Set pulse amplitudes and durations
k=length(params);
for j=1:k-1
aref(1,2+3*(j-1))=params(j);
tref(1,2+3*(j-1))=tref(1,2+3*(j-1))/aref(1,2+3*(j-1));
end

aref(1,2+3*(k-1):3:3*ne-1)=params(k)*ones(1,ne-(k-1));
tref(1,2+3*(k-1):3:3*ne-1)=tref(1,2+3*(k-1):3:3*ne-1)./aref(1,2+3*(k-1):3:3*ne-1);

% Calculate echoes with timing correction and phase cycling
[mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref,pref,aref,ne,del_w,t_acq);
[mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref,pref,aref,ne,del_w,t_acq);
mecho=(mecho1-mecho2)/2;

figure(2);
% Calculate echo integrals
eint=zeros(1,ne);
for n=1:ne
    eint(n)=trapz(del_w,abs(mecho(n,:).*masy))/normasy;
    plot(del_w+2*(n-1)*max(del_w),abs(masy),'r--'); hold on;
    plot(del_w+2*(n-1)*max(del_w),abs(mecho(n,:)));
end

figure(3);
plot(eint,'bo-');