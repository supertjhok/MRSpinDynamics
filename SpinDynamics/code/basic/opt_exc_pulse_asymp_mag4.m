% Optimize CPMG excitation pulse, precalculate refocusing matrix
% Soumyajit Mandal, 09/21/10
% --------------------------------------------------------------
% Allow arbitrary pulse amplitudes, 02/25/11
% Use VAN_EXC as a starting point, 08/22/11
% Gradually make excitation pulse shorter, 09/16/11

function [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag4(neff,del_w,len_acq,nseg,start)

T_90=pi/2; % normalized

delt=0.01*T_90;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));

nstart=length(start);
del=nstart-nseg; % Make starting phase vector shorter
start=start(del/2+1:del/2+nseg); 

texc=0.2*T_90*ones(1,nseg);
aexc=ones(1,nseg); % Segments have arbitrary phase and constant amplitude

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=zeros(1,nseg); % Lower bound
ub=2*pi*ones(1,nseg); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',10000);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',5000);

pexc=fmincon(@(params)fit_function(params,texc,aexc,neff,del_w,len_acq,tvect),start,[],[],[],[],lb,ub,[],options);
[masy]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);

for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end

echo_pk=max(abs(echo));
echo_rms=sqrt(trapz(tvect,abs(echo).^2));

function val=fit_function(pexc,texc,aexc,neff,del_w,len_acq,tvect)

[masy]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);

% Calculate time-domain echo
echo=zeros(1,length(tvect));
for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end

% Phase inversion leaves behind only the symmetric part of the spectrum,
% i.e., the real component of the time-domain echo
echo = real(echo);

% Optimize echo peak
%val=-0.01*max(abs(echo));

% Optimize echo RMS + echo peak
val=-0.01*(sqrt(trapz(tvect,abs(echo).^2))+max(abs(echo)));

% Optimize echo RMS + echo peak
%masy=real(masy);
%val=-sqrt(trapz(del_w,masy.^2))-0.33*trapz(del_w,masy);