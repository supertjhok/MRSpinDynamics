% Optimize CPMG excitation pulse, precalculate refocusing matrix
% Soumyajit Mandal, 09/21/10
% --------------------------------------------------------------
% Allow arbitrary pulse amplitudes, 02/25/11
% Use VAN_EXC as a starting point, 08/22/11
% Gradually make excitation pulse shorter -> fewer segments (09/12/11),
% shorter segments (09/13/11)
% Gradually make excitation pulse longer -> more segments (09/16/11)
% Use Colm's optimized code for speed -> 09/19/11.
% Gradually make excitation pulse longer -> stretch segments (03/29/12)
% --------------------------------------------------------------
% Using Colm's code only ~17% of the time is being spent in calculating the
% fitness function, i.e., the spin dynamics (for 100-segment long
% excitation pulses and 10 echoes). The rest is fmincon overhead, so
% further optimization of the spin dynamics code won't speed up the
% optimization process. Net speedup relative to my Matlab code is ~2.5x,
% which is far less than the 8x - 10x speedup of the spin dynamics itself.
% In order to get more speedup the optimization routine has to be written
% in C++ as well!
% --------------------------------------------------------------

function [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag8(texc,pexc,tref,pref,aref,delta,len_acq)

T_90=pi/2; % normalized
numEchoes=10;

delt=0.01*T_90;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));

nseg=length(texc)+delta;
start=zeros(1,nseg); texc_new=start;

start(1+delta/2:nseg-delta/2)=pexc;
start(1:delta/2)=pexc(1); % Add segments
start(nseg-delta/2+1:nseg)=pexc(nseg-delta);

texc_new(1+delta/2:nseg-delta/2)=texc;
texc_new(1:delta/2)=texc(1); % Add segments
texc_new(nseg-delta/2+1:nseg)=texc(nseg-delta);
texc=texc_new;

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

pexc=fmincon(@(params)fit_function(params,texc,aexc,tref,pref,aref,T_90,numEchoes,len_acq),start,[],[],[],[],lb,ub,[],options);
[masy,del_w]=runCPMG(texc,pexc,aexc,tref,pref,aref,T_90,numEchoes,len_acq);

for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end

echo_pk=max(abs(echo));
echo_rms=sqrt(trapz(tvect,abs(echo).^2));

function val=fit_function(pexc,texc,aexc,tref,pref,aref,T_90,numEchoes,len_acq)

[masy,del_w]=runCPMG(texc,pexc,aexc,tref,pref,aref,T_90,numEchoes,len_acq);

% Calculate time-domain echo
%echo=zeros(1,length(tvect));
%for i=1:length(tvect)
%    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
%end

% Phase inversion leaves behind only the symmetric part of the spectrum,
% i.e., the real component of the time-domain echo
%echo = real(echo);

% Optimize echo peak
%val=-0.01*max(abs(echo));

% Optimize echo RMS + echo peak
%val=-0.01*(sqrt(trapz(tvect,abs(echo).^2))+max(abs(echo)));

masy=real(masy);
val=-sqrt(trapz(del_w,masy.^2))-0.33*trapz(del_w,masy);