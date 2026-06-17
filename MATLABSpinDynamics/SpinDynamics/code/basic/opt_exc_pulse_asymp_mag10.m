% Optimize CPMG excitation pulse, precalculate refocusing matrix
% Soumyajit Mandal, 09/21/10
% --------------------------------------------------------------
% Allow arbitrary pulse amplitudes, 02/25/11
% Use VAN_EXC as a starting point, 08/22/11
% Gradually make excitation pulse shorter -> fewer segments (09/12/11),
% shorter segments (09/13/11)
% Gradually make excitation pulse longer -> more segments (09/16/11)
% Use Colm's optimized code for speed -> 09/19/11.
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
% Re-optimize results_mag9 -> (09/21/11)
% Add relaxation during the pulse

function [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag10(pulse_num,NE,len_acq,T1,T2)

T_90=pi/2; % normalized

delt=0.01*T_90;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));

% RP2-1.0 refocusing pulse

% Default pulse
%tref=pi*[3 0.14 0.72 0.14 3];
%pref=pi*[0 1 0 1 0];
%aref=[0 1 1 1 0];

% Add extra segments for more accurate in-pulse relaxation calculation
tref=pi*[3 0.14 0.24 0.24 0.24 0.14 3];
pref=pi*[0 1 0 0 0 1 0];
aref=[0 1 1 1 1 1 0];

% New constant-amplitude OCT excitation pulses
tmp=load(fullfile('dat_files','results_mag_all.mat'));
results=tmp.results_sort;

% Reoptimize pulses optimized for relaxation
%tmp=load(fullfile('dat_files','results_mag_relax.mat'));
%results=tmp.results;

texc=results{pulse_num,1};
pexc=results{pulse_num,2};
aexc=ones(1,length(texc)); % Segments have arbitrary phase and constant amplitude

nseg=length(texc);
start=pexc;

% Reference asymptotic magnetization
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);
[masy_ref]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
masy_ref=real(masy_ref);

% Excitation pulse definition
% Use nonlinear function minimization - all segment phases lie between 0
% and 2*pi
lb=zeros(1,nseg); % Lower bound
ub=2*pi*ones(1,nseg); % Upper bound

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',30000);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',30000);

pexc=fmincon(@(params)fit_function...
    (params,texc,aexc,tref,pref,aref,NE,len_acq,T1,T2,masy_ref),start,[],[],[],[],lb,ub,[],options);

% No frequency offsets in pulses
oexc=zeros(1,length(pexc));
oref=zeros(1,length(pref));

[macq,del_w]=...
    cpmg_van_spin_dynamics_arb(texc,pexc,aexc,oexc,tref,pref,aref,oref,NE,len_acq,T1,T2);
masy=macq(NE,:);

for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end

echo_pk=max(abs(echo));
echo_rms=sqrt(trapz(tvect,abs(echo).^2));

function val=fit_function(pexc,texc,aexc,tref,pref,aref,NE,len_acq,T1,T2,masy_ref)

% No frequency offsets in pulses
oexc=zeros(1,length(pexc));
oref=zeros(1,length(pref));

[macq,del_w]=...
    cpmg_van_spin_dynamics_arb(texc,pexc,aexc,oexc,tref,pref,aref,oref,NE,len_acq,T1,T2);
masy=real(macq(NE,:));

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

%val=-sqrt(trapz(del_w,masy.^2))-0.33*trapz(del_w,masy);

% Optimize echo RMS
val=-sqrt(trapz(del_w,masy.*masy_ref));
