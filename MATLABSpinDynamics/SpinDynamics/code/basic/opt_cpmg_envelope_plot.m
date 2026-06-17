function [eint,eint2]=opt_cpmg_envelope_plot(ne,T_E,avals)

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

[eint,eint2]=plot_function(avals,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,masy,normasy);


function [eint,eint2]=plot_function(params,texc,pexc,aexc,tref,pref,aref,ne,del_w,t_acq,masy,normasy)

% Set pulse amplitudes and durations
aref(1,2:3:3*ne-1)=params;
% Adjust delays to keep T_E constant
%tref(1,1:3:3*ne-2)=tref(1,1:3:3*ne-2)+tref(1,2:3:3*ne-1).*(1-1./aref(1,2:3:3*ne-1));
%tref(1,3:3:3*ne)=tref(1,1:3:3*ne-2);
% Adjust pulse lengths
tref(1,2:3:3*ne-1)=tref(1,2:3:3*ne-1)./aref(1,2:3:3*ne-1);

% Calculate echoes with timing correction and phase cycling
[mecho1]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0],[aexc 0],tref,pref,aref,ne,del_w,t_acq);
[mecho2]=sim_spin_dynamics_arba_echoes([texc -1/aexc(1)],[pexc 0]+pi,[aexc 0],tref,pref,aref,ne,del_w,t_acq);
mecho=(mecho1-mecho2)/2;

% Calculate time domain echoe waveforms
dt=0.05; % sampling period for observing echo
necho=round(t_acq/dt); % number of time points for observing echo

delt=dt*necho; % Final time period for observing echo
tvect=linspace(-delt/2,delt/2,necho);

% Calculate echo integrals and time domain waveforms
eint=zeros(1,ne); % Frequency domain
eint2=eint; % Time domain
echo=zeros(1,necho); echo_asy=echo;

% Asymptotic echo
for j=1:necho
    % Filter with default asymptotic echo
    % echo_asy(j)=sum(masy.*exp(-1i*del_w*tvect(j))); 
    % Filter with actual asymptotic echo
    echo_asy(j)=sum(mecho(ne,:).*exp(-1i*del_w*tvect(j)));
end
normasy2=sqrt(trapz(tvect,abs(echo_asy.*echo_asy)));

for n=1:ne
    % Filter with default asymptotic echo
    %eint(n)=trapz(del_w,abs(mecho(n,:).*masy))/normasy; % Frequency domain
    % Filter with actual asymptotic echo
    eint(n)=trapz(del_w,abs(mecho(n,:).*mecho(ne,:)))/normasy; % Frequency domain
    
    figure(2);
    plot3(n*ones(1,length(del_w)),del_w,abs(masy),'r--'); hold on;
    plot3(n*ones(1,length(del_w)),del_w,abs(mecho(n,:)));
    
    for j=1:necho
        echo(j)=sum(mecho(n,:).*exp(-1i*del_w*tvect(j)));
    end
    eint2(n)=trapz(tvect,abs(echo_asy.*echo))/normasy2; % Time domain
    
    figure(3);
    plot(tvect+n*delt,abs(echo_asy),'r--'); hold on;
    plot(tvect+n*delt,abs(echo));
end

figure(2);
xlabel('Echo number')
ylabel('\omega_{0}')
zlabel('|Spectra|');

figure(3);
xlabel('Time')
ylabel('|Echoes|');

% Final / asymptotic echo
figure(4);
plot(tvect,abs(echo_asy),'r--'); hold on;
plot(tvect,abs(echo));

figure(4);
xlabel('Time')
ylabel('|Asymptotic echo|');

% Normalize echo integrals to 1
%eint=eint/normasy;
%eint2=eint2/normasy2;

figure(5);
plot(eint,'bo--'); hold on;
plot(eint2,'r*--');
