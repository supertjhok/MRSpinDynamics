function sim_echo_transient_pulsevar(plt)

close all;
NE=100; echonum=linspace(1,NE,NE);

texc=[pi/2 -1]; pexc=[pi/2 0]; aexc=[1 0]; % Rectangular excitation pulse with Martin's timing correction
%texc=[pi/2 -1]/10; pexc=[pi/2 0]; aexc=[1 0]*10; % High-power rectangular excitation pulse

tref=pi*[3 1 3]; pref=pi*[0 0 0]; aref=[0 1 0]; % Rectangular refocusing pulse
%tref=pi*[3 0.14 0.72 0.14 3]; pref=pi*[0 1 0 1 0]; aref=[0 1 1 1 0]; % RP2 refocusing pulse

% Pulse amplitude and phase variations
del_amax=0.0; del_pmax=0*2*pi; del_phi=2*pi/2; phi0=pi/2;

avec=sin(del_phi*echonum+phi0); % sine wave
%avec=sin(del_phi*echonum+phi0); avec(abs(avec)<0.7)=0; % thresholded sine wave
%avec=(sin(del_phi*echonum+phi0)>0.5)-0.5; % square wave
del_aref=del_amax*avec;

pvec=sin(del_phi*echonum+phi0); % sine wave
%pvec=sin(del_phi*echonum+phi0); pvec(abs(pvec)<0.7)=0; % thresholded sine wave
%pvec=(sin(del_phi*echonum+phi0)>0)-0.5; % square wave
del_pref=del_pmax*pvec;

% Acquisition window length, tE
len_acq=4*pi; tE=sum(tref);

% Phase alternating pair
[macq1,del_w]=cpmg_van_spin_dynamics_arb_pulsevar(texc,pexc,aexc,tref,pref,aref,NE,len_acq,del_pref,del_aref);
[macq2,~]=cpmg_van_spin_dynamics_arb_pulsevar(texc,pexc+pi,aexc,tref,pref,aref,NE,len_acq,del_pref,del_aref);
macq=(macq1-macq2)/2;

% Use asymptotic echo with no perturbations as a reference
[neff]=calc_rot_axis_arba3(tref,pref,aref,del_w,0);
[masy]=cpmg_van_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,del_w,len_acq);
[echo_ref,tvect]=calc_time_domain_echo(masy,del_w);

ind=find(abs(tvect)<len_acq/2);
echo_ref=echo_ref(ind); tvect=tvect(ind);
norm=sqrt(trapz(tvect,abs(echo_ref).^2));
if plt
    figure(1); plot(tvect,real(echo_ref),'b-'); hold on; plot(tvect,imag(echo_ref),'r-');
end

echoes=zeros(length(tvect),NE); echo_int=zeros(1,NE);
for i=1:NE
    [tmp,~]=calc_time_domain_echo(macq(i,:),del_w);
    echoes(:,i)=tmp(ind);
    
    echo_int(i)=trapz(tvect,echoes(:,i).*conj(echo_ref))/norm; % Use asymptotic echo as a matched filter
    
    if plt
        figure(2); plot(tvect/tE+i,real(echoes(:,i)),'b-'); hold on;
        plot(tvect/tE+i,imag(echoes(:,i)),'r-');
        
        figure(3); plot(del_w/(2*max(del_w))+i,real(macq(i,:)),'b-'); hold on;
        plot(del_w/(2*max(del_w))+i,imag(macq(i,:)),'r-');
    end
end
echo_int=echo_int/norm; % Normalize initial amplitude to 1

figure(4);
subplot(2,1,1);
plot(echonum,del_aref,'b-'); hold on; plot(echonum,del_pref/(2*pi),'r-');
legend({'\deltaa_{k}','\delta\phi_{k}'});

subplot(2,1,2);
plot(echonum,real(echo_int),'b-'); hold on; plot(echonum,imag(echo_int),'r-');
plot(echonum,abs(echo_int),'k-');