function opt_exc_pulse_mag_results_sweep

T_90=pi/2;
len_acq=4*pi;

% Default / rectangular case
tref=pi*[3 1 3]; % 180 degrees
%tref=pi*[3.125 0.75 3.125]; % 135 degrees
pref=pi*[0 0 0];
aref=[0 1 0];

texc=pi/2;
pexc=pi/2;
aexc=1;

[echo_pk_ref,echo_rms_ref]=run_sequence(tref,pref,aref,texc,pexc,aexc,T_90,len_acq);

% RP2 refocusing cycle
tref=pi*[3 0.14 0.72 0.14 3];
pref=pi*[0 1 0 1 0];
aref=[0 1 1 1 0];

% 4: Shorter OCT excitation pulses (fewer segments)
% 5: Shorter OCT excitation pulses (shorter segments, linear decrease)
% 6: Shorter OCT excitation pulses (shorter segments, geometric decrease)
% 7: Longer OCT excitation pulses (more segments)
% 8: Longer OCT excitation pulses (more segments) -> after re-optimization
% 9: Longer OCT excitation pulses (more segments) -> using Colm's code
% 10: Longer OCT excitation pulses (more segments) -> using Colm's code after re-optimization

clr={'bo-','r*-','kd-','ms-','g^-','bv-','r.-'};
for i=1:7
    plot_oct(tref,pref,aref,T_90,len_acq,echo_pk_ref,echo_rms_ref,i+3,clr{i});
end


function plot_oct(tref,pref,aref,T_90,len_acq,echo_pk_ref,echo_rms_ref,expt,clr)

% Original constant-amplitude OCT excitation pulse
tmp=load('dat_files\results_mag3_rms_2.mat');
texc=tmp.texc;
pexc=tmp.pexc;
aexc=ones(1,length(texc));

% New OCT excitation pulses
filname=['dat_files\results_mag' num2str(expt) '.mat'];
tmp=load(filname);
results=tmp.results;
sizres=size(results);
numexp=sizres(1);

if expt==4 % Plot original OCT pulse as well
    echo_pk=zeros(1,numexp+1); echo_rms=echo_pk; tlen=echo_pk;
    for i=1:numexp+1
        [echo_pk(i),echo_rms(1,i)]=run_sequence(tref,pref,aref,texc,pexc,aexc,T_90,len_acq);
        tlen(i)=sum(texc);
        
        if i<numexp+1
            texc=results{i,1};
            pexc=results{i,2};
            aexc=ones(1,length(texc));
        end
        disp(i)
    end
else % Only plot new OCT pulses
    echo_pk=zeros(1,numexp); echo_rms=echo_pk; tlen=echo_pk;
    for i=1:numexp
        texc=results{i,1};
        pexc=results{i,2};
        aexc=ones(1,length(texc));
        [echo_pk(i),echo_rms(1,i)]=run_sequence(tref,pref,aref,texc,pexc,aexc,T_90,len_acq);
        tlen(i)=sum(texc);
        
        disp(i)
    end
end

figure(6);
plot(tlen/(2*T_90),(echo_rms/echo_rms_ref).^2,clr); hold on;
xlabel('Pulse length (units of T_{180})');
ylabel('Normalized SNR (power units)');

figure(7);
plot(tlen/(2*T_90),echo_pk/echo_pk_ref,clr); hold on;
xlabel('Pulse length (units of T_{180})');
ylabel('Normalized echo peak');



function [echo_pk,echo_rms]=run_sequence(tref,pref,aref,texc,pexc,aexc,T_90,len_acq)

[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);

% Phase cycling
[masy1]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);
if length(texc)>1 % OCT pulse, use phase inversion
    plot_instfreq(texc,pexc,T_90);
    [masy2]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc,neff,del_w,len_acq);
else
    [masy2]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc+pi,aexc,neff,del_w,len_acq);
end
masy=(masy1-masy2)/2; % Phase cycling / inversion

[echo_pk,echo_rms]=plot_results(masy,del_w,T_90,len_acq);

function [echo_pk,echo_rms]=plot_results(masy,del_w,T_90,len_acq)

delt=0.01*T_90;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));

for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end

echo_pk=max(abs(echo));
echo_rms=sqrt(trapz(tvect,abs(echo).^2));

figure(3);
plot(del_w,real(masy),'b-'); hold on;
plot(del_w,imag(masy),'r-');
xlabel('\Delta\omega_{0} / \omega_{1}');

figure(4);
plot(tvect/T_90,real(echo),'b-'); hold on;
plot(tvect/T_90,imag(echo),'r-');
xlabel('Time');

function plot_instfreq(tvect,pvect,T_90)

nseg=length(tvect);

w1=2*pi/(4*T_90);
del_p=diff(pvect);

% Try to create minimum phase
pvectm=pvect;
nvect=linspace(-3,3,7); nl=length(nvect);
for i=2:nseg
    phl=zeros(1,nl);
    for j=1:nl
        phl(j)=abs(del_p(i-1)+nvect(j)*2*pi);
    end
    [~,ind]=min(phl);
    pvectm(i)=pvectm(i)+nvect(ind)*2*pi;
    del_p=diff(pvectm);
end

del_t=(tvect(1:nseg-1)+tvect(2:nseg))/2;
wvect=(del_p./del_t)/w1;

figure(1);
subplot(2,1,1);
stairs(pvect*180/pi,'b-'); hold on;
stairs(pvectm*180/pi,'b--');
ylabel('Phase (degrees)');

subplot(2,1,2);
stairs(wvect,'r-'); hold on;
ylabel('\Delta\omega / \omega_{1}');
xlabel('Segment number');