% quant -> phase quantization yes / no

function opt_exc_pulse_mag_results_w1(expt,quant)

T_90=pi/2;
len_acq=5*pi;

num_pexc=128; % Number of phase quantization steps

nvect=0.5:0.1:1.5; % Vector for nutation curve
npts=length(nvect);

switch expt
    
    case 1
        % Default / rectangular case
        tref=pi*[3 1 3]; % 180 degrees
        pref=pi*[0 0 0];
        aref=[0 1 0];
        
        texc=pi/2;
        pexc=pi/2;
        aexc=1;
        
    case 2    
        % Default / rectangular case
        tref=pi*[3.125 0.75 3.125]; % 135 degrees
        pref=pi*[0 0 0];
        aref=[0 1 0];
        
        texc=pi/2;
        pexc=pi/2;
        aexc=1;
        
    case 3
        % Van's excitation pulse
        tref=pi*[3 0.14 0.72 0.14 3];
        pref=pi*[0 1 0 1 0];
        aref=[0 1 1 1 0];
        
        tmp=load('van_exc.mat');
        van_exc=tmp.van_exc;
        
        texc=0.2*T_90*ones(1,1e2);
        pexc=van_exc(:,2)'*(pi/180);
        aexc=van_exc(:,1)';
        %aexc=ones(1,1e2); % No amplitude modulation
        
    case 4
        % Constant-amplitude OCT excitation pulse
        tref=pi*[3 0.14 0.72 0.14 3];
        pref=pi*[0 1 0 1 0];
        aref=[0 1 1 1 0];
        
        tmp=load('results_mag3_rms_2.mat');
        texc=tmp.texc;
        pexc=tmp.pexc;
        aexc=ones(1,length(texc));
end

if quant
    del=2*pi/num_pexc;
    pexc=del*round(pexc/del); % Phase quantization
end

if length(texc)>1
    plot_instfreq(texc,pexc,T_90);
end

echo_rms=zeros(1,npts);

for i=1:npts
    [neff,del_w]=calc_rot_axis_arba(tref,pref,aref*nvect(i));
    
    [masy1]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc*nvect(i),neff,del_w,len_acq);
    if length(texc)==1e2  % OCT excitation pulse, use phase inversion instead of cycling
        [masy2]=cpmg_van_spin_dynamics_asymp_mag2(texc,-pexc,aexc*nvect(i),neff,del_w,len_acq);
    else
        [masy2]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc+pi,aexc*nvect(i),neff,del_w,len_acq);
    end
    masy=(masy1-masy2)/2; % Phase cycling / inversion
    
    echo_rms(i)=plot_results(masy,del_w,len_acq);
    disp(i)
end

[~,ind]=min(abs(nvect-1)); % Normalize to default case

figure(5);
plot(nvect,echo_rms/echo_rms(ind),'bo-'); hold on;
xlabel('\omega_{1} / \omega_{1,nom}');
ylabel('Normalized echo amplitude');

function [echo_rms]=plot_results(masy,del_w,len_acq)

delt=0.01;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));

for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end

echo_pk=max(abs(echo))
echo_rms=sqrt(trapz(tvect,abs(echo).^2))

figure(3);
plot(del_w,real(masy),'b-'); hold on;
plot(del_w,imag(masy),'r-');
xlabel('\Delta\omega_{0} / \omega_{1}');

figure(4);
plot(tvect,real(echo),'b-'); hold on;
plot(tvect,imag(echo),'r-');
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