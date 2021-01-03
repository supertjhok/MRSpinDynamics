% Theoretical evaluation of Bloch-Siegert frequency shift (spin-echo)
% Include relaxation during free precession intervals
% bs_param = properties of B-S pulses [amplitude length frequency_offset]
% Soumyajit Mandal, 08/30/11

function [ph,amp]=bloch_siegert_se(expt,T1,T2,bs_param)

close all;

T_180 = pi;
T_FP = 9*T_180;
len_acq = 5*T_180;

necho = 1e3;
tvect=linspace(-len_acq/2,len_acq/2,necho);

NE=1; % Single spin-echo

switch expt
    
    case 1
        display('Rectangular excitation (0 dB), rectangular refocusing (0 dB, 180 degrees)');
        
        texc=T_180/2; pexc=pi/2; aexc=1;
        tref=T_180; pref=0; aref=1;
        
    case 2
        display('Rectangular excitation (0 dB), rectangular refocusing (0 dB, 135 degrees)');
        
        texc=T_180/2; pexc=pi/2; aexc=1;
        tref=3*T_180/4; pref=0; aref=1;
        
end

% BS-pulses
[tp,phi,amp,offs,acq]=create_bs_se_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq1,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

[tp,phi,amp,offs,acq]=create_bs_se_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq2,del_w]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

% No BS-pulses
bs_param(1)=0; % zero-amplitude
[tp,phi,amp,offs,acq]=create_bs_se_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq3,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

[tp,phi,amp,offs,acq]=create_bs_se_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq4,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

[~,echo_bs,~]=plot_function(macq1,macq2,del_w,tvect);
[~,echo,~]=plot_function(macq3,macq4,del_w,tvect);

[ph,amp]=estimate_phase(echo_bs,echo,tvect);

function [tp,phi,amp,offs,acq]=create_bs_se_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param)

a_bs=bs_param(1);
T_bs=bs_param(2);
df_bs=bs_param(3);

nexc=length(texc)+1;
nref=length(tref)+4;

ntot=nexc+NE*nref;
tp=zeros(1,ntot); phi=tp; amp=tp; offs=tp; acq=tp;

% Excitation
tp(1:nexc-1)=texc;
phi(1:nexc-1)=pexc;
amp(1:nexc-1)=aexc;
if nexc==2
    tp(nexc)=-1/aexc(1); % Timing correction for rectangular pulses
end

% Refocusing
for i=1:NE
    tp(nexc+(i-1)*nref+1)=T_FP/2-T_bs;
    tp(nexc+(i-1)*nref+2)=T_bs;
    tp(nexc+(i-1)*nref+3:nexc+i*nref-2)=tref;
    tp(nexc+i*nref-1)=T_bs;
    tp(nexc+i*nref)=T_FP/2-T_bs;
    
    if i==1
        amp(nexc+(i-1)*nref+2)=a_bs;
    end
    amp(nexc+(i-1)*nref+3:nexc+i*nref-2)=aref;
    amp(nexc+i*nref-1)=a_bs;
    
    phi(nexc+(i-1)*nref+3:nexc+i*nref-2)=pref;
    
    offs(nexc+(i-1)*nref+2)=df_bs; % Frequency offset of B-S pulses
    offs(nexc+i*nref-1)=-df_bs;
end

acq(ntot)=1; % Acquire at the end of the CPMG train


function [macq,echo,eint]=plot_function(macq1,macq2,del_w,tvect)

necho=length(tvect);
len_acq=max(tvect)-min(tvect);
echo=zeros(1,necho);

macq=(macq1-macq2)/2; % Phase cycle

% Calculate time-domain waveform
for j=1:necho
    echo(j)=sum(macq.*exp(-1i*del_w*tvect(j))); % -1 coherence at time of detection
end

% LPF to remove echo components far from resonance
%del_w_max=2;
%del_f_max=del_w_max/(2*pi);
%fs=1/(tvect(2)-tvect(1));
%[b,a]=butter(4,del_f_max/(fs/2));

%echo=filter(b,a,echo);

figure(1);
plot(del_w,real(macq),'b-'); hold on;
plot(del_w,imag(macq),'r-');
plot(del_w,abs(macq),'k-');
xlabel('\Delta\omega_{0} / \omega_{1}');

figure(2);
plot(tvect/len_acq,real(echo),'b-'); hold on;
plot(tvect/len_acq,imag(echo),'r-');
plot(tvect/len_acq,abs(echo),'k-');
xlabel('Time');

% Calculate echo integral
eint=sqrt(trapz(del_w,abs(macq.^2)));

function [ph,amp]=estimate_phase(echo_bs,echo,tvect)

% Calculate relative phase
% Use the entire echo shape after matched filtering to calculate phase
%mf=abs(echo);
%ph=atan(sum(imag(echo_bs).*mf)/sum(real(echo_bs).*mf));
% Use the echo peak / on-resonance component to calculate phase - gives the
% most accurate results
ph=atan(max(imag(echo_bs))/max(real(echo_bs)));

% Calculate relative amplitude
amp=sqrt(trapz(tvect,abs(echo_bs).^2))/sqrt(trapz(tvect,abs(echo).^2));
