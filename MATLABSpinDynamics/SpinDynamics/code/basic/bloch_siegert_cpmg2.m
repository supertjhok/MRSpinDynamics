% ------------------------------------------------------------------------
% Theoretical evaluation of Bloch-Siegert frequency shift (CPMG)
% Include relaxation during free precession intervals
% plen = normalized excitation / refocusing pulse length
% bs_param = properties of B-S pulses [amplitude length frequency_offset]
% Alternate B-S pulse timing (applied after the echo)
% ------------------------------------------------------------------------
% Note: set length x frequency_offset = n x pi, where n = an integer, to 
% ensure phase coherence (as in the physical experiment)
% ------------------------------------------------------------------------
% Soumyajit Mandal, 08/30/11

function [ph,amp]=bloch_siegert_cpmg2(expt,NE,T1,T2,plen,bs_param)

close all;

T_180 = pi;
T_FP = 9*T_180;
len_acq = 2*T_180;

necho = 1e3;
tvect=linspace(-len_acq/2,len_acq/2,necho);

switch expt
    
    case 1
        display('Rectangular excitation (0 dB), rectangular refocusing (0 dB, 180 degrees)');
        
        texc=plen*T_180/2; pexc=pi/2; aexc=1;
        tref=plen*T_180; pref=0; aref=1;
        
    case 2
        display('Rectangular excitation (0 dB), rectangular refocusing (0 dB, 135 degrees)');
        
        texc=plen*T_180/2; pexc=pi/2; aexc=1;
        tref=plen*3*T_180/4; pref=0; aref=1;
        
end

% The BS phase cycle is designed to cancel any excitation produced by the
% BS pulse. Without it, the phase shift depends quite a bit on plen
% ------------------------------------------------------------------------
% BS-pulses, +ve offset
bs_param(4)=0;
[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq1a,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq1b,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

bs_param(4)=pi;
[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq1c,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq1d,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

macq1=(macq1a-macq1b+macq1c-macq1d)/4; % Phase cycle

% BS-pulses, -ve offset
bs_param(3)=-bs_param(3);
bs_param(4)=0;
[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq2a,del_w]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq2b,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

bs_param(4)=pi;
[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq2c,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq2d,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

macq2=(macq2a-macq2b+macq2c-macq2d)/4; % Phase cycle

% The regular CPMG phase cycle is a simple PAP
% ------------------------------------------------------------------------
% No BS-pulses
bs_param(1)=0; % zero-amplitude
[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq3a,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);

[tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP,bs_param);
[macq3b,~]=sim_spin_dynamics_arba_relax_offs(tp,phi,amp,offs,acq,len_acq,T1,T2);
macq3=(macq3a-macq3b)/2; % Phase cycle

[~,echo_bs1,~]=plot_function(macq1,del_w,tvect);
[~,echo_bs2,~]=plot_function(macq2,del_w,tvect);
[~,echo,~]=plot_function(macq3,del_w,tvect);

[ph,amp]=estimate_phase(echo_bs1,echo_bs2,echo,tvect);

function [tp,phi,amp,offs,acq]=create_bs_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP,bs_param)

a_bs=bs_param(1);
T_bs=bs_param(2);
df_bs=bs_param(3);
ph_bs=bs_param(4);

nexc=length(texc)+1;
nref=length(tref)+3;

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
    
    if i==1
        tp(nexc+(i-1)*nref+1)=T_FP/2-T_bs;
        tp(nexc+(i-1)*nref+2)=T_bs;
    else
        tp(nexc+(i-1)*nref+1)=T_FP/2-2*T_bs;
        tp(nexc+(i-1)*nref+2)=2*T_bs;      
    end
    tp(nexc+(i-1)*nref+3:nexc+i*nref-1)=tref;
    tp(nexc+i*nref)=T_FP/2;
    
    amp(nexc+(i-1)*nref+2)=a_bs;
    amp(nexc+(i-1)*nref+3:nexc+i*nref-1)=aref;
    
    phi(nexc+(i-1)*nref+2)=ph_bs;
    phi(nexc+(i-1)*nref+3:nexc+i*nref-1)=pref;
    
    offs(nexc+(i-1)*nref+2)=df_bs; % Frequency offset of B-S pulses
end

acq(ntot)=1; % Acquire at the end of the CPMG train



function [macq,echo,eint]=plot_function(macq,del_w,tvect)

necho=length(tvect);
len_acq=max(tvect)-min(tvect);
echo=zeros(1,necho);

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

function [ph,amp]=estimate_phase(echo_bs1,echo_bs2,echo,tvect)

len_acq=max(tvect)-min(tvect);

% Calculate relative phase of positive and negative frequency offsets
% separately
% ------------------------------------------------------------------------
% Use the entire echo shape
ph1=atan(sum(imag(echo_bs1))/sum(real(echo_bs1)));
ph2=atan(sum(imag(echo_bs2))/sum(real(echo_bs2)));

% Use the entire echo shape after matched filtering to calculate phase
%mf=abs(echo);
%ph1=atan(sum(imag(echo_bs1).*mf)/sum(real(echo_bs1).*mf));
%ph2=atan(sum(imag(echo_bs2).*mf)/sum(real(echo_bs2).*mf));

% Use the echo peak / on-resonance component to calculate phase
%ph1=atan(max(abs(imag(echo_bs1)))/max(abs(real(echo_bs1))));
%ph2=atan(max(abs(imag(echo_bs2)))/max(abs(real(echo_bs2))));

% Use the t = 0 values to calculate phase
%[~,ind]=min(abs(tvect));
%ph1=atan(abs(imag(echo_bs1(ind)))/abs(real(echo_bs1(ind))));
%ph1=atan(abs(imag(echo_bs2(ind)))/abs(real(echo_bs2(ind))));

% Use the difference between positive and negative frequency offsets to
% calculate the BS phase. This cancels the resonant frequency offset
% dependence of the phase shift (to first order).
% ------------------------------------------------------------------------
ph=ph1-ph2;

% Symmetrize echo shape for display purposes
echo_bs=(echo_bs1+fliplr(conj(echo_bs2)))/2;
%ph=atan(sum(imag(echo_bs))/sum(real(echo_bs)));

% Calculate relative amplitude%
% ------------------------------------------------------------------------
amp=sqrt(trapz(tvect,abs(echo_bs).^2))/sqrt(trapz(tvect,abs(echo).^2));

figure(3);
plot(tvect/len_acq,real(echo_bs),'b-'); hold on;
plot(tvect/len_acq,imag(echo_bs),'r-');
xlabel('Time');
