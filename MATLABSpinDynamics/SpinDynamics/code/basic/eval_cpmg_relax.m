% Theoretical evaluation of various excitation and refocusing pulses
% Include relaxation during free precession intervals
% NE = number of echoes
% find_nvals = calculate <n_x^2>, <n_y^2>, <n_z^2> if 1, don't if 0
% Soumyajit Mandal, 03/16/11

function [T_relax,eint,macq,nvals,del_w]=eval_cpmg_relax(expt,NE,T1,T2,find_nvals)

T_180 = pi;
T_FP = 6*T_180;
len_acq=T_FP;

switch expt
    
    case 1
        display('Rectangular excitation (0 dB), rectangular refocusing (0 dB, 180 degrees)');
        
        texc=T_180/2; pexc=0; aexc=1;
        tref=T_180; pref=pi/2; aref=1;
        T_E=T_FP+sum(tref);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP);
        [macq1,~]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP);
        [macq2,del_w]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        if find_nvals
            nvals=calc_nvals(tref,pref,aref,T_FP,NE,macq1,macq2,del_w);
        end
        
    case 2
        display('VAN_EXC (0 dB), RP2-1.0 refocusing (0 dB)');
        
        tmp=load('van_exc.mat'); van_exc=tmp.van_exc;
        texc=0.1*T_180*ones(1,1e2); pexc=van_exc(:,2)'*(pi/180); aexc=van_exc(:,1)';
        tref=T_180*[0.14 0.72 0.14]; pref=(pi/2)*[2 0 2]; aref=[1 1 1];
        T_E=T_FP+sum(tref);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP);
        [macq1,~]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,-pexc,aexc,tref,pref,aref,NE,T_FP);
        [macq2,del_w]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        if find_nvals
            nvals=calc_nvals(tref,pref,aref,T_FP,NE,macq1,macq2,del_w);
        end
        
    case 3
        display('CP-M15 (0 dB), RP2-1.0 refocusing (0 dB)');
        
        tmp=load('exc_timings.mat'); results=tmp.results;
        texc=(T_180/2)*results{16,2}; pexc=results{16,3}; aexc=ones(1,20);
        tref=T_180*[0.14 0.72 0.14]; pref=(pi/2)*[3 1 3]; aref=[1 1 1];
        T_E=T_FP+sum(tref);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP);
        [macq1,~]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP);
        [macq2,del_w]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        if find_nvals
            nvals=calc_nvals(tref,pref,aref,T_FP,NE,macq1,macq2,del_w);
        end
        
    case 4
        display('Rectangular (20 dB), RP2-1.0 refocusing (0 dB)');
        
        texc=T_180/20; pexc=0; aexc=10;
        tref=T_180*[0.14 0.72 0.14]; pref=(pi/2)*[3 1 3]; aref=[1 1 1];
        T_E=T_FP+sum(tref);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP);
        [macq1,~]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP);
        [macq2,del_w]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        if find_nvals
            nvals=calc_nvals(tref,pref,aref,T_FP,NE,macq1,macq2,del_w);
        end
        
    case 5
        display('Rectangular excitation (0 dB), rectangular refocusing (0 dB, 135 degrees)');
        
        texc=T_180/2; pexc=0; aexc=1;
        tref=3*T_180/4; pref=pi/2; aref=1;
        T_E=T_FP+sum(tref);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP);
        [macq1,~]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc+pi,aexc,tref,pref,aref,NE,T_FP);
        [macq2,del_w]=sim_spin_dynamics_arba_relax(tp,phi,amp,acq,len_acq,T1,T2);
        
        if find_nvals
            nvals=calc_nvals(tref,pref,aref,T_FP,NE,macq1,macq2,del_w);
        end
        
end

[T_relax,eint,macq]=plot_function(macq1,macq2,del_w,NE,T_E);
T_relax=T_relax*T_FP/T_E; % Correct for duty cycle
disp(['Decay time constant = ' num2str(T_relax/T2) ' x T2'])

function [tp,phi,amp,acq]=create_cpmg_timing(texc,pexc,aexc,tref,pref,aref,NE,T_FP)

nexc=length(texc)+1;
nref=length(tref)+2;
tp=zeros(1,nexc+NE*nref); phi=tp; amp=tp; acq=tp;

% Excitation
tp(1:nexc-1)=texc;
phi(1:nexc-1)=pexc;
amp(1:nexc-1)=aexc;
if nexc==2
    tp(nexc)=-1/aexc(1); % Timing correction for rectangular pulses
end

% Refocusing
for i=1:NE
    tp(nexc+(i-1)*nref+1)=T_FP/2;
    tp(nexc+(i-1)*nref+2:nexc+i*nref-1)=tref;
    tp(nexc+i*nref)=T_FP/2;
    
    amp(nexc+(i-1)*nref+2:nexc+i*nref-1)=aref;
    phi(nexc+(i-1)*nref+2:nexc+i*nref-1)=pref;
    
    acq(nexc+i*nref)=1; % Acquire at the end of each refocusing cycle
end

function [T_relax,eint,macq]=plot_function(macq1,macq2,del_w,NE,T_E)

macq=zeros(NE,length(del_w));
eint=zeros(1,NE);

figure(1);
for n=1:NE
    macq(n,:)=(macq1(n,:)-macq2(n,:))/2; % Phase cycle
    plot(del_w,abs(macq(n,:))); hold on;
    xlabel('\Delta\omega_{0}');
end

% Filter with last echo to get echo integrals
masy=macq(NE,:);
normasy=sqrt(trapz(del_w,abs(masy.*masy)));
for n=1:NE
    eint(n)=trapz(del_w,abs(macq(n,:).*masy))/normasy;
end
techo=T_E*linspace(1,NE,NE);

nignore=4;
ind=nignore+1:NE;
p=polyfit(techo(ind),log(eint(ind)),1);
log_eint_est=polyval(p,techo(ind));
eint_est=exp(log_eint_est);

T_relax=-1/p(1);
figure(2);
semilogy(eint,'b.-'); hold on;
semilogy(ind,eint_est,'b--');
xlabel('Echo number');
ylabel('Echo amplitude');

function nvals = calc_nvals(tref,pref,aref,T_FP,NE,macq1,macq2,del_w)

nvals=zeros(3,1);

[n,~]=calc_rot_axis_arba([T_FP/2 tref T_FP/2],[0 pref 0],[0 aref 0]);

% Asymptotic signal
%masy=real(macq1(NE,:)-macq2(NE,:)); % For VAN_EXC
masy=imag(macq1(NE,:)-macq2(NE,:)); % For everything else

for k=1:length(del_w)
    % Weigh n^2 by expected asymptotic signal
    nvals=nvals+masy(k)*n(:,1,k).^2;
end
nvals=nvals/sum(masy); % Normalize