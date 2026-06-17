% Study effects of hard 90 pulses with various pulse powers
% T_90 is the vector of 90 pulse lengths

function [echo_pk,echo_rms]=hard90_effects_plot(name,T_90,T_180,NE,T_FP,delt)

zf=8; % zero-filling to get smoother spectra
f1=1/(2*T_180*1e-6);

tmp=load('ref_timings.mat');
results=tmp.results;

tmp=size(results);
numres=tmp(1);
for i=1:numres
    if strcmpi(name,results{i,1})
        tref=T_180*results{i,2};
        pref=results{i,3};
    end
end

% Rectangular excitation pulse
pexc=0;

echo_pk=zeros(1,length(T_90));
echo_rms=echo_pk;


% Reference (rectangular, equal power levels)
refmat=calc_refocusing_mat(T_180,pi/2,T_180/2,NE,T_FP);
[ref(1) ref(2)]=cpmg_van_spin_dynamics_refmat(T_180/2,pexc,T_180/2,T_FP,refmat,delt);

refmat=calc_refocusing_mat(tref,pref,T_180/2,NE,T_FP);

% Unequal power levels
for i=1:length(T_90)
    [echo,tvect,outs(1),outs(2)]=cpmg_van_spin_dynamics_refmat_pow_plot(T_90(i),pexc,T_90(i),T_180,T_FP,refmat,delt);
    echo_pk(i)=outs(1)/ref(1);
    echo_rms(i)=outs(2)/ref(2);
    
    figure(3);
    plot(tvect/(T_180*1e-6),abs(echo),'b-'); hold on;
    set(gca,'FontSize',14);
    xlabel('Normalized time t / T_{180}');
    ylabel('Echo amplitude')
    
    fs=1/(tvect(2)-tvect(1));
    fvect=linspace(-fs/2,fs/2,length(tvect)*zf);
    echo_zf=zeros(zf*length(echo),1);
    echo_zf((zf-1)*length(echo)/2:(zf+1)*length(echo)/2-1)=echo;
    spect=zf*abs(fftshift(fft(real(echo_zf))+fft(imag(echo_zf))))/length(echo_zf);
    
    figure(4); %clf;
    plot(fvect/f1,spect,'k-'); hold on;
    set(gca,'FontSize',14);
    xlabel('Normalized frequency \omega/\omega_{1}');
    ylabel('|Echo spectrum|')
end

figure(1); %clf;
plot(T_180./T_90,echo_pk,'bo-'); hold on;
set(gca,'FontSize',14);
xlabel('T_{180} / T_{90}');
ylabel('Peak echo amplitude');

figure(2); %clf;
plot(T_180./T_90,echo_rms,'bo-'); hold on;
set(gca,'FontSize',14);
xlabel('T_{180} / T_{90}');
ylabel('Squared echo area');