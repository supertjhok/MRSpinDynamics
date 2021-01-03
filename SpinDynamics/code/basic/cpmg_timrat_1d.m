% Simulation of CPMG echo amplitude as a function of excitation and refocusing
% pulse tip angles (keeping amplitudes constant)
% Acquire 1D plot (fixed T_180:T_90 ratio)

function [echo_rms,lvect]=cpmg_timrat_1d(rat)

numpts=20;
lvect=linspace(0.1,2,numpts);

% Rectangular excitation pulse
texc=pi/2;
pexc=0;

% Refocusing pulses
tref=texc*rat; % 180 degrees, rectangular
pref=pi/2;

%tref=texc*rat*[0.14 0.72 0.14]; % RP2-1.0
%pref=(pi/2)*[3 1 3];

nref=length(tref);

NE=10;
T_FP=7*pi;
len_acq=T_FP; delt=T_FP;

echo_rms=zeros(numpts,1);
for i=1:numpts
    [refmat,del_w]=calc_refocusing_mat_arba(tref*lvect(i),pref,ones(1,nref),NE,T_FP);
    [~,~,~,echo_rms(i)]=cpmg_van_spin_dynamics_refmat_arba(texc*lvect(i),pexc,1,refmat,del_w,delt,len_acq);
    disp(i)
end

save('cpmg_timrat_1d.mat','echo_rms','lvect');
%[~,ind]=min(abs(lvect-1)); % Normalize to (1,1) case
%echo_rms=echo_rms/echo_rms(ind);

figure(1);
plot(lvect,echo_rms,'bo-');
hold on;
xlabel('Normalized pulse length');
ylabel('Normalized echo amplitude');