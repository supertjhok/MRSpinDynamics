% Simulation of CPMG echo amplitude as a function of excitation and refocusing
% pulse tip angles (keeping amplitudes constant)

function [echo_rms,lvect]=cpmg_timrat

numpts=20;
lvect=linspace(0.1,2,numpts);

% Refocusing pulses
tref=pi; % 180 degrees, rectangular
pref=pi/2;

%tref=pi*[0.14 0.72 0.14]; % RP2-1.0
%pref=(pi/2)*[3 1 3];

nref=length(tref);

% Rectangular excitation pulse
texc=pi/2;
pexc=0;

NE=10;
T_FP=7*pi;
len_acq=T_FP; delt=T_FP;

echo_rms=zeros(numpts,numpts);
for i=1:numpts
    [refmat,del_w]=calc_refocusing_mat_arba(tref*lvect(i),pref,ones(1,nref),NE,T_FP);
    for j=1:numpts
        [~,~,~,echo_rms(i,j)]=cpmg_van_spin_dynamics_refmat_arba(texc*lvect(j),pexc,1,refmat,del_w,delt,len_acq);     
    end
    disp(i)
end

save('cpmg_timrat.mat','echo_rms','lvect');
[~,ind]=min(abs(lvect-1)); % Normalize to (1,1) case
echo_rms=echo_rms/echo_rms(ind,ind);

figure(1);
surf(lvect,lvect,echo_rms);
xlabel('Excitation length');
ylabel('Refocusing length');

figure(2);
imagesc(lvect,lvect,echo_rms);
xlabel('Excitation length');
ylabel('Refocusing length');