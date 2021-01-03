% Simulation of CPMG echo amplitude as a function of excitation and refocusing
% pulse amplitudes (keeping on-resonance tip angles constant)

function [echo_rms,avect]=cpmg_powrat

numpts=21;
avect=logspace(-1,1,numpts);

% Refocusing pulses
% tref=pi; % 180 degrees
%pref=pi/2;

%tref=3*pi/4; % 135 degrees
%pref=pi/2;

tref=pi*[0.14 0.72 0.14]; % RP2
pref=(pi/2)*[3 1 3];

nref=length(tref);

% Rectangular excitation pulse
texc=pi/2;
pexc=0;

NE=10;
T_FP=7*pi;
len_acq=T_FP; delt=T_FP;

echo_rms=zeros(numpts,numpts);
for i=1:numpts
    [refmat,del_w]=calc_refocusing_mat_arba(tref/avect(i),pref,avect(i)*ones(1,nref),NE,T_FP);
    for j=1:numpts
        [echo,tvect,echo_pk,echo_rms(i,j)]=cpmg_van_spin_dynamics_refmat_arba(texc/avect(j),pexc,avect(j),refmat,del_w,delt,len_acq);     
    end
    disp(i)
end

ind=find(avect==1); % Normalize to (1,1) case
echo_rms=echo_rms/echo_rms(ind,ind);

figure(1);
surf(20*log10(avect),20*log10(avect),echo_rms);
xlabel('Excitation power (dB)');
ylabel('Refocusing power (dB)');

figure(2);
imagesc(20*log10(avect),20*log10(avect),echo_rms);
xlabel('Excitation power (dB)');
ylabel('Refocusing power (dB)');