% Study effects of (alpha,beta) values on RP2 pulses for fixed power
% consumption = len=(2*alpha+beta)
% Short rectangular excitation pulses assumed 
% Asymptotic echo shape calculated

function [max_rms,max_alpha,max_beta]=hard90_fixpow_asymp(len,T_90,T_180,T_FP,delt)

numpts=20;
alpha=linspace(0,len/2,numpts);
beta=len-2*alpha;

% Rectangular excitation pulse
texc=T_90;
pexc=0;

echo_pk=zeros(1,numpts);
echo_rms=echo_pk;

% Reference (rectangular)
[ref(1) ref(2)]=cpmg_van_spin_dynamics_asymp(texc,pexc,T_180,pi/2,T_90,T_180,T_FP,delt);

% RP2
pref=(pi/2)*[1,3,1];

for i=1:numpts
        tref=T_180*[alpha(i) beta(i) alpha(i)];
        [outs(1) outs(2)]=cpmg_van_spin_dynamics_asymp(texc,pexc,tref,pref,T_90,T_180,T_FP,delt);
        echo_pk(i)=outs(1)/ref(1);
        echo_rms(i)=outs(2)/ref(2);
    disp(i)
end

[max_rms,ind]=max(echo_rms);
max_alpha=alpha(ind);
max_beta=beta(ind);

figure(1);
plot(alpha,echo_pk); hold on;
xlabel('\alpha/\pi');
ylabel('Maximum amplitude of echo');
set(gca,'YDir','normal')

figure(2);
plot(alpha,echo_rms); hold on;
xlabel('\alpha/\pi');
ylabel('Squared integral of echo');
set(gca,'YDir','normal')