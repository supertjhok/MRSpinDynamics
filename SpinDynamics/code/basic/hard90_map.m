% Study effects of (alpha,beta) values on RP2 pulses
% Short rectangular excitation pulses assumed

function [echo_pk,echo_rms,alpha,beta]=hard90_map(T_90,T_180,NE,T_FP,delt)

alpha=0:0.1:4;
beta=0:0.1:4;

% Rectangular excitation pulse
texc=T_90;
pexc=0;

echo_pk=zeros(length(alpha),length(beta));
echo_rms=echo_pk;

% Reference (rectangular)
refmat=calc_refocusing_mat(T_180,pi/2,T_180/2,NE,T_FP);
[ref(1) ref(2)]=cpmg_van_spin_dynamics_refmat_pow(texc,pexc,T_90,T_180,T_FP,refmat,delt);

% RP2
pref=(pi/2)*[1,3,1];

for i=1:length(alpha)
    for j=1:length(beta)
        tref=T_180*[alpha(i) beta(j) alpha(i)];
        refmat=calc_refocusing_mat(tref,pref,T_180/2,NE,T_FP);
        [outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat_pow(texc,pexc,T_90,T_180,T_FP,refmat,delt);
        echo_pk(j,i)=outs(1)/ref(1);
        echo_rms(j,i)=outs(2)/ref(2);
    end
    disp(i)
end

close all;

figure(1);
imagesc(alpha,beta,echo_pk);
xlabel('\alpha/\pi');
ylabel('\beta/\pi');
set(gca,'YDir','normal')

figure(2);
imagesc(alpha,beta,echo_rms);
xlabel('\alpha/\pi');
ylabel('\beta/\pi');
set(gca,'YDir','normal')