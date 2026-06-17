% Study effects of hard 90 pulses with various pulse powers
% T_90 is the vector of 90 pulse lengths

function [echo_pk,echo_rms]=hard90_effects(name,T_90,T_180,NE,T_FP,delt)

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

refmat=calc_refocusing_mat(tref,pref,T_180/2,NE,T_FP);

% Reference (equal power levels)
[ref(1) ref(2)]=cpmg_van_spin_dynamics_refmat(T_180/2,pexc,T_180/2,T_FP,refmat,delt);

% Unequal power levels
for i=1:length(T_90)
    [outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat_pow(T_90(i),pexc,T_90(i),T_180,T_FP,refmat,delt);
    echo_pk(i)=outs(1)/ref(1);
    echo_rms(i)=outs(2)/ref(2);
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