% Study effects of B1 inhomogeneity

function [echo_pk,echo_rms]=b1_effects(name,T_90,del_w1,NE,T_FP,delt)

tmp=load('echo_timings.mat');
results=tmp.results;

tmp=size(results);
numres=tmp(1);
for i=1:numres
    if strcmpi(name,results{i,1})
        texc=T_90*results{i,2};
        pexc=results{i,3};
    end
end

if strcmpi(name,'rectangular')
    % Rectangular refocusing pulse
    tref=2*T_90;
    pref=pi/2;
else
    % RP2-1.0a refocusing pulse
    tref=2*T_90*[0.14,0.72,0.14];
    pref=pi*[3,1,3]/2;
end

echo_pk=zeros(1,length(del_w1));
echo_rms=echo_pk;

refmat=calc_refocusing_mat(tref,pref,T_90,NE,T_FP);

% Reference (no inhomogeneity)
[ref(1) ref(2)]=cpmg_van_spin_dynamics_refmat(texc,pexc,T_90,T_FP,refmat,delt);

% Inhomogeneous B1
for i=1:length(del_w1)
    refmat=calc_refocusing_mat_delw1(tref,pref,T_90,del_w1(i),NE,T_FP);
    [outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat_delw1(texc,pexc,T_90,del_w1(i),T_FP,refmat,delt);
    echo_pk(i)=outs(1)%/ref(1);
    echo_rms(i)=outs(2)%/ref(2);
end

figure(1); %clf;
plot(del_w1,echo_pk,'bo-'); hold on;
set(gca,'FontSize',14);
xlabel('\Delta\omega_{1}');
ylabel('Peak echo amplitude');

figure(2); %clf;
plot(del_w1,echo_rms,'bo-'); hold on;
set(gca,'FontSize',14);
xlabel('\Delta\omega_{1}');
ylabel('Squared echo area');