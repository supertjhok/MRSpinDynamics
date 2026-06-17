% Study effects of refocusing pulses (RP2 type, length fixed = 2 x T_90)
% RP2-1.0 pulses can be described by a single parameter, alpha

function [echo_pk,echo_rms]=ref_effects(texc,pexc,alpha,T_90,del_w1,NE,T_FP,delt)

echo_pk=zeros(1,length(alpha));
echo_rms=echo_pk;

% Vary alpha
for i=1:length(alpha)
    
    % RP2 definition
    tref=2*T_90*[alpha(i),1-2*alpha(i),alpha(i)];
    pref=pi*[3,1,3]/2;
    
    refmat=calc_refocusing_mat_delw1(tref,pref,T_90,del_w1,NE,T_FP);
    [outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat_delw1(texc,pexc,T_90,del_w1,T_FP,refmat,delt);
    echo_pk(i)=outs(1);%/ref(1);
    echo_rms(i)=outs(2);%/ref(2);
end

figure(1); %clf;
plot(alpha,echo_pk,'bo-'); hold on;
set(gca,'FontSize',14);
xlabel('\alpha');
ylabel('Peak echo amplitude');

figure(2); %clf;
plot(alpha,echo_rms,'bo-'); hold on;
set(gca,'FontSize',14);
xlabel('\alpha');
ylabel('Squared echo area');