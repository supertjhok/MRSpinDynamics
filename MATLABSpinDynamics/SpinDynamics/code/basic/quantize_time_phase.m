% Study effects of time and phase quantization of excitation pulse on echo properties

function [echo_pk,echo_rms]=quantize_time_phase(texc,pexc,tref,pref,T_90,NE,T_FP,delt,delt_vect,delph_vect)

echo_pk=zeros(length(delt_vect),length(delph_vect));
echo_rms=echo_pk;

refmat=calc_refocusing_mat(tref,pref,T_90,NE,T_FP);

% Reference (no quantization)
[ref(1) ref(2)]=cpmg_van_spin_dynamics_refmat(texc,pexc,T_90,T_FP,refmat,delt);
        
% Quantized
for i=1:length(delt_vect)
    texc_approx=delt_vect(i)*round(texc/delt_vect(i));
    for j=1:length(delph_vect)
        pexc_approx=delph_vect(j)*round(pexc/delph_vect(j));
        [outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat(texc_approx,pexc_approx,T_90,T_FP,refmat,delt);
        echo_pk(i,j)=outs(1)/ref(1);
        echo_rms(i,j)=outs(2)/ref(2);
    end
end

figure(1); clf;
for j=1:length(delph_vect)
    plot(delt_vect/T_90,echo_pk(:,j),'bo-'); hold on;
end
set(gca,'FontSize',14);
xlabel('\Delta T/T_{90}');
ylabel('Peak echo amplitude');

figure(2); clf;
for j=1:length(delph_vect)
    plot(delt_vect/T_90,echo_rms(:,j),'bo-'); hold on;
end
set(gca,'FontSize',14);
xlabel('\Delta T/T_{90}');
ylabel('Squared echo area');


