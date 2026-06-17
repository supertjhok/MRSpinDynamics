% Compare diffusion sensitivity and SNR for rectangular CPMG sequences with
% varying 90 / 180 ratio

function diff_rect_rat(rat,Nfinal)

numpts=length(rat);
snr=zeros(1,numpts);

% Pulse definitions
texc=pi/2;
pexc=pi;
aexc=1;

pref=pi/2;
aref=1;

T_E=15*pi;

% Variables to store echo integrals
eta=linspace(0,3,3e2+1);
echo_int=zeros(length(eta),numpts);
echo_int2=echo_int;
diff_results={};

for j=1:numpts
    disp(j)
    tref=rat(j)*texc;
    
    % Calculate echo integrals (no diffusion)
    [masy,del_w]=asy_new(texc,pexc,aexc,tref,pref,aref,T_E);
    snr(j)=sqrt(trapz(del_w,abs(masy).^2));
    
    % Calculate diffusion rate spectrum
    [axistime,disttimes]=...
        coherences_new(texc,pexc,aexc,tref,pref,aref,T_E,Nfinal);
    diff_results{j,1}=axistime; diff_results{j,2}=disttimes;
    
    % Calculate echo integrals (with diffusion)
    a_l=disttimes(Nfinal,:);
    eta_l=1./axistime;
    
    for k=1:length(eta)
        echo_int(k,j)=sum(a_l.*exp(-eta_l*eta(k))); % Echo integral
        echo_int2(k,j)=echo_int(k,j)*exp(eta(k)); % Echo  integral - relative to on-resonance case
    end
end

save('diff_rect_rat_results.mat','rat','eta','snr','diff_results','echo_int','echo_int2');

figure(2);
plot(rat,snr);
xlabel('T_{180} / T_{90}');
ylabel('Asymptotic echo amplitude (no diffusion)');

figure(3);
plot(rat,snr/max(snr),'b--'); hold on;
eta_vals=[0,0.5,1,2,3];
clrs={'b-','r-','k-','m-','g-'};
for j=1:length(eta_vals)
    plot(rat,echo_int(eta==eta_vals(j),:)/max(echo_int(eta==eta_vals(j),:)),clrs{j});
end
xlabel('T_{180} / T_{90}');
ylabel('Asymptotic echo amplitude (normalized)');

figure(4);
imagesc(rat,eta,echo_int2);
xlabel('T_{180} / T_{90}');
ylabel('\eta = \gamma^{2}g^{2}Dt_{E}^{3}N / 12');