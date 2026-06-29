% Two-capacitor matching network computation
% n = impedance transformation ratio
% Q = quality factor of coil at w=1 (matched frequency)

function [w1]=matching_network(n,Q)

w1=zeros(2,length(Q));

% Possible resonant frequencies of LC tank (coil + tuning capacitor)
w1(1,:)=1./sqrt((1+sqrt(1-(1-1/n)*(1+1./Q.^2)))./(1+1./Q.^2));
w1(2,:)=1./sqrt((1-sqrt(1-(1-1/n)*(1+1./Q.^2)))./(1+1./Q.^2));

R0=n; % Resistance to match to
X0=n*(Q.*(1-1./w1(2,:).^2)-1./(Q.*w1(2,:).^2)); % Matching capacitor

numfreq=1e4; i=sqrt(-1);
Z=zeros(length(Q),numfreq);
Qeff=zeros(1,length(Q));

for j=1:length(Q)
    w=linspace(1-2/Q(j),1+1.5/Q(j),numfreq);
    Z(j,:)=(1+i*Q(j).*w)./(i*w./(Q(j).*w1(2,j).^2)+(1-(w./w1(2,j)).^2));
    Rtot=R0+real(Z(j,:));
    Xtot=X0(j)-imag(Z(j,:));
    
    % Calculate effective Q
    [tmp,ind1]=min(abs(Rtot+Xtot));
    [tmp,ind2]=min(abs(Rtot-Xtot));
    Qeff(j)=1/abs(w(ind1)-w(ind2));
    
    figure(3);
    plot(w,Rtot,'b-'); hold on;
    plot(w,Xtot,'r-');
    plot([w(ind1),w(ind2)],[Rtot(ind1),Rtot(ind2)],'k*');
end

figure(1);
plot(Q,w1(1,:),'bo-'); hold on;
plot(Q,w1(2,:),'rs-');

figure(2);
plot(Q,Qeff,'bo-'); hold on;