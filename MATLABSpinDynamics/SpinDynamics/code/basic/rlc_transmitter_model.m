% Transmitter model
% Assume series RLC transmitter
% Based on expressions in Mehring & Waugh (1972)
% Written by: Soumyajit Mandal, 10/09/10

function rlc_transmitter_model(tp,amp,phi,tf,f0,f,Q)

tp=tp/1e6;
tf=tf/1e6;

w0=f0*2*pi; w=f*2*pi; 
delt=1/(1e2*w); % 100 points per RF cycle

% Define parameters
dw=w0/Q;
wr=sqrt(w0^2-(dw/2)^2);
del=wr-w;
ph=atan(Q*(w^2-w0^2)/(w*w0));
i0=amp*cos(ph);

tot=sum(tp)+sum(tf); % Total simulation time
numpts=round(tot/delt);
tvect=linspace(0,(numpts-1)*delt,numpts);
irf=zeros(1,numpts);

np=length(tp); tw=0;
for j=1:np
    % Pulse turn-on edge
    t=tvect-tw;
    iq=exp(-dw*t/2).*(sin(del*t+2*ph-2*phi(j))-((wr+w)/(2*wr))*(sin(del*t)+...
        sin(del*t+2*ph-2*phi(j)))-(dw/(4*wr))*(cos(del*t)-cos(del*t+2*ph-2*phi(j)))); % Quadrature current
    ip=exp(-dw*t/2).*(cos(del*t+2*ph-2*phi(j))-((wr+w)/(2*wr))*(cos(del*t)+...
        cos(del*t+2*ph-2*phi(j)))+(dw/(4*wr))*(sin(del*t)-sin(del*t+2*ph-2*phi(j)))); % In-phase current
    it=i0(j)*((1+ip).*sin(w*tvect+ph-phi(j))+iq.*cos(w*tvect+ph-phi(j)));
    it(tvect<tw)=0;
    irf=irf+it;
        
    tw=tw+tp(j);
    % Pulse turn-off edge
    t=tvect-tw; 
    iq=exp(-dw*t/2).*(sin(del*t+2*ph-2*phi(j))-((wr+w)/(2*wr))*(sin(del*t)+...
        sin(del*t+2*ph-2*phi(j)))-(dw/(4*wr))*(cos(del*t)-cos(del*t+2*ph-2*phi(j)))); % Quadrature current
    ip=exp(-dw*t/2).*(cos(del*t+2*ph-2*phi(j))-((wr+w)/(2*wr))*(cos(del*t)+...
        cos(del*t+2*ph-2*phi(j)))+(dw/(4*wr))*(sin(del*t)-sin(del*t+2*ph-2*phi(j)))); % In-phase current
    it=-i0(j)*((1+ip).*sin(w*tvect+ph-phi(j))+iq.*cos(w*tvect+ph-phi(j)));
    it(tvect<tw)=0;
    irf=irf+it;
    
    tw=tw+tf(j);
end

figure(1); clf;
plot(tvect*1e6,irf);
xlabel('Time (\mus)');
ylabel('Coil current');