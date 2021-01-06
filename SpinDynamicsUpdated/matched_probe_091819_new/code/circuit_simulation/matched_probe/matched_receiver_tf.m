function [tf1,tf2] = matched_receiver_tf(sp,pp)

% Read probe parameters
L=sp.L;
Q=sp.Q;
f0=sp.f0;
Rs=sp.Rs;
C1=sp.C1;
C2=sp.C2;
fin=sp.fin;
    
% Calculate normalization parameters
n=C1/C2; 
Z0=sqrt(L/C1); 
wp=1/sqrt(L*C1);
wn=(2*pi*fin)/wp; % Normalized input frequency
Rc=(2*pi*f0*L)/Q; % Coil series resistance (Ohms)

c1=n*Z0/Rs; 
c2=(Rc/Rs)*(n+1)+1;
c3=Rc/Z0+(Z0/Rs)*(n+1);
Vs0=2*sqrt(Rc/Rs);

% Simulation frequency range
w1=pi/(2*pp.T_90);
wv=(2*pi*f0+w1*sp.del_w)/wp;

% Transfer functions
tf1 = (1i*wv)./((1i*wv).^3+c3*(1i*wv).^2+c2*(1i*wv)+c1); % Noise

% Signal, including inductive detection and polarization effects
tf2 = -1i*(1i*wv).^3./((1i*wv).^3+c3*(1i*wv).^2+c2*(1i*wv)+c1);