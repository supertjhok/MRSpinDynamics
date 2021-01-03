% Find rotating-field current in a tuned and matched probe
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% 04/04/19: Modified to include coil response to discontinuities in the 
% input voltage waveform
% 04/09/19: Modified to include absolute RF phase parameter (psi)

function [tvec2, yr2, tvec, y, tf1, tf2] = find_coil_current_orig(sp,pp)

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

% Read pulse sequence parameters
tp=wp*pp.tp;
phi=pp.phi;
amp=pp.amp;
N=pp.N; % quantization step = N x input RF frequency
psi=pp.psi;

% Simulation frequency range
w1=pi/(2*pp.T_90);
wv=(2*pi*f0+w1*sp.del_w)/wp;

ttot=sum(tp); % Total simulation time
delt=2*pi/(wn*N); % Step size
ntot=floor(ttot/delt)+1; % Total number of points
tvec=linspace(0,ttot,ntot);

% Impulse response of probe
s=tf('s');
tf0=Vs0/(s^3+c3*s^2+c2*s+c1);
[y_imp,~]=impulse(tf0,0:delt:ttot);

% Set ODE solver options
opts = odeset('RelTol',1e-3,'AbsTol',1e-6);

% Solve differential equation to find coil current
time_el=0; % Elapsed time
ysin=zeros(ntot,3); ycos=ysin;
ysin_imp=zeros(ntot,1); ycos_imp=ysin_imp; 
for i=1:length(tp)
    ode_params=[c1 c2 c3 amp(i)*Vs0 phi(i) wn psi];
    if i==1
        ind_last=1;
        ycos0=[0 0 0]; % Assume zero initial conditions
        ysin0=[0 0 0];
    else
        ind_last=ind2;
        ycos0=ycos(ind_last,:); % Initial conditions from last segment
        ysin0=ysin(ind_last,:);
    end
    ind=find((tvec >= time_el) & (tvec <= time_el+tp(i)));
    ind1=ind_last; ind2=ind(length(ind));
    [~,ycos(ind1:ind2,:)] = ode45(@(t,y) odefcn_cos(t,y,ode_params,Rc,Rs), tvec(ind1:ind2), ycos0, opts);
    [~,ysin(ind1:ind2,:)] = ode45(@(t,y) odefcn_sin(t,y,ode_params,Rc,Rs), tvec(ind1:ind2), ysin0, opts);
    time_el=time_el+tp(i);
    
    % Add impulse responses
    ysin_imp(ind1:ntot)=ysin_imp(ind1:ntot)+amp(i)*sin(wn*tvec(ind1)+phi(i)+psi)*y_imp(1:ntot-ind1+1);
    ysin_imp(ind2:ntot)=ysin_imp(ind2:ntot)-amp(i)*sin(wn*tvec(ind2)+phi(i)+psi)*y_imp(1:ntot-ind2+1);
    
    ycos_imp(ind1:ntot)=ycos_imp(ind1:ntot)+amp(i)*cos(wn*tvec(ind1)+phi(i)+psi)*y_imp(1:ntot-ind1+1);
    ycos_imp(ind2:ntot)=ycos_imp(ind2:ntot)-amp(i)*cos(wn*tvec(ind2)+phi(i)+psi)*y_imp(1:ntot-ind2+1);
end

% Create complex solution
y=ycos(:,1)+1i*ysin(:,1); % Without impulses
y=y+(ycos_imp+1i*ysin_imp); % Add impulses

% Separately plot responses i) without impulses, and ii) to impulses (for debug) 
% figure;
% subplot(2,1,1); plot(tvec,ycos(:,1)); hold on; plot(tvec,ysin(:,1));
% subplot(2,1,2); plot(tvec,ycos_imp); hold on; plot(tvec,ysin_imp);

% Move to rotating frame
yr=y.*exp(-1i*wn*tvec')*exp(-1i*psi);

% Average over windows of length = 1/(2*w) to remove residual 2*w component
ntot2=floor(ntot*2/N); tvec2=zeros(1,ntot2); yr2=tvec2;
for i=1:ntot2
    ind=(i-1)*N/2+1:i*N/2;
    tvec2(i)=mean(tvec(ind));
    yr2(i)=mean(yr(ind));
end

% Convert to real time
tvec=tvec/wp; 
tvec2=tvec2/wp;

% Plot solution if requested
if sp.plt_tx
%     figure; subplot(1,2,1);
%     plot(tvec*1e6,ycos(:,1)); hold on; plot(tvec*1e6,ysin(:,1));
%     ylabel('RF coil current (normalized)'); xlabel('Time (\mus)');
%     legend(['Real';'Imag']); set(gca,'FontSize',14);
    
     figure; 
    plot(tvec*1e6,ycos(:,1)); hold on; plot(tvec*1e6,ysin(:,1));
    ylabel('RF coil current (normalized)'); xlabel('Time (\mus)');
    legend(['Real';'Imag']); set(gca,'FontSize',14);
    whiteBg;
    setSize;
    font;
    
%     subplot(1,2,2);
%     %plot(tvec*1e6,real(yr)); hold on; plot(tvec*1e6,imag(yr));
%     plot(tvec2*1e6,real(yr2)); hold on; plot(tvec2*1e6,imag(yr2));
%     ylabel('Rotating frame current (normalized)'); xlabel('Time (\mus)');
%     legend(['Real';'Imag']); set(gca,'FontSize',14);
figure
    %plot(tvec*1e6,real(yr)); hold on; plot(tvec*1e6,imag(yr));
    plot(tvec2*1e6,real(yr2),'LineWidth',2); hold on; plot(tvec2*1e6,imag(yr2),'LineWidth',2);
    ylabel('Rotating frame current (normalized)'); xlabel('Time (\mus)');
    legend(['Real';'Imag']); 
    set(gca,'FontSize',14);
    whiteBg;
    setSize;
    font;
end

% Find receiver and noise transfer functions
[tf1,tf2] = receiver_tf(wv,ode_params);

% Plot solution if requested
if sp.plt_rx
    figure;
    subplot(2,1,1); plot(wv*wp/(2*pi*1e6),abs(tf1)); hold on;
    subplot(2,1,1); plot(wv*wp/(2*pi*1e6),abs(tf2));
    ylabel('|TF|'); xlabel('Frequency (MHz)');
    ax=gca; set(ax,'FontSize',14); 
    whiteBg;
    legend(ax,'V_{out}/V_{in}','V_{out}/M');
    setSize;
    
    subplot(2,1,2); plot(wv*wp/(2*pi*1e6),angle(tf1)); hold on;
    subplot(2,1,2); plot(wv*wp/(2*pi*1e6),angle(tf2));
    ylabel('angle(TF) (rad)'); xlabel('Frequency (MHz)');
    set(gca,'FontSize',14);
    ax=gca; set(ax,'FontSize',14); 
    setSize;
    %legend(ax,'V_{out}/V_{in}','V_{out}/M');
end

function dydt = odefcn_cos(t,y,params,Rc,Rs)

c1=params(1); c2=params(2); c3=params(3); Vs0=params(4);
phi=params(5); wn=params(6); psi=params(7);

dydt = zeros(3,1);
dydt(1) = y(2);
dydt(2) = y(3);
dydt(3) = (-c3*y(3)-c2*y(2)-c1*y(1)-Vs0*wn*sin(wn*t+phi+psi));

function dydt = odefcn_sin(t,y,params,Rc,Rs)

c1=params(1); c2=params(2); c3=params(3); Vs0=params(4);
phi=params(5); wn=params(6); psi=params(7);

dydt = zeros(3,1);
dydt(1) = y(2);
dydt(2) = y(3);
dydt(3) = -c3*y(3)-c2*y(2)-c1*y(1)+Vs0*wn*cos(wn*t+phi+psi);

function [tf1,tf2] = receiver_tf(wv,params)

c1=params(1); c2=params(2); c3=params(3); 
tf1 = (1i*wv)./((1i*wv).^3+c3*(1i*wv).^2+c2*(1i*wv)+c1); % Noise

% Signal, including inductive detection and polarization effects
tf2 = -1i*(1i*wv).^3./((1i*wv).^3+c3*(1i*wv).^2+c2*(1i*wv)+c1);