% Normal assignment of Tx states:
% (1) negative pulse, (2) zero, (3) positive pulse, (4) Q-switch

function [out]=resonant_tx_thirdorder(params)

% Probe parameters
R1=params.R1; Q0=params.Q0;
Rs = params.Rs; ns = length(Rs); % Length of state list 

% Pulse parameters
ttran = params.ttran; % Tx state transition times
txstat = params.txstat; % Tx state numbers
nstat = length(txstat); % Total number of states

% Plotting parameters
plt_tx = params.plt_tx; clr = params.clr;

% Create Tx state coefficients
a=zeros(ns,4);
for i=1:ns
    a(i,:)=calc_coefficients(params,i);
end

% Solve for coil current and capacitor voltage
y0=zeros(3,1); tf=[]; I_L=[]; V_C=[];
for i = 1:nstat
    acurr=a(txstat(i),:);
    [t,y]=ode45(@(t,y)resonant_probe(t,y,acurr),[ttran(i),ttran(i+1)],y0);
    numt=length(t); y0=y(numt,:);
    if i==1
        tf=t; I_L=y(:,1); V_C=R1*(Q0*y(:,2)+y(:,1));
    else
        % Remove first point (same as last point of previous state)
        tf=vertcat(tf,t(2:numt)); I_L=vertcat(I_L,y(2:numt,1));
        V_C=vertcat(V_C,R1*(Q0*y(2:numt,2)+y(2:numt,1)));
    end
end

if plt_tx
    figure(1);
    stairs(ttran(1:end-1)/(2*pi),txstat);
    
    figure(2);
    plot(tf/(2*pi),I_L,clr); hold on;
    
    figure(3);
    plot(tf/(2*pi),V_C,clr); hold on;
end

out.tf=tf; out.I_L=I_L; out.V_C=V_C;

% ode function
function dy = resonant_probe(t,y,a)

dy = zeros(3,1);
dy(1) = y(2);
dy(2) = y(3);
dy(3) = a(1) + a(2)*y(1) + a(3)*y(2) + a(4)*y(3);

% calculate coefficients for ode solver
% num = tx state number
function [avect] = calc_coefficients(params,num)

% Probe parameters (fixed)
R1 = params.R1; L1 = params.L1; Q0 = params.Q0;

% Tx parameters (can vary between states)
Rs = params.Rs; Ls = params.Ls; VBB = params.VBB;

alpha_R = R1/Rs(num);
alpha_L = Ls(num)/L1;
Isc = VBB(num)/Rs(num);

avect=zeros(1,4);
den=1/(alpha_R*alpha_L*Q0);
avect(1) = Isc*den; avect(2) = -(1+alpha_R)*den;
avect(3) = -(alpha_R*(1+alpha_L)*Q0+1/Q0)*den; avect(4) = -(1+alpha_R*alpha_L)*den;