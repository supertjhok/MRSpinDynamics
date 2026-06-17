% Analytical solution of current in a tuned probe
% assuming linearly polarized input: cos(w*t+phi)
% Returns solution with quantization step = N x the input RF frequency
% All segments lengths should be integral multiples of the quantization
% step
% --------------------------------------------------------------------
% v 0.1 Soumyajit Mandal, 03/18/2013
% v 0.2 Soumyajit Mandal, 07/23/2013 -> simplified analytical expressions
% prevent numerical blowup and increase calculation speed

function [tvect2,Icr2,tvect,Icr] = tuned_probe_lp_Orig(sp,pp)

L=sp.L; R=sp.R; C=sp.C;

% Perform all computations in normalized time t x wp
wp=1/sqrt(L*C); Z0=sqrt(L/C);

w0=sp.w0/wp; % Larmor frequency
tp=wp*pp.tref; phi=pp.pref; amp=pp.aref; Rs=pp.Rsref;
w=pp.w/wp; N=pp.N; % quantization step = N x input RF frequency
np=round(tp*N*w/(2*pi));
Ncyc=sum(np); % Total number of simulation cycles
% Ncyc=size(np,2);
tvect=2*pi*linspace(1,Ncyc,Ncyc)/(w*N);

% Assume zero initial conditions
Ic0=[0; 0]; % Ic(t), dIc(t)/dt

Ic=zeros(1,Ncyc); cnt=1;
% Calculate RF coil current (A/V)
for j=1:length(np)
    gamma=0.5*(R/Z0+Z0/Rs(j));
    wn=sqrt(1+R/Rs(j)); alpha=sqrt(gamma^2-wn^2);
    lambda=-gamma+[1 -1]*alpha;
    lmat=[[1 1]; lambda];
    
    tv=2*pi*linspace(1,np(j),np(j))/(w*N); tve=tv(end); % Time vector
    phi_eff=w*tvect(cnt)-2*pi/N+phi(j); % Adjust effective input phase to ensure phase coherence
    
    % Homogeneous solution
    ch=lmat\Ic0; % Use the fact that the particular solution = 0 at t = 0
    Ich=ch(1)*exp(lambda(1)*tv)+ch(2)*exp(lambda(2)*tv);
    dIch=ch(1)*lambda(1)*exp(lambda(1)*tv(end))+ch(2)*lambda(2)*exp(lambda(2)*tv(end));
    
    % Particular solution - from Mathematica symbolic expression
    if amp(j)==0
        Icd=0;
        dIcd=0;
    else
        % +w component
        Icd1=(amp(j)/Rs(j))*exp(1i*phi_eff)*(exp(1i*w*tv)+((lambda(2)-1i*w)*exp(lambda(1)*tv)/(2*alpha)-...
            (lambda(1)-1i*w)*exp(lambda(2)*tv))/(2*alpha))/(wn^2-w^2+2*1i*gamma*w);
        % -w component
        Icd2=(amp(j)/Rs(j))*exp(-1i*phi_eff)*(exp(-1i*w*tv)+((lambda(2)+1i*w)*exp(lambda(1)*tv)/(2*alpha)-...
            (lambda(1)+1i*w)*exp(lambda(2)*tv))/(2*alpha))/(wn^2-w^2-2*1i*gamma*w);
        Icd=0.5*(Icd1+Icd2);
        
        % +w component
        dIcd1=(amp(j)/Rs(j))*exp(1i*phi_eff)*(1i*w*exp(1i*w*tve)+(lambda(1)*(lambda(2)-1i*w)*exp(lambda(1)*tve)/(2*alpha)-...
            lambda(2)*(lambda(1)-1i*w)*exp(lambda(2)*tve))/(2*alpha))/(wn^2-w^2+2*1i*gamma*w);
        % -w component
        dIcd2=(amp(j)/Rs(j))*exp(-1i*phi_eff)*(-1i*w*exp(-1i*w*tve)+(lambda(1)*(lambda(2)+1i*w)*exp(lambda(1)*tve)/(2*alpha)-...
            lambda(2)*(lambda(1)+1i*w)*exp(lambda(2)*tve))/(2*alpha))/(wn^2-w^2-2*1i*gamma*w);
        dIcd=0.5*(dIcd1+dIcd2);
    end
    Ic(cnt:cnt+np(j)-1)=Ich+Icd; % Add homogeneous and particular solutions
    
    % Calculate initial conditions for next segment - from Mathematica symbolic expression
    Ic0(1)=Ic(cnt+np(j)-1);
    Ic0(2)=dIch+dIcd;
    cnt=cnt+np(j);
end

% Demodulate to baseband / rotating frame
Icr=Ic.*exp(-1i*w0*tvect);

% Average over windows of length = 1/(2*w) to remove 2*w component
%numpts=floor(Ncyc*2/N); tvect2=zeros(1,numpts); Icr2=tvect2;
%for i=1:numpts
%    ind=(i-1)*N/2+1:i*N/2;
%    tvect2(i)=mean(tvect(ind));
%    Icr2(i)=mean(Icr(ind));
%end

% Average over windows of length = 1/(w) to remove 2*w component
numpts=floor(Ncyc/N); tvect2=zeros(1,numpts); Icr2=tvect2;
for i=1:numpts
    ind=(i-1)*N+1:i*N;
    tvect2(i)=mean(tvect(ind));
    Icr2(i)=mean(Icr(ind));
end

% Convert to real time
tvect=tvect/wp;
tvect2=tvect2/wp;

% Plot results
if sp.plt_tx
    figure(1+sp.plt_tx); clf;
    % Rotating frame
    %plot(tvect2*1e6,real(Icr2));  hold on; plot(tvect2*1e6,imag(Icr2),'r-');
    %plot(tvect2*1e6,abs(Icr2),'k-');
    %ylabel('Rotating frame coil current (A/V)');
    % RF (imaginary component should be 0)
    plot(tvect*1e6,real(Ic));  hold on; plot(tvect*1e6,imag(Ic),'r-');
    xlabel('Time (\mus)');
    ylabel('Actual coil current (A/V)');
end
