% Analytical solution of current in an untuned probe (neglecting C),
% assuming linearly polarized input: cos(w*t+phi)
% Returns solution with quantization step = N x the input RF frequency
% All segments lengths should be integral multiples of the quantization
% step
% --------------------------------------------------------------------
% Soumyajit Mandal, 03/18/2013

function [tvect2,Icr2,tvect,Ic] = untuned_probe_lp(sp,pp)

L=sp.L; R=sp.R;

w0=sp.w0; % Larmor frequency
tp=pp.tref; phi=pp.pref; amp=pp.aref; Rs=pp.Rsref;
w=pp.w; N=pp.N; % quantization step = N x input RF frequency
np=round(tp*N*w/(2*pi)); Ncyc=sum(np); % Total number of simulation cycles
tvect=2*pi*linspace(1,Ncyc,Ncyc)/(w*N);

% Assume zero initial conditions
Ic0=0; % Ic(t)

Ic=zeros(1,Ncyc); cnt=1; phi_eff=phi(1);
% Calculate RF coil current (A/V)
for j=1:length(np)
    tau=L/(R+Rs(j)); % Define circuit time constant tau
    
    tv=2*pi*linspace(1,np(j),np(j))/(w*N); % Time vector
    if cnt>1 % Adjust effective input phase to ensure phase coherence
        phi_eff=w*tvect(cnt-1)+phi(j);
    end
    
    % Homogeneous solution
    ch=Ic0; % Use the fact that the particular solution = 0 at t = 0
    Ich=ch*exp(-tv/tau);
    
    % Particular solution - from Mathematica symbolic expression
    if amp(j)==0
        Icd=0;
    else
        Icd1=(amp(j)/L)*exp(1i*phi_eff)*(exp(1i*w*tv)-exp(-tv/tau))*tau/(1+1i*w*tau); % +w component
        Icd2=(amp(j)/L)*exp(-1i*phi_eff)*(exp(-1i*w*tv)-exp(-tv/tau))*tau/(1-1i*w*tau); % -w component
        Icd=0.5*(Icd1+Icd2);
    end
    Ic(cnt:cnt+np(j)-1)=Ich+Icd; % Add homogeneous and particular solutions
    
    % Calculate initial conditions for next segment - from Mathematica symbolic expression
    Ic0=Ic(cnt+np(j)-1);
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

% Plot results
if sp.plt_tx
    figure(11+sp.plt_tx); clf;
    % Rotating frame
    plot(tvect2*1e6,real(Icr2));  hold on; plot(tvect2*1e6,imag(Icr2),'r-');
    %plot(tvect2*1e6,abs(Icr2),'k-');
    xlabel('Time (\mus)');
    ylabel('Coil current, rotating frame (A/V)');
    
    figure(11+sp.plt_tx+1); clf;
    % RF (imaginary component should be 0)
    plot(tvect*1e6,real(Ic));  hold on; plot(tvect*1e6,imag(Ic),'r-');
    xlabel('Time (\mus)');
    ylabel('Coil current (A/V)');
end
