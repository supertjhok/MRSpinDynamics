% Calculate receiver filtering function for a matched probe
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19

function [mrx,pnoise,f] = matched_probe_rx_no_mf(sp,pp,macq,tf1,tf2)

k=sp.k; T=sp.T; 
L=sp.L; f0=sp.f0; Q=sp.Q;

Rc=(2*pi*f0*L)/Q; % Coil series resistance (Ohms)

del_w=sp.del_w; % Load w0 map
w_1r=sp.w_1r; % Load receive coil sensitivity factor map
w1_max=(pi/2)/pp.T_90; % B1 normalization factor

f=(2*pi*f0+del_w*w1_max)/(2*pi); % Un-normalized frequency axis

% Calculate signal and probe noise, including 
mrx=macq.*tf2.*w_1r; % Signal
vni2=4*k*T*Rc*abs(tf1).^2; % Probe noise PSD

% Receiver noise (assume white)
Fn=10^(sp.NF/10); % Noise factor
vn2=4*k*T*sp.Rin*(Fn-1)*ones(1,length(f));

pnoise=vni2+vn2; % Total noise PSD

if sp.plt_rx
    figure;
    plot(del_w,real(macq),'b--'); hold on;
    plot(del_w,imag(macq),'r--');
    plot(del_w,real(mrx)/max(abs(tf2)),'b-');
    plot(del_w,imag(mrx)/max(abs(tf2)),'r-');
    title('Asymptotic magnetization')
    xlabel('\Delta\omega_{0}/\omega_{1,max}')
    ylabel('M_{asy} (dashed), M_{rx} (solid)')
    set(gca,'FontSize',14);
    
    figure;
    plot(del_w,sqrt(pnoise)*1e9);
    title('Noise PSD at receiver')
    xlabel('\Delta\omega_{0}/\omega_{1,max}')
    ylabel('v_{n}, nV/Hz^{1/2}')
    set(gca,'FontSize',14);
end