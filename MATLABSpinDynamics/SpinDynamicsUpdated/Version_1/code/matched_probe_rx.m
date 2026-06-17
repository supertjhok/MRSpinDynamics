% Calculate receiver filtering function for a matched probe
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19

function [mrx,SNR] = matched_probe_rx(sp,pp,macq,tf1,tf2)

k=sp.k; T=sp.T; 
L=sp.L; f0=sp.f0; Q=sp.Q;

Rc=(2*pi*f0*L)/Q; % Coil series resistance (Ohms)

mf_type=sp.mf_type;
del_w=sp.del_w;
w1_max=(pi/2)/pp.T_90;

f=(2*pi*f0+del_w*w1_max)/(2*pi); % Un-normalized frequency axis

mrx=macq.*tf2; % Signal
vni2=4*k*T*Rc*abs(tf1).^2; % Probe noise PSD

% Receiver noise (assume white)
Fn=10^(sp.NF/10); % Noise factor
vn2=4*k*T*sp.Rin*(Fn-1)*ones(1,length(f));

pnoise=vni2+vn2; % Total noise PSD

% Matched filtering (maximize SNR at t=0)
switch mf_type
    case 1 % Matched filter (for white noise)
        mf=conj(mrx);
    case 2 % Matched filter (for colored noise)
        mf=conj(mrx)./pnoise;
end
mf=mf/sqrt(trapz(del_w,abs(mf).^2)); % Normalize matched filter amplitude

vsig=trapz(del_w,mrx.*mf); % Signal amplitude (at t=0)
vnoise=sqrt(trapz(f,pnoise.*abs(mf).^2)); % Total integrated noise (V_rms)
SNR=real(vsig)/vnoise; % Signal-to-noise ratio (SNR) per scan in voltage units
SNR=SNR/1e8;

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