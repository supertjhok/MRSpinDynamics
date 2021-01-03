% Calculate receiver filtering function for a tuned probe

function [mrx,SNR,vsig] = tuned_probe_rx(sp,pp,macq)

k=sp.k;
T=sp.T;
L=sp.L;
R=sp.R; 
C=sp.C;
Cin=sp.Cin; 
Rin=sp.Rin; 
Rd=sp.Rd;
vn=sp.vn;
in=sp.in;

mf_type=sp.mf_type;

w0=sp.w0; % Larmor frequency
del_w=sp.del_w;
w1_max=(pi/2)/pp.T_90;

s=1i*(w0+del_w*w1_max);
f=imag(s)/(2*pi); % Un-normalized frequency axis
Yin=s*Cin+1/Rin; % Input admittance
Yp=s*C+1/Rd+1./(s*L+R); % Admittance of parallel resonator

tf=1./(1+(s*L+R).*(s*C+1/Rd+Yin));
Zs=1./(Yin+Yp);

% Calculate receiver waveform and noise PSD
mrx=macq.*tf.*(2*pi*f/w0).^2;
% mrx=macq.*tf;
% Note multiplication of mrx(w) by w^2 to account for polarization and
% inductive detection efficiency factors

vni2=4*k*T*R*abs(tf).^2;
pnoise=vn.^2+in.^2*abs(Zs).^2+vni2;

% Matched filtering (maximize SNR at t=0)
switch mf_type
    case 0 % Rectangular in time over window
        theta=atan2(sum(imag(mrx)),sum(real(mrx)));
        %T_W=pp.len_acq; % window length = len_acq
        T_W=0.8*pi; % window length = 0.8*T_180
        mf=sinc(del_w*T_W/(2*pi))*exp(-1i*theta); 
    case 1 % Matched filter (for white noise)
        mf=conj(mrx);
    case 2 % Matched filter (for colored noise)
        mf=conj(mrx)./pnoise;
end

mf=mf/sqrt(trapz(del_w,abs(mf).^2)); % Normalize matched filter amplitude

vsig=trapz(del_w,mrx.*mf); % Signal amplitude (at t=0)
vnoise=sqrt(trapz(f,pnoise.*abs(mf).^2)); % Total integrated noise (V_rms)
SNR=real(vsig)/vnoise; % Signal-to-noise ratio (SNR) per scan in voltage units

NF=pnoise./vni2; % Calculate receiver noise figure

% if sp.plt_rx
%     figure;
%     subplot(1,3,1);
%  %   plot(del_w,real(macq),'b--'); hold on;
%  %   plot(del_w,imag(macq),'r--');
%     plot(del_w,real(mrx)/max(abs(tf)),'b-'); hold on;
%     plot(del_w,imag(mrx)/max(abs(tf)),'r-');
%  %   ylabel('m_{acq}(\omega), m_{rx}(\omega)');
%     ylabel('m_{rx}(\omega)');
%     xlabel('\Delta\omega_{0}/\omega_{1,max}');
% %     
%     subplot(1,2,1);
%     plot(del_w,pnoise/vn^2,'k-'); hold on;
%     ylabel('N(\omega)');
%     xlabel('\Delta\omega_{0}/\omega_{1,max}');
%    title('Filter functions');
%     
%     subplot(1,2,2);
%     plot(del_w,abs(mf),'r-'); hold on;
%     plot(del_w,real(mf),'r-'); hold on; plot(del_w,imag(mf),'b')
%     ylabel('|H(\omega)|');
%     xlabel('\Delta\omega_{0}/\omega_{1,max}');
%      
%     figure(7);
%     subplot(2,1,1); plot(del_w,abs(tf),'b-');
%     ylabel('mag(TF)');
%     title('Receiver gain A(\omega)');
%     
%     subplot(2,1,2); plot(del_w,(180/pi)*phase(tf),'r-');
%     ylabel('phase(TF)');
%     xlabel('\Delta\omega_{0}/\omega_{1,max}');
%     
%     figure;
%     plot(del_w,10*log10(NF),'b-'); hold on;
%     ylabel('NF (dB)');
%     xlabel('\Delta\omega_{0}/\omega_{1,max}');
%     title('Receiver NF');
% end

% mrx=mrx/max(abs(tf)); % Normalize amplitude of received magnetization