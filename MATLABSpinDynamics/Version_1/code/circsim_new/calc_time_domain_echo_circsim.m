% Calculate time domain echo from spectrum
% Use zero-filling to get smoother echo shapes

function [echo,tvect]=calc_time_domain_echo_circsim(spect,wvect,plt)

zf=4; % Zero-filling ratio
T_180=pi; % Normalized T_180 time

ts=pi/(zf*max(wvect));
numpts=length(spect);

spect_zf=zeros(zf*numpts,1);
spect_zf((zf-1)*floor(numpts/2)+1:(zf+1)*floor(numpts/2),:)=spect(1:2*floor(numpts/2));

tvect=ts*linspace(-zf*floor(numpts/2),zf*floor(numpts/2),zf*numpts);
echo=zf*ifftshift(ifft(fftshift(spect_zf)));

if plt
    figure(99);
    plot(tvect/T_180,real(echo),'b-'); hold on;
    plot(tvect/T_180,imag(echo),'r-');
    plot(tvect/T_180,abs(echo),'k-');
    xlabel('Normalized time, t / T_{180}');
    xlim([-3 3]);
end