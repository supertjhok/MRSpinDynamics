% Calculate time domain echo from spectrum
% Use zero-filling to get smoother echo shapes

function [echo,tvect]=calc_time_domain_echo(spect,wvect)

zf=4; % Zero-filling ratio

ts=pi/(zf*max(wvect));
numpts=length(spect);

spect_zf=zeros(zf*numpts,1);
spect_zf((zf-1)*floor(numpts/2)+1:(zf+1)*floor(numpts/2),:)=spect(1:2*floor(numpts/2));

tvect=ts*linspace(-zf*floor(numpts/2),zf*floor(numpts/2),zf*numpts);
echo=zf*ifftshift(ifft(fftshift(spect_zf)));

%figure(3);
%plot(tvect*1e6,real(echo),'b-'); hold on;
%plot(tvect*1e6,imag(echo),'r-');
%xlabel('Time (\mus)');