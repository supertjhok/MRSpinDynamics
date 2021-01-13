% Calculate time domain echo from spectrum
% Use zero-filling to get smoother echo shapes
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19

function [echo,tvect]=calc_time_domain_echo(spect,wvect,plt,savePlot)

zf=4; % Zero-filling ratio
T_180=pi; % Normalized T_180 time

ts=pi/(zf*max(wvect));
numpts=length(spect);

spect_zf=zeros(zf*numpts,1);
spect_zf((zf-1)*floor(numpts/2)+1:(zf+1)*floor(numpts/2),:)=spect(1:2*floor(numpts/2));

tvect=ts*linspace(-zf*floor(numpts/2),zf*floor(numpts/2),zf*numpts);
echo=zf*ifftshift(ifft(fftshift(spect_zf)));

if plt
    figure;
    plot(tvect/T_180,real(echo),'LineWidth',1); hold on;
    plot(tvect/T_180,imag(echo),'LineWidth',1);
    % plot(tvect/T_180,abs(echo),'k-');
    set(gca,'FontSize',14);
    xlabel('Normalized time, t / T_{180}');
    ylabel('Signal amplitude (a.u.)');
    title('Time-domain echo');
    legend('Real','Imaginary')
    xlim([-3 3]);
    set(gca,'FontSize',15); set(gca,'FontWeight','bold');
    if savePlot
%         export_fig D:\Dropbox\TuneMatchJMR\Figures\Updated\echoAsympTuned.pdf
    end
end