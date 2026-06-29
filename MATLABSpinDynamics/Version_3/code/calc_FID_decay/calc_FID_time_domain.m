% Calculate time domain echo from arbitrary magnetization
% Use zero-filling to get smoother echo shapes
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% Modified for arbitary magnetization, 09/09/19

% wvect -> offset frequency
% tacq -> acquisition window length, tdw -> dwell time
function [echo,tvect]=calc_FID_time_domain(mrx,wvect,tacq,tdw,plt)

T_180=pi/2; % Normalized T_180 time
numpts=length(mrx); % Number of isochromats

nacq=round(tacq/tdw)+1; % Number of acquired points
tvect=linspace(0,tacq,nacq); % Acquisition vector

echo=zeros(1,nacq);
for i=1:numpts
    echo=echo+mrx(i)*exp(1i*wvect(i)*tvect);
end

if plt
    figure;
    plot(tvect/T_180,real(echo),'b-'); hold on;
    plot(tvect/T_180,imag(echo),'r-');
    plot(tvect/T_180,abs(echo),'k-');
    set(gca,'FontSize',14);
    xlabel('Normalized time, t / T_{180}');
    title('Time-domain echo');
    xlim([-3 3]);
end